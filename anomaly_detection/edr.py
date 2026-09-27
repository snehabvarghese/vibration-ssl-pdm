"""
STEP 8a: Entropy Divergence Rate (EDR).

HONEST SCOPE STATEMENT
======================
The reference paper introduces a metric it calls the Entropy Divergence Rate
and states that it is based on KL divergence between the current embedding
distribution and a healthy-state reference. The full text of the paper was not
available to us (the supplied PDF was corrupt and only the abstract could be
recovered), so the exact estimator, normalisation and windowing it uses are
unknown.

Everything below is therefore **our own implementation of an EDR-style metric,
inspired by the paper's description**. It is NOT claimed to be numerically
equivalent to the paper's EDR, and our numbers must not be compared against
the paper's reported values.

MATHEMATICAL FORMULATION (what we actually compute)
===================================================
Let f(.) be the frozen self-supervised encoder and P a PCA projection to d
components fitted on healthy embeddings only.

1. Healthy reference. From healthy (Normal) training segments:

       H = { P f(x_i) : x_i healthy },   q = N(mu_H, Sigma_H)

   with mu_H, Sigma_H the sample mean/covariance (ridge-regularised:
   Sigma + eps*I, so it stays invertible when d is close to the sample count).

2. Current distribution. Segments arriving in order are grouped into
   consecutive windows W_t of `window_segments` segments with a stride:

       p_t = N(mu_t, Sigma_t)   estimated from { P f(x) : x in W_t }

3. Divergence. For two d-variate Gaussians the KL divergence is closed-form:

       D_KL(p_t || q) = 1/2 [ tr(Sigma_H^-1 Sigma_t)
                              + (mu_H - mu_t)^T Sigma_H^-1 (mu_H - mu_t)
                              - d + ln(det Sigma_H / det Sigma_t) ]

   We use the *reverse* direction D_KL(current || healthy) because it is
   large when the current data puts mass where the healthy model does not --
   exactly the "new behaviour appeared" case we want to flag. (Computed in
   log-determinant space via Cholesky for numerical stability.)

4. Rate. The paper's name says "rate", so we normalise the divergence per
   unit of observed time, which makes the value independent of the window
   length and of the sampling rate:

       EDR_t = D_KL(p_t || q) / T_window      [nats / second]

   where T_window = window_segments * hop_seconds.

5. Decision. A threshold is calibrated on HEALTHY VALIDATION data only:

       tau = mean(EDR_healthy_val) + n_sigma * std(EDR_healthy_val)

   A window is flagged anomalous when EDR_t > tau. No faulty data is used to
   set the threshold, which is what makes this usable when faults have never
   been observed.

ALTERNATIVE ESTIMATOR
---------------------
`estimator="histogram"` computes the sum over PCA components of the 1-D
histogram KL divergence with Laplace smoothing. It drops the Gaussian
assumption but ignores correlations between components; it is provided as a
robustness check, not as the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA

from config import CFG, EDRConfig


# --------------------------------------------------------------------------
# Low-level divergence estimators
# --------------------------------------------------------------------------
def _regularized_cov(x: np.ndarray, eps: float) -> np.ndarray:
    """Sample covariance with a ridge on the diagonal (keeps it invertible)."""
    if x.shape[0] < 2:
        return np.eye(x.shape[1]) * eps
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    return cov + eps * np.eye(cov.shape[0])


def gaussian_kl(
    mu_p: np.ndarray, cov_p: np.ndarray,
    mu_q: np.ndarray, cov_q: np.ndarray,
) -> float:
    """Closed-form D_KL( N(mu_p, cov_p) || N(mu_q, cov_q) ), in nats."""
    d = mu_p.shape[0]
    chol_q = np.linalg.cholesky(cov_q)
    # Solve with the Cholesky factor instead of inverting: stabler and faster.
    solved = np.linalg.solve(chol_q, np.linalg.solve(chol_q, cov_p).T).T
    trace_term = float(np.trace(solved))
    diff = (mu_q - mu_p).reshape(-1, 1)
    z = np.linalg.solve(chol_q, diff)
    quad_term = float((z * z).sum())
    logdet_q = 2.0 * float(np.log(np.diag(chol_q)).sum())
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    if sign_p <= 0:
        logdet_p = logdet_q  # degenerate window: neutralise the log-det term
    return 0.5 * (trace_term + quad_term - d + (logdet_q - logdet_p))


def histogram_kl(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int,
    edges: Optional[List[np.ndarray]] = None,
) -> float:
    """Sum of per-dimension 1-D histogram KL(p || q) with Laplace smoothing."""
    total = 0.0
    for j in range(p_samples.shape[1]):
        e = edges[j] if edges is not None else np.histogram_bin_edges(q_samples[:, j], bins)
        p_hist, _ = np.histogram(p_samples[:, j], bins=e)
        q_hist, _ = np.histogram(q_samples[:, j], bins=e)
        p = (p_hist + 1.0) / (p_hist.sum() + len(p_hist))
        q = (q_hist + 1.0) / (q_hist.sum() + len(q_hist))
        total += float(np.sum(p * np.log(p / q)))
    return total


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------
@dataclass
class EDRResult:
    """Per-window EDR output."""

    scores: np.ndarray            # (W,) EDR in nats/second
    window_index: np.ndarray      # (W,) index of the first segment in each window
    is_anomalous: np.ndarray      # (W,) bool, scores > threshold
    threshold: float


class EDRDetector:
    """EDR-style anomaly detector over a learned embedding space.

    Usage
    -----
        det = EDRDetector(hop_seconds=0.125)
        det.fit(healthy_train_embeddings)          # baseline q
        det.calibrate(healthy_val_embeddings)      # threshold tau
        res = det.score_stream(new_embeddings)     # EDR trend + flags
    """

    def __init__(
        self,
        cfg: Optional[EDRConfig] = None,
        hop_seconds: float = 1.0,
    ) -> None:
        self.cfg = cfg or CFG.edr
        self.hop_seconds = hop_seconds
        self.pca: Optional[PCA] = None
        self.mu_h: Optional[np.ndarray] = None
        self.cov_h: Optional[np.ndarray] = None
        self.healthy_proj: Optional[np.ndarray] = None
        self.hist_edges: Optional[List[np.ndarray]] = None
        self.threshold: float = float("inf")

    # -- baseline ---------------------------------------------------------
    def fit(self, healthy_embeddings: np.ndarray) -> "EDRDetector":
        """Fit PCA + the healthy reference distribution q. Healthy data only."""
        n, dim = healthy_embeddings.shape
        n_comp = int(min(self.cfg.pca_components, dim, max(1, n - 1)))
        self.pca = PCA(n_components=n_comp, random_state=CFG.seed)
        proj = self.pca.fit_transform(healthy_embeddings)
        self.healthy_proj = proj
        self.mu_h = proj.mean(axis=0)
        self.cov_h = _regularized_cov(proj, self.cfg.regularization)
        self.hist_edges = [
            np.histogram_bin_edges(proj[:, j], self.cfg.histogram_bins)
            for j in range(n_comp)
        ]
        return self

    def _check_fitted(self) -> None:
        if self.pca is None:
            raise RuntimeError("EDRDetector.fit() must be called with healthy data first")

    # -- scoring ----------------------------------------------------------
    def divergence(self, embeddings: np.ndarray) -> float:
        """Raw KL divergence (nats) of one batch of embeddings vs the baseline."""
        self._check_fitted()
        proj = self.pca.transform(embeddings)
        if self.cfg.estimator == "histogram":
            return histogram_kl(
                proj, self.healthy_proj, self.cfg.histogram_bins, self.hist_edges
            )
        mu_p = proj.mean(axis=0)
        cov_p = _regularized_cov(proj, self.cfg.regularization)
        return gaussian_kl(mu_p, cov_p, self.mu_h, self.cov_h)

    def score_stream(
        self,
        embeddings: np.ndarray,
        window: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> EDRResult:
        """Sliding-window EDR trend over an ordered sequence of embeddings."""
        self._check_fitted()
        window = window or self.cfg.window_segments
        stride = stride or self.cfg.window_stride
        n = embeddings.shape[0]
        if n < window:  # short stream: one window over everything we have
            window, stride = n, max(1, n)

        scores, starts = [], []
        for s in range(0, n - window + 1, stride):
            kl = self.divergence(embeddings[s : s + window])
            scores.append(kl / (window * self.hop_seconds))  # -> nats/second
            starts.append(s)

        sc = np.asarray(scores, dtype=float)
        return EDRResult(
            scores=sc,
            window_index=np.asarray(starts, dtype=int),
            is_anomalous=sc > self.threshold,
            threshold=self.threshold,
        )

    # -- threshold --------------------------------------------------------
    def calibrate(
        self,
        healthy_validation_embeddings: np.ndarray,
        n_sigma: Optional[float] = None,
    ) -> float:
        """tau = mean + n_sigma * std of EDR on HEALTHY validation windows."""
        self._check_fitted()
        n_sigma = self.cfg.threshold_n_sigma if n_sigma is None else n_sigma
        self.threshold = -np.inf  # score_stream must not filter during calibration
        res = self.score_stream(healthy_validation_embeddings)
        self.threshold = float(res.scores.mean() + n_sigma * res.scores.std())
        return self.threshold

    def summary(self) -> Dict[str, object]:
        return {
            "estimator": self.cfg.estimator,
            "pca_components": None if self.pca is None else int(self.pca.n_components_),
            "window_segments": self.cfg.window_segments,
            "window_stride": self.cfg.window_stride,
            "hop_seconds": self.hop_seconds,
            "threshold_n_sigma": self.cfg.threshold_n_sigma,
            "threshold": self.threshold,
            "note": "EDR here is our approximation inspired by the reference "
                    "paper, not a reproduction of its exact formulation.",
        }
