"""
STEP 8b: end-to-end anomaly detection built on the EDR metric.

WHAT IT DOES
------------
1. Fits the healthy baseline on a "commissioning period" of Normal data.
2. Calibrates the decision threshold on a later, non-overlapping healthy
   period (still healthy, never faulty).
3. Scores held-out data:
     * per-window EDR over a constructed "degradation timeline" that starts
       healthy and transitions into a fault -- this produces the anomaly
       TREND the dashboard shows;
     * per-class EDR on the test set, which answers "does the score rise for
       every fault type, and stay low for unseen healthy data?".
4. Also reports a per-segment Mahalanobis distance to the healthy baseline,
   because the dashboard needs a score for a SINGLE uploaded window (EDR is
   defined over a window of segments, not a single one).

WHY THE HEALTHY SPLIT IS CHRONOLOGICAL, NOT RECORD-LEVEL
--------------------------------------------------------
The classification experiments split at record level. That is the right
protocol there, but it is the WRONG protocol for a healthy baseline: CWRU has
exactly four Normal recordings, one per motor load, so a record-level split
would fit the baseline at some loads and test it at loads it had never seen.
We measured that directly -- it flags 100 % of healthy test segments, because
the embedding shift caused by changing load is larger than the spread within
one load.

That is an artefact of the protocol, not of the method. A real deployment
commissions the monitor by recording the machine while it is known to be
healthy, across its normal operating range, and only then starts monitoring.
We mirror that: within each Normal recording, the first 50 % of segments are
the commissioning baseline, the next 25 % calibrate the threshold, and the
last 25 % are held-out healthy data used for evaluation. A one-segment gap is
inserted at each boundary so that 50 %-overlapping windows never straddle it.
Faulty evaluation data still comes from the held-out TEST records, which the
encoder and the detector have never seen.

WHY A CONSTRUCTED TIMELINE
------------------------
CWRU recordings are steady-state: each file is entirely healthy or entirely
faulty, so there is no real run-to-failure degradation in the dataset. To
demonstrate trend monitoring we concatenate held-out healthy segments
followed by faulty ones. This is clearly an illustrative construction, not a
measurement of real degradation -- it is labelled as such in the figure and in
the metrics file. Phase 2, with continuously logged machine data, is where a
real degradation trend can be measured.

OUTPUTS
-------
  results/metrics/anomaly_detection.json
  results/figures/07_edr_trend.png
  results/figures/08_edr_by_class.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import joblib

from anomaly_detection.edr import EDRDetector, _regularized_cov
from config import CFG, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, set_seed
from evaluation.visualization import load_embeddings

HEALTHY_CLASS = "Normal"


class HealthMonitor:
    """EDR window score + per-segment Mahalanobis score against one baseline."""

    def __init__(self, hop_seconds: float) -> None:
        self.edr = EDRDetector(hop_seconds=hop_seconds)
        self._inv_cov: Optional[np.ndarray] = None
        self._mu: Optional[np.ndarray] = None
        self.mahalanobis_threshold: float = float("inf")

    def fit(self, healthy_embeddings: np.ndarray) -> "HealthMonitor":
        self.edr.fit(healthy_embeddings)
        proj = self.edr.pca.transform(healthy_embeddings)
        self._mu = proj.mean(axis=0)
        self._inv_cov = np.linalg.inv(_regularized_cov(proj, CFG.edr.regularization))
        return self

    def calibrate(self, healthy_val_embeddings: np.ndarray) -> Dict[str, float]:
        edr_tau = self.edr.calibrate(healthy_val_embeddings)
        d = self.segment_scores(healthy_val_embeddings)
        self.mahalanobis_threshold = float(np.percentile(d, 99))
        return {"edr_threshold": edr_tau, "mahalanobis_threshold": self.mahalanobis_threshold}

    def segment_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of each segment to the healthy centre."""
        proj = self.edr.pca.transform(embeddings)
        diff = proj - self._mu
        return np.einsum("ij,jk,ik->i", diff, self._inv_cov, diff)


def plot_trend(
    scores: np.ndarray,
    threshold: float,
    switch_window: int,
    hop_seconds: float,
    stride: int,
    out_path: Path,
) -> None:
    t = np.arange(len(scores)) * stride * hop_seconds
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, scores, marker="o", ms=3, color="#4C72B0", label="EDR (nats/s)")
    ax.axhline(threshold, color="#C44E52", ls="--",
               label=f"threshold (healthy mean + {CFG.edr.threshold_n_sigma:g}$\\sigma$)")
    ax.axvline(switch_window * stride * hop_seconds, color="grey", ls=":",
               label="healthy -> faulty transition")
    ax.set_xlabel("time along the constructed timeline (s)")
    ax.set_ylabel("EDR (nats / s)")
    ax.set_yscale("symlog")
    ax.set_title("EDR anomaly trend (illustrative timeline: held-out healthy then faulty segments)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_by_class(per_class: Dict[str, float], threshold: float, out_path: Path) -> None:
    names = sorted(per_class, key=lambda k: (k != HEALTHY_CLASS, k))
    vals = [per_class[n] for n in names]
    colors = ["#55A868" if n == HEALTHY_CLASS else "#C44E52" for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, vals, color=colors)
    ax.axhline(threshold, color="black", ls="--", lw=1, label="anomaly threshold")
    ax.set_yscale("symlog")
    ax.set_ylabel("mean EDR (nats / s)")
    ax.set_title("EDR per test class (green = healthy baseline class)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def chronological_healthy_split(
    record_ids: np.ndarray,
    fit_frac: float = 0.5,
    calib_frac: float = 0.25,
    gap: int = 1,
) -> Dict[str, np.ndarray]:
    """Split each healthy recording into commissioning / calibration / held-out.

    `record_ids` must be the ids of the healthy segments, in acquisition order.
    A `gap` of one segment is dropped at each boundary so that no 50 %-
    overlapping pair straddles two parts.
    """
    parts: Dict[str, List[int]] = {"fit": [], "calib": [], "heldout": []}
    for rid in np.unique(record_ids):
        idx = np.where(record_ids == rid)[0]        # already chronological
        n = len(idx)
        a = int(n * fit_frac)
        b = int(n * (fit_frac + calib_frac))
        parts["fit"].extend(idx[: max(1, a - gap)].tolist())
        parts["calib"].extend(idx[a : max(a + 1, b - gap)].tolist())
        parts["heldout"].extend(idx[b:].tolist())
    return {k: np.array(sorted(v), dtype=int) for k, v in parts.items()}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EDR-based anomaly detection")
    parser.add_argument("--estimator", type=str, default=CFG.edr.estimator,
                        choices=["gaussian", "histogram"])
    args = parser.parse_args(argv)
    CFG.edr.estimator = args.estimator

    set_seed(CFG.seed)
    d = load_embeddings()
    emb, labels, split = d["embeddings"], d["labels"], d["split"].astype(str)
    classes = [str(c) for c in d["classes"]]
    healthy_id = classes.index(HEALTHY_CLASS)

    hop_seconds = CFG.preprocess.window_seconds * (1.0 - CFG.preprocess.overlap)

    # --- healthy data: chronological commissioning split (see module docstring)
    healthy_mask = labels == healthy_id
    healthy_emb = emb[healthy_mask]
    healthy_records = d["record_ids"].astype(str)[healthy_mask]
    if len(healthy_emb) == 0:
        raise RuntimeError("no healthy segments found -- check the dataset build")
    hs = chronological_healthy_split(healthy_records)

    monitor = HealthMonitor(hop_seconds).fit(healthy_emb[hs["fit"]])
    thresholds = monitor.calibrate(healthy_emb[hs["calib"]])
    print(f"healthy commissioning fit: {len(hs['fit'])} segments | "
          f"calibration: {len(hs['calib'])} | held-out healthy: {len(hs['heldout'])}")
    print(f"EDR threshold = {thresholds['edr_threshold']:.4f} nats/s")

    healthy_eval = healthy_emb[hs["heldout"]]
    faulty_mask = (split == "test") & (labels != healthy_id)
    faulty_eval, faulty_labels = emb[faulty_mask], labels[faulty_mask]

    # --- 1. constructed degradation timeline ----------------------------
    rng = np.random.default_rng(CFG.seed)
    n_each = min(len(healthy_eval), len(faulty_eval))
    faulty_pick = faulty_eval[np.sort(rng.choice(len(faulty_eval), n_each, replace=False))]
    timeline = np.concatenate([healthy_eval[:n_each], faulty_pick], axis=0)
    # A finer stride than the default is used here purely so the trend figure
    # has enough points to read; the window length is unchanged.
    trend_stride = max(1, CFG.edr.window_stride // 4)
    trend = monitor.edr.score_stream(timeline, stride=trend_stride)
    # First window that contains any faulty segment.
    switch_window = int(np.searchsorted(
        trend.window_index, n_each - CFG.edr.window_segments + 1, side="left"))
    plot_trend(trend.scores, trend.threshold, switch_window, hop_seconds,
               trend_stride, FIGURES_DIR / "07_edr_trend.png")

    # --- 2. per-class EDR -----------------------------------------------
    per_class: Dict[str, float] = {HEALTHY_CLASS: float(
        monitor.edr.score_stream(healthy_eval).scores.mean())}
    for c, name in enumerate(classes):
        if c == healthy_id:
            continue
        e = faulty_eval[faulty_labels == c]
        if len(e) < 2:
            continue
        per_class[name] = float(monitor.edr.score_stream(e).scores.mean())
    plot_by_class(per_class, trend.threshold, FIGURES_DIR / "08_edr_by_class.png")

    # --- 3. healthy-vs-faulty detection quality at segment level --------
    eval_emb = np.concatenate([healthy_eval, faulty_eval], axis=0)
    is_fault = np.concatenate([np.zeros(len(healthy_eval), int),
                               np.ones(len(faulty_eval), int)])
    seg_scores = monitor.segment_scores(eval_emb)
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(is_fault, seg_scores)) if is_fault.min() != is_fault.max() else float("nan")
    flagged = seg_scores > monitor.mahalanobis_threshold
    tpr = float(flagged[is_fault == 1].mean())
    fpr = float(flagged[is_fault == 0].mean())

    metrics = {
        "detector": monitor.edr.summary(),
        "hop_seconds": hop_seconds,
        "healthy_protocol": "chronological per-recording split of the Normal "
                            "recordings (50% commissioning / 25% calibration / "
                            "25% held-out), one-segment gap at each boundary",
        "n_healthy_fit": int(len(hs["fit"])),
        "n_healthy_calibration": int(len(hs["calib"])),
        "n_healthy_heldout": int(len(hs["heldout"])),
        "n_faulty_eval": int(len(faulty_eval)),
        "edr_threshold_nats_per_s": trend.threshold,
        "timeline": {
            "description": "held-out healthy segments followed by faulty segments "
                           "from unseen test records; illustrative, CWRU has no "
                           "run-to-failure data",
            "n_healthy": int(n_each),
            "n_faulty": int(n_each),
            "window_stride_used": trend_stride,
            "switch_window": switch_window,
            "scores": trend.scores.tolist(),
            "flagged": trend.is_anomalous.tolist(),
            "windows_flagged_before_switch": int(trend.is_anomalous[:switch_window].sum()),
            "windows_flagged_after_switch": int(trend.is_anomalous[switch_window:].sum()),
            "windows_total_before_switch": int(switch_window),
            "windows_total_after_switch": int(len(trend.scores) - switch_window),
        },
        "edr_per_test_class_nats_per_s": per_class,
        "segment_level": {
            "score": "squared Mahalanobis distance to the healthy baseline (PCA space)",
            "roc_auc_fault_vs_healthy": auc,
            "threshold_99th_pct_healthy_calibration": monitor.mahalanobis_threshold,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
        },
    }
    (METRICS_DIR / "anomaly_detection.json").write_text(json.dumps(metrics, indent=2))

    # The dashboard needs the fitted baseline, not just the numbers.
    joblib.dump(monitor, CHECKPOINTS_DIR / "health_monitor.joblib")

    print(f"timeline: {metrics['timeline']['windows_flagged_before_switch']}"
          f"/{switch_window} healthy windows flagged, "
          f"{metrics['timeline']['windows_flagged_after_switch']}"
          f"/{len(trend.scores) - switch_window} faulty windows flagged")
    print(f"segment-level ROC-AUC (fault vs healthy): {auc:.3f}  TPR {tpr:.3f}  FPR {fpr:.3f}")
    print(f"metrics -> {METRICS_DIR / 'anomaly_detection.json'}")
    print(f"monitor -> {CHECKPOINTS_DIR / 'health_monitor.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
