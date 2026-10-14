"""
Core analysis module decoupled from Streamlit.

Provides cached loaders for the encoder, health monitor, embeddings, and classifiers,
as well as the primary analyse_signal function for evaluating vibration data.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

# Root path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure __main__.HealthMonitor workaround for unpickling joblib checkpoints
import __main__
from anomaly_detection.anomaly_detector import HealthMonitor

if not hasattr(__main__, "HealthMonitor"):
    __main__.HealthMonitor = HealthMonitor

from config import CFG, CHECKPOINTS_DIR
from evaluation.visualization import load_embeddings
from models.classifier import build_classifier
from preprocessing.segmentation import preprocess_record
from training.extract_embeddings import embed_segments, load_encoder

MONITOR_PATH = CHECKPOINTS_DIR / "health_monitor.joblib"


# --------------------------------------------------------------------------
# Cached Artefact Loaders (functools.lru_cache)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_encoder():
    """Load and cache the frozen SSL encoder."""
    return load_encoder()


@lru_cache(maxsize=1)
def get_monitor():
    """Load and cache the fitted HealthMonitor."""
    if not MONITOR_PATH.exists():
        return None
    return joblib.load(MONITOR_PATH)


@lru_cache(maxsize=1)
def get_embeddings() -> Dict[str, np.ndarray]:
    """Load and cache precomputed benchmark embeddings and metadata."""
    return load_embeddings()


@lru_cache(maxsize=4)
def get_classifier(clf_name: str = "svm_rbf"):
    """Fit and cache downstream classifier on training embeddings."""
    d = get_embeddings()
    tr = d["split"].astype(str) == "train"
    clf = build_classifier(clf_name)
    clf.fit(d["embeddings"][tr], d["labels"][tr])
    return clf


@lru_cache(maxsize=1)
def get_map(max_points: int = 1500) -> Tuple[np.ndarray, np.ndarray, List[str], object]:
    """Project cached embeddings to 2D PCA space for visualization."""
    from sklearn.decomposition import PCA

    d = get_embeddings()
    emb, labels = d["embeddings"], d["labels"]
    rng = np.random.default_rng(CFG.seed)
    if len(emb) > max_points:
        sel = rng.choice(len(emb), max_points, replace=False)
        emb, labels = emb[sel], labels[sel]

    pca = PCA(n_components=2, random_state=CFG.seed).fit(d["embeddings"])
    return pca.transform(emb), labels, [str(c) for c in d["classes"]], pca


# --------------------------------------------------------------------------
# Core Signal Analysis Pipeline
# --------------------------------------------------------------------------
def analyse_signal(
    signal: np.ndarray,
    sampling_rate: int,
    clf_name: str = "svm_rbf",
) -> Optional[Dict[str, object]]:
    """raw vibration -> preprocessing -> embeddings -> health + fault verdict.

    Parameters
    ----------
    signal : np.ndarray
        1-D vibration acceleration signal array.
    sampling_rate : int
        Signal sampling rate in Hz.
    clf_name : str, default "svm_rbf"
        Classifier to use for fault categorization.

    Returns
    -------
    dict or None
        Analysis results dictionary matching dashboard schema, or None if signal
        is too short to produce valid segments.
    """
    segments, _ = preprocess_record(signal, sampling_rate)
    if len(segments) == 0:
        return None

    encoder = get_encoder()
    emb = embed_segments(encoder, segments)

    monitor = get_monitor()
    clf = get_classifier(clf_name)
    classes = [str(c) for c in get_embeddings()["classes"]]

    proba = clf.predict_proba(emb).mean(axis=0)  # average over segments
    pred_idx = int(np.argmax(proba))

    out: Dict[str, object] = {
        "n_segments": len(segments),
        "embeddings": emb,
        "predicted_class": classes[pred_idx],
        "confidence": float(proba[pred_idx]),
        "class_probabilities": dict(zip(classes, proba.tolist())),
    }

    if monitor is not None:
        trend = monitor.edr.score_stream(emb, stride=max(1, CFG.edr.window_stride // 4))
        seg_scores = monitor.segment_scores(emb)
        out.update({
            "edr_scores": trend.scores,
            "edr_mean": float(trend.scores.mean()),
            "edr_threshold": float(trend.threshold),
            "anomalous": bool(trend.scores.mean() > trend.threshold),
            "segment_scores": seg_scores,
            "segment_threshold": float(monitor.mahalanobis_threshold),
            "fraction_segments_flagged": float(
                (seg_scores > monitor.mahalanobis_threshold).mean()
            ),
        })
    return out
