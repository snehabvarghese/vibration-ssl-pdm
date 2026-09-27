"""
Shared classification metrics and confusion-matrix plotting.

Macro averaging is used for precision/recall/F1 because the CWRU classes are
unbalanced (the Normal class has fewer recordings than the fault families
combined). Macro treats every fault class as equally important, which is the
right emphasis for maintenance: missing a rare fault is not cheaper than
missing a common one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    title: str,
    out_path: Path,
    normalize: bool = True,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    shown = cm.astype(float)
    if normalize:
        row = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, row, out=np.zeros_like(shown), where=row > 0)

    fig, ax = plt.subplots(figsize=(1 + 0.55 * len(class_names), 1 + 0.5 * len(class_names)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() if shown.size else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=70, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised" if normalize else "count")
    if len(class_names) <= 20:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if shown[i, j] > 0.005:
                    ax.text(j, i, f"{shown[i, j]:.2f}" if normalize else int(cm[i, j]),
                            ha="center", va="center", fontsize=5.5,
                            color="white" if shown[i, j] > shown.max() * 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def summarize_repeats(runs: List[Dict[str, float]]) -> Dict[str, float]:
    """mean/std across repeated label-subset draws."""
    keys = runs[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def sample_labelled_subset(
    labels: np.ndarray,
    fraction: float,
    seed: int,
    min_per_class: int = 1,
) -> np.ndarray:
    """Class-stratified random subset of indices, at least `min_per_class` each.

    Stratifying matters at 1 %: an unstratified draw would frequently miss
    whole classes and make the comparison meaningless.
    """
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n = max(min_per_class, int(round(fraction * len(idx))))
        n = min(n, len(idx))
        chosen.extend(rng.choice(idx, n, replace=False).tolist())
    return np.array(sorted(chosen))
