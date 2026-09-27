"""
STEP 9: fault classification from frozen embeddings, using few labels.

EXPERIMENT
----------
For each labelled fraction (1 %, 5 %, 10 %, 20 % of the TRAINING segments) and
each classifier (RBF-SVM, MLP):

    sample a stratified labelled subset  ->  fit on its embeddings
                                         ->  evaluate on the full TEST split

The encoder is frozen: no gradient flows into it here. The label fraction is
applied to the training split only; the test split is always evaluated in
full, and comes from different recordings (record-level split), so nothing
leaks.

Each (fraction, classifier) cell is repeated `n_repeats` times with different
random label draws, and we report mean +/- std -- with 1 % of the data a
single draw is mostly luck.

OUTPUTS
-------
  results/metrics/downstream_classification.json
  results/figures/09_label_efficiency.png
  results/figures/10_confusion_ssl_<clf>.png   (best fraction)
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

from config import CFG, FIGURES_DIR, METRICS_DIR, set_seed
from evaluation.metrics import (
    classification_metrics,
    plot_confusion,
    sample_labelled_subset,
    summarize_repeats,
)
from evaluation.visualization import load_embeddings
from models.classifier import build_classifier


def evaluate_fraction(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    emb_test: np.ndarray,
    y_test: np.ndarray,
    clf_name: str,
    fraction: float,
    n_repeats: int,
) -> Dict[str, object]:
    runs, last_pred, n_used = [], None, 0
    for rep in range(n_repeats):
        seed = CFG.seed + 1000 * rep
        sub = sample_labelled_subset(y_train, fraction, seed)
        n_used = len(sub)
        clf = build_classifier(clf_name, seed=seed)
        clf.fit(emb_train[sub], y_train[sub])
        pred = clf.predict(emb_test)
        runs.append(classification_metrics(y_test, pred))
        last_pred = pred
    out = summarize_repeats(runs)
    out["n_labelled"] = int(n_used)
    out["per_run"] = runs
    return out, last_pred


def plot_label_efficiency(results: Dict[str, Dict[str, dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for clf_name, by_frac in results.items():
        fracs = sorted(float(f) for f in by_frac)
        means = [by_frac[f"{f}"]["f1_macro_mean"] for f in fracs]
        stds = [by_frac[f"{f}"]["f1_macro_std"] for f in fracs]
        ax.errorbar([f * 100 for f in fracs], means, yerr=stds, marker="o",
                    capsize=3, label=f"SSL embeddings + {clf_name}")
    ax.set_xlabel("labelled fraction of the training split (%)")
    ax.set_ylabel("macro F1 on the test split")
    ax.set_title("Label efficiency of the self-supervised representation")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Limited-label downstream classification")
    parser.add_argument("--fractions", type=float, nargs="*", default=CFG.downstream.label_fractions)
    parser.add_argument("--classifiers", type=str, nargs="*", default=CFG.downstream.classifiers)
    parser.add_argument("--repeats", type=int, default=CFG.downstream.n_repeats)
    args = parser.parse_args(argv)

    set_seed(CFG.seed)
    d = load_embeddings()
    emb, labels, split = d["embeddings"], d["labels"], d["split"].astype(str)
    classes = [str(c) for c in d["classes"]]

    tr, te = split == "train", split == "test"
    emb_train, y_train = emb[tr], labels[tr]
    emb_test, y_test = emb[te], labels[te]
    print(f"train {len(y_train)} segments | test {len(y_test)} segments | {len(classes)} classes")

    results: Dict[str, Dict[str, dict]] = {}
    for clf_name in args.classifiers:
        results[clf_name] = {}
        for frac in args.fractions:
            res, pred = evaluate_fraction(
                emb_train, y_train, emb_test, y_test, clf_name, frac, args.repeats
            )
            results[clf_name][f"{frac}"] = res
            print(f"{clf_name:8s} {frac*100:5.1f}%  n={res['n_labelled']:4d}  "
                  f"acc {res['accuracy_mean']:.3f}+-{res['accuracy_std']:.3f}  "
                  f"F1 {res['f1_macro_mean']:.3f}+-{res['f1_macro_std']:.3f}")
            if frac == max(args.fractions):
                plot_confusion(
                    y_test, pred, classes,
                    f"SSL embeddings + {clf_name} ({frac*100:g}% labels)",
                    FIGURES_DIR / f"10_confusion_ssl_{clf_name}.png",
                )

    plot_label_efficiency(results, FIGURES_DIR / "09_label_efficiency.png")
    payload = {
        "setup": {
            "encoder": "frozen self-supervised 1-D CNN",
            "split_strategy": CFG.split.strategy,
            "n_train_segments": int(len(y_train)),
            "n_test_segments": int(len(y_test)),
            "classes": classes,
            "repeats": args.repeats,
        },
        "results": results,
    }
    (METRICS_DIR / "downstream_classification.json").write_text(json.dumps(payload, indent=2))
    print(f"metrics -> {METRICS_DIR / 'downstream_classification.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
