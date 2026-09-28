"""
STEP 11: comparison of the proposed pipeline against the supervised baseline.

Reads the metric files written by:
    training/train_classifier.py   (SSL embeddings + SVM / MLP)
    training/train_baseline.py     (supervised CNN from scratch)
and produces a single comparison table + figure.

WHAT THE COMPARISON MEANS
-------------------------
Both arms see exactly the same labelled subsets, the same record-level split
and the same test segments. The only asymmetry is that the SSL arm additionally
exploited the *unlabelled* training segments during pretraining. So the gap at
small label fractions is the quantity the project is actually about.

We report our own measured numbers only. The reference paper's F1 of 0.93 was
obtained on different data with a different protocol and is not a target here.

OUTPUTS
-------
  results/metrics/comparison.json
  results/metrics/comparison.csv
  results/figures/12_ssl_vs_baseline.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, METRICS_DIR

FINETUNE_FILE = METRICS_DIR / "finetune_classification.json"


def _require(path: Path, how: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `{how}` first")
    return json.loads(path.read_text())


def build_rows(ssl: dict, base: dict, finetune: Optional[dict] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for clf, by_frac in ssl["results"].items():
        for frac, res in by_frac.items():
            rows.append({
                "method": f"SSL + {clf}",
                "label_fraction": float(frac),
                "n_labelled": res["n_labelled"],
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "precision_macro_mean": res["precision_macro_mean"],
                "recall_macro_mean": res["recall_macro_mean"],
                "f1_macro_mean": res["f1_macro_mean"],
                "f1_macro_std": res["f1_macro_std"],
            })
    if finetune and "results" in finetune:
        for frac, res in finetune["results"].items():
            rows.append({
                "method": "SSL Fine-tuned",
                "label_fraction": float(frac),
                "n_labelled": res["n_labelled"],
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "precision_macro_mean": res["precision_macro_mean"],
                "recall_macro_mean": res["recall_macro_mean"],
                "f1_macro_mean": res["f1_macro_mean"],
                "f1_macro_std": res["f1_macro_std"],
            })
    for frac, res in base["results"].items():
        rows.append({
            "method": "Supervised CNN (baseline)",
            "label_fraction": float(frac),
            "n_labelled": res["n_labelled"],
            "accuracy_mean": res["accuracy_mean"],
            "accuracy_std": res["accuracy_std"],
            "precision_macro_mean": res["precision_macro_mean"],
            "recall_macro_mean": res["recall_macro_mean"],
            "f1_macro_mean": res["f1_macro_mean"],
            "f1_macro_std": res["f1_macro_std"],
        })
    return sorted(rows, key=lambda r: (r["label_fraction"], r["method"]))


def plot_comparison(rows: List[Dict[str, object]], out_path: Path) -> None:
    methods = sorted({str(r["method"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in methods:
        pts = sorted([r for r in rows if r["method"] == m], key=lambda r: r["label_fraction"])
        x = [float(r["label_fraction"]) * 100 for r in pts]
        y = [float(r["f1_macro_mean"]) for r in pts]
        e = [float(r["f1_macro_std"]) for r in pts]
        style = dict(marker="s", ls="--") if "baseline" in m else dict(marker="o", ls="-")
        ax.errorbar(x, y, yerr=e, capsize=3, label=m, **style)
    ax.set_xscale("log")
    ax.set_xlabel("labelled fraction of the training split (%)")
    ax.set_ylabel("macro F1 on the held-out test records")
    ax.set_title("Self-supervised pretraining vs supervised training from scratch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


SSL_FILE = METRICS_DIR / "downstream_classification.json"
BASE_FILE = METRICS_DIR / "baseline_classification.json"
FINETUNE_FILE = METRICS_DIR / "finetune_classification.json"


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(description="Compare SSL pipeline with the baseline").parse_args(argv)

    ssl = _require(SSL_FILE, "python -m training.train_classifier")
    base = _require(BASE_FILE, "python -m training.train_baseline")
    finetune = json.loads(FINETUNE_FILE.read_text()) if FINETUNE_FILE.exists() else None
    rows = build_rows(ssl, base, finetune)

    header = f"{'method':28s} {'labels%':>8s} {'n':>5s} {'acc':>14s} {'macroF1':>14s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['method']:28s} {float(r['label_fraction'])*100:7.1f}% {r['n_labelled']:5d} "
              f"{r['accuracy_mean']:.3f}+-{r['accuracy_std']:.3f}  "
              f"{r['f1_macro_mean']:.3f}+-{r['f1_macro_std']:.3f}")

    # Best SSL variant vs baseline at each fraction.
    deltas: Dict[str, Dict[str, float]] = {}
    for frac in sorted({float(r["label_fraction"]) for r in rows}):
        at = [r for r in rows if float(r["label_fraction"]) == frac]
        b = next(r for r in at if "baseline" in str(r["method"]))
        best = max((r for r in at if "baseline" not in str(r["method"])),
                   key=lambda r: float(r["f1_macro_mean"]))
        deltas[f"{frac}"] = {
            "best_ssl_method": str(best["method"]),
            "ssl_f1_macro": float(best["f1_macro_mean"]),
            "baseline_f1_macro": float(b["f1_macro_mean"]),
            "absolute_gain": float(best["f1_macro_mean"]) - float(b["f1_macro_mean"]),
            "relative_gain_pct": (
                100.0 * (float(best["f1_macro_mean"]) - float(b["f1_macro_mean"]))
                / max(1e-9, float(b["f1_macro_mean"]))
            ),
        }

    print("\nbest SSL vs baseline (macro F1):")
    for frac, dd in deltas.items():
        print(f"  {float(frac)*100:5.1f}%  {dd['ssl_f1_macro']:.3f} vs {dd['baseline_f1_macro']:.3f}"
              f"  ({dd['absolute_gain']:+.3f}, {dd['relative_gain_pct']:+.1f}%)")

    (METRICS_DIR / "comparison.json").write_text(json.dumps(
        {"rows": rows, "ssl_vs_baseline": deltas,
         "note": "all values measured in this repository; the reference paper's "
                 "F1 of 0.93 refers to different data and is not a target"},
        indent=2))
    with (METRICS_DIR / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_comparison(rows, FIGURES_DIR / "12_ssl_vs_baseline.png")
    print(f"\nwritten: {METRICS_DIR/'comparison.json'}, {METRICS_DIR/'comparison.csv'}, "
          f"{FIGURES_DIR/'12_ssl_vs_baseline.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
