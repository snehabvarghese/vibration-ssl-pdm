"""
Generates the four presentation notebooks from the modules that already exist.

The notebooks are thin on purpose: every heavy step lives in a tested module,
and the notebook only narrates it and shows the output. Regenerate with:

    python notebooks/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent


def nb(cells: List[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


SETUP = code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))   # run from notebooks/
%matplotlib inline
""")


NOTEBOOKS = {
    "01_data_exploration.ipynb": nb([
        md("""
# 01 — Dataset exploration

The CWRU bearing dataset: what is in it, how the files differ, and why we
resample everything to 12 kHz. Run `python -m preprocessing.download_cwru --all`
first.
"""),
        SETUP,
        code("""
from preprocessing.cwru_manifest import select_records
records = select_records()
print(len(records), "records")
for r in records[:5]:
    print(r.file_id, r.label(), r.load_hp, "hp", r.sampling_rate, "Hz")
"""),
        md("The Normal recordings are natively 48 kHz while the fault files are "
           "12 kHz, so the loader resamples before anything else touches the signal."),
        code("""
from preprocessing.loader import load_signal
normal = load_signal([r for r in records if r.fault_type == "Normal"][0])
fault  = load_signal([r for r in records if r.fault_type != "Normal"][0])
for s in (normal, fault):
    print(f"{s.record_id:6s} {s.label:10s} {len(s.signal):7d} samples @ {s.sampling_rate} Hz")
"""),
        code("""
import matplotlib.pyplot as plt, numpy as np
fig, axes = plt.subplots(2, 1, figsize=(11, 4), sharex=True)
for ax, s in zip(axes, (normal, fault), strict=True):
    t = np.arange(3000) / s.sampling_rate
    ax.plot(t, s.signal[:3000], lw=.7); ax.set_title(s.label, loc="left", fontsize=9)
axes[-1].set_xlabel("time (s)"); plt.tight_layout()
"""),
        md("The full report and figures come from the module:"),
        code("!cd .. && PYTHONPATH=. python -m preprocessing.explore"),
    ]),

    "02_preprocessing.ipynb": nb([
        md("""
# 02 — Preprocessing and augmentation

Raw signal → band-pass → 0.25 s windows @ 50 % overlap → per-segment z-score →
two augmented views (the positive pair for contrastive learning).
"""),
        SETUP,
        code("""
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import load_signal
from preprocessing.filtering import filter_signal
from preprocessing.segmentation import preprocess_record
from config import CFG

sig = load_signal(select_records()[0])
filtered = filter_signal(sig.signal, sig.sampling_rate)
segments, rate = preprocess_record(sig.signal, sig.sampling_rate)
print("band-pass", CFG.preprocess.filter_low_hz, "-", CFG.preprocess.filter_high_hz, "Hz")
print("segments:", segments.shape, "@", rate, "Hz")
print("per-segment mean/std:", segments[0].mean().round(12), segments[0].std().round(6))
"""),
        md("""
Why 10–5000 Hz and no decimation: an earlier version decimated by 4, which put
Nyquist at 1.5 kHz and discarded the 2–4 kHz bearing resonances that carry most
of the fault signature.
"""),
        code("""
import numpy as np, matplotlib.pyplot as plt
from preprocessing.augmentation import make_two_views
rng = np.random.default_rng(0)
a, b = make_two_views(segments[0], rng=rng)
fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True, sharey=True)
for ax, (name, x) in zip(axes, [("original", segments[0]), ("view A", a), ("view B", b)],
                         strict=True):
    ax.plot(x, lw=.7); ax.set_title(name, loc="left", fontsize=9)
plt.tight_layout()
print("corr(view A, original) =", round(float(np.corrcoef(a, segments[0])[0, 1]), 3))
"""),
        md("""
The cached segment array for the whole dataset is built once from the shell:

```bash
PYTHONPATH=. python -m preprocessing.build_dataset
```
"""),
    ]),

    "03_ssl_training.ipynb": nb([
        md("""
# 03 — Self-supervised (contrastive) training

Two augmented views of the same segment are a positive pair; everything else in
the batch is a negative. **No fault labels are used anywhere in this notebook.**
"""),
        SETUP,
        code("""
import torch
from models.siamese import SiameseContrastiveModel, NTXentLoss
model = SiameseContrastiveModel()
print("encoder parameters:", model.encoder.n_parameters())
x = torch.randn(8, 3000)
za, zb = model(x, x)
print("projection output:", tuple(za.shape), "| embedding output:", tuple(model.embed(x).shape))
print("NT-Xent on aligned views:", float(NTXentLoss()(za, zb)))
"""),
        md("""
The full run is launched from the shell so that the notebook never overwrites a
trained checkpoint:

```bash
PYTHONPATH=. python -m training.train_ssl --epochs 150
```

The curve below is the run whose checkpoint the rest of the pipeline uses.
"""),
        code("""
import json
hist = json.load(open("../results/metrics/ssl_training.json"))
import matplotlib.pyplot as plt
plt.plot(hist["history"]["train_loss"], label="train")
plt.plot(hist["history"]["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("NT-Xent"); plt.legend(); plt.grid(alpha=.3)
"""),
    ]),

    "04_evaluation.ipynb": nb([
        md("""
# 04 — Embeddings, anomaly detection and the comparison

Reads the artefacts produced by the pipeline; it does not retrain anything.
"""),
        SETUP,
        code("""
from evaluation.visualization import load_embeddings
d = load_embeddings()
print("embeddings:", d["embeddings"].shape)
import collections; print(collections.Counter(d["split"].astype(str)))
"""),
        code("""
import json
edr = json.load(open("../results/metrics/anomaly_detection.json"))
print("EDR threshold:", round(edr["edr_threshold_nats_per_s"], 3), "nats/s")
print("healthy windows flagged:", edr["timeline"]["windows_flagged_before_switch"],
      "/", edr["timeline"]["windows_total_before_switch"])
print("faulty windows flagged :", edr["timeline"]["windows_flagged_after_switch"],
      "/", edr["timeline"]["windows_total_after_switch"])
print("segment ROC-AUC:", round(edr["segment_level"]["roc_auc_fault_vs_healthy"], 3))
"""),
        md("""
The healthy baseline is fitted on a *chronological* commissioning portion of each
Normal recording, not on a record-level split: CWRU has one Normal recording per
motor load, so a record split makes the detector call unseen healthy loads faulty.
"""),
        code("""
import pandas as pd
df = pd.read_csv("../results/metrics/comparison.csv")
df.pivot_table(index="label_fraction", columns="method", values="f1_macro_mean").round(3)
"""),
        code("""
from IPython.display import Image, display
for f in ["12_ssl_vs_baseline.png", "05_tsne_embeddings.png", "07_edr_trend.png"]:
    display(Image(filename=f"../results/figures/{f}"))
"""),
    ]),
}


def main() -> None:
    for name, content in NOTEBOOKS.items():
        (HERE / name).write_text(json.dumps(content, indent=1))
        print("wrote", HERE / name)


if __name__ == "__main__":
    main()
