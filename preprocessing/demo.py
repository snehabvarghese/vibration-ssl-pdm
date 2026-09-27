"""
STEP 2-3 visual check: what preprocessing and augmentation actually do.

Produces
  results/figures/02_preprocessing_stages.png  raw -> filtered -> window -> z-scored
  results/figures/03_augmented_views.png       one segment and two positive views

RUN
  python -m preprocessing.demo
"""

from __future__ import annotations

import sys

import numpy as np

from config import CFG, FIGURES_DIR, set_seed
from preprocessing.augmentation import make_two_views
from preprocessing.cwru_manifest import select_records
from preprocessing.filtering import downsample_signal, filter_signal
from preprocessing.loader import load_signal
from preprocessing.segmentation import normalize_signal, segment_signal


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(CFG.seed)
    rng = np.random.default_rng(CFG.seed)

    # A fault record makes the effect of each stage easy to see.
    rec = next(r for r in select_records() if r.label() == "IR007" and r.load_hp == 0)
    sig = load_signal(rec)

    filtered = filter_signal(sig.signal, sig.sampling_rate)
    reduced, fs = downsample_signal(filtered, sig.sampling_rate)
    window = CFG.preprocess.window_size(sig.sampling_rate)
    segments = segment_signal(reduced, window, hop_size=CFG.preprocess.hop_size(sig.sampling_rate))
    normed = normalize_signal(segments)

    # ---- figure 1: pipeline stages --------------------------------------
    n_show = int(0.1 * sig.sampling_rate)
    stages = [
        ("1. Raw", sig.signal[:n_show], sig.sampling_rate),
        (f"2. Band-pass {CFG.preprocess.filter_low_hz:.0f}-{CFG.preprocess.filter_high_hz:.0f} Hz",
         filtered[:n_show], sig.sampling_rate),
        (f"3. One window ({window} samples @ {fs} Hz)", segments[0], fs),
        ("4. Z-scored window (model input)", normed[0], fs),
    ]
    fig, axes = plt.subplots(len(stages), 1, figsize=(11, 8))
    for ax, (title, data, rate) in zip(axes, stages, strict=True):
        ax.plot(np.arange(len(data)) / rate, data, linewidth=0.7, color="#4C72B0")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("a", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Preprocessing stages -- record {rec.file_id} ({rec.label()})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES_DIR / "02_preprocessing_stages.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: positive pair ----------------------------------------
    base = normed[0]
    view_a, view_b = make_two_views(base, rng=rng)
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, (title, data, color) in zip(
        axes,
        [
            ("Original z-scored segment", base, "#4C72B0"),
            ("Augmented view A", view_a, "#55A868"),
            ("Augmented view B", view_b, "#C44E52"),
        ],
        strict=True,
    ):
        ax.plot(np.arange(len(data)) / fs, data, linewidth=0.7, color=color)
        ax.set_title(title, fontsize=9, loc="left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Positive pair for contrastive learning (same segment, two views)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "03_augmented_views.png", dpi=150)
    plt.close(fig)

    print(f"record            : {rec.file_id} ({rec.label()})")
    print(f"raw samples       : {len(sig.signal)} @ {sig.sampling_rate} Hz")
    print(f"segments          : {segments.shape}")
    print(f"z-scored segment  : mean={normed[0].mean():+.2e}  std={normed[0].std():.4f}")
    print(f"view A vs original: corr={np.corrcoef(base, view_a)[0, 1]:.3f}")
    print(f"view B vs original: corr={np.corrcoef(base, view_b)[0, 1]:.3f}")
    print(f"figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
