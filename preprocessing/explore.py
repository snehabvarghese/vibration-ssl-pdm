"""
STEP 1 deliverable: dataset exploration.

WHAT IT DOES
------------
Loads every selected record and reports, from the data itself (nothing
hard-coded):
  * number of records and total samples / total duration
  * class list and class distribution (by record and by second of signal)
  * sampling rate(s), native and after resampling
  * signal lengths
  * basic per-class statistics (RMS, kurtosis, peak) -- classic vibration
    health indicators, useful as a sanity check that the classes really differ
  * missing / unreadable files

OUTPUTS
-------
  results/metrics/dataset_summary.json   machine-readable summary
  results/figures/01_class_distribution.png
  results/figures/01_example_waveforms.png
  results/figures/01_example_spectra.png

RUN
---
  python -m preprocessing.explore
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import CFG, DATA_RAW_DIR, FIGURES_DIR, METRICS_DIR, set_seed
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import LoadedSignal, SignalLoadError, load_signal


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def signal_statistics(x: np.ndarray) -> Dict[str, float]:
    """Classic time-domain vibration health indicators.

    rms      overall energy
    peak     largest absolute acceleration
    crest    peak / rms -- sensitive to impulsive (bearing) faults
    kurtosis 4th standardized moment -- high for impulsive signals
    """
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    std = float(np.std(x))
    kurt = float(np.mean((x - x.mean()) ** 4) / (std**4)) if std > 0 else float("nan")
    return {
        "mean": float(x.mean()),
        "std": std,
        "rms": rms,
        "peak": peak,
        "crest_factor": float(peak / rms) if rms > 0 else float("nan"),
        "kurtosis": kurt,
    }


def summarize(signals: List[LoadedSignal], errors: List[str]) -> Dict[str, object]:
    """Build the machine-readable dataset summary."""
    per_class_records: Counter = Counter()
    per_class_seconds: Dict[str, float] = defaultdict(float)
    per_class_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    lengths: List[int] = []

    records_out = []
    for s in signals:
        per_class_records[s.label] += 1
        per_class_seconds[s.label] += s.duration_s
        stats = signal_statistics(s.signal)
        per_class_stats[s.label].append(stats)
        lengths.append(len(s.signal))
        records_out.append(
            {
                "record_id": s.record_id,
                "label": s.label,
                "fault_type": s.fault_type,
                "load_hp": s.load_hp,
                "rpm": s.rpm,
                "n_samples": int(len(s.signal)),
                "duration_s": round(s.duration_s, 3),
                "native_sampling_rate": s.meta["native_sampling_rate"],
                "sampling_rate": s.sampling_rate,
                "statistics": {k: round(v, 6) for k, v in stats.items()},
            }
        )

    # Average the per-record statistics within each class.
    class_stats_mean = {
        label: {
            k: round(float(np.mean([st[k] for st in sts])), 6)
            for k in sts[0]
        }
        for label, sts in per_class_stats.items()
    }

    return {
        "dataset": CFG.dataset.name,
        "label_mode": CFG.dataset.label_mode,
        "channel": CFG.dataset.channel,
        "target_sampling_rate_hz": CFG.dataset.sampling_rate,
        "n_records": len(signals),
        "n_classes": len(per_class_records),
        "classes": sorted(per_class_records),
        "class_distribution_records": dict(sorted(per_class_records.items())),
        "class_distribution_seconds": {
            k: round(v, 2) for k, v in sorted(per_class_seconds.items())
        },
        "signal_length_samples": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "total_duration_s": round(sum(per_class_seconds.values()), 2),
        "per_class_mean_statistics": class_stats_mean,
        "unreadable_records": errors,
        "records": records_out,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_class_distribution(summary: Dict[str, object], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist = summary["class_distribution_seconds"]  # type: ignore[index]
    labels = list(dist)
    values = [dist[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("Total signal duration (s)")
    ax.set_xlabel("Class")
    ax.set_title("CWRU class distribution (selected slice)")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_waveforms(signals: List[LoadedSignal], out_path: Path, seconds: float = 0.2) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One example record per class, preferring load 0 for comparability.
    by_label: Dict[str, LoadedSignal] = {}
    for s in sorted(signals, key=lambda s: s.load_hp):
        by_label.setdefault(s.label, s)

    labels = sorted(by_label)
    n = len(labels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 1.9 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, label in zip(axes, labels, strict=False):
        s = by_label[label]
        n_show = int(seconds * s.sampling_rate)
        t = np.arange(n_show) / s.sampling_rate
        ax.plot(t, s.signal[:n_show], linewidth=0.6, color="#C44E52")
        ax.set_title(f"{label} (record {s.record_id}, {s.load_hp} HP)", fontsize=9)
        ax.set_ylabel("a (g)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(labels):]:
        ax.axis("off")
    axes[min(len(labels), len(axes)) - 1].set_xlabel("Time (s)")
    fig.suptitle(f"Raw drive-end vibration, first {seconds:.2f} s per class", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_spectra(signals: List[LoadedSignal], out_path: Path) -> None:
    """FFT magnitude per class -- shows that fault classes differ spectrally."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_label: Dict[str, LoadedSignal] = {}
    for s in sorted(signals, key=lambda s: s.load_hp):
        by_label.setdefault(s.label, s)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in sorted(by_label):
        s = by_label[label]
        n = min(len(s.signal), 2**15)
        spec = np.abs(np.fft.rfft(s.signal[:n] * np.hanning(n))) / n
        freqs = np.fft.rfftfreq(n, d=1.0 / s.sampling_rate)
        ax.semilogy(freqs, spec + 1e-12, linewidth=0.7, label=label, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT| (log scale)")
    ax.set_title("Per-class spectra (one record per class)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore the vibration dataset")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    set_seed(CFG.seed)

    records = select_records(subsets=CFG.dataset.subsets, loads=CFG.dataset.loads)
    signals: List[LoadedSignal] = []
    errors: List[str] = []
    for rec in records:
        try:
            signals.append(load_signal(rec, raw_dir=args.raw_dir))
        except (FileNotFoundError, SignalLoadError) as exc:
            errors.append(f"{rec.filename}: {exc}")

    if not signals:
        print("No records could be loaded. Run `python -m preprocessing.download_cwru` first.")
        return 1

    summary = summarize(signals, errors)

    out_json = METRICS_DIR / "dataset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- human-readable report ------------------------------------------
    print("=" * 72)
    print("DATASET EXPLORATION -- STEP 1")
    print("=" * 72)
    print(f"Dataset               : {summary['dataset']} ({CFG.dataset.channel} channel)")
    print(f"Records loaded        : {summary['n_records']} "
          f"({len(errors)} unreadable)")
    print(f"Classes ({summary['n_classes']:2d})          : {', '.join(summary['classes'])}")
    print(f"Target sampling rate  : {summary['target_sampling_rate_hz']} Hz "
          f"(normal baseline natively 48 kHz, resampled)")
    sl = summary["signal_length_samples"]
    print(f"Signal length (samples): min {sl['min']}, max {sl['max']}, mean {sl['mean']:.0f}")
    print(f"Total signal duration : {summary['total_duration_s']:.1f} s")
    print()
    print(f"{'class':<12} {'records':>8} {'seconds':>9} {'rms':>9} {'kurtosis':>9} {'crest':>8}")
    print("-" * 62)
    for label in summary["classes"]:
        st = summary["per_class_mean_statistics"][label]
        print(f"{label:<12} "
              f"{summary['class_distribution_records'][label]:>8} "
              f"{summary['class_distribution_seconds'][label]:>9.1f} "
              f"{st['rms']:>9.4f} {st['kurtosis']:>9.2f} {st['crest_factor']:>8.2f}")
    if errors:
        print("\nUnreadable records:")
        for e in errors:
            print(f"  - {e}")
    print(f"\nSummary written to {out_json}")

    if not args.no_figures:
        plot_class_distribution(summary, FIGURES_DIR / "01_class_distribution.png")
        plot_example_waveforms(signals, FIGURES_DIR / "01_example_waveforms.png")
        plot_example_spectra(signals, FIGURES_DIR / "01_example_spectra.png")
        print(f"Figures written to {FIGURES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
