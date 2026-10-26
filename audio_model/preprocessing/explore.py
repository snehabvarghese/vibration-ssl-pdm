"""
Explore MIMII DG / DCASE 2022 Task 2 dataset.

Inspects the real directory layout and filenames in data/audio/raw/,
summarizes class distributions across machines, sections, domains, and conditions,
writes results/audio/metrics/dataset_summary.json,
prints a structured summary table,
and saves waveform & log-Mel spectrogram comparison figures to results/audio/figures/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import scipy.signal

from audio_model.config_audio import (
    AUDIO_CFG,
    FIGURES_AUDIO_DIR,
    METRICS_AUDIO_DIR,
    RAW_AUDIO_DIR,
)

# Regex pattern for DCASE 2022 Task 2 / MIMII DG filenames:
# e.g., section_00_source_train_normal_0000_m-n_W.wav
# e.g., section_01_target_test_anomaly_0003_vel_6.wav
FILENAME_PATTERN = re.compile(
    r"^(?P<section>section_\d+)_(?P<domain>source|target)_(?P<split>train|test)_(?P<condition>normal|anomaly)_(?P<index>\d+)(?:_(?P<attributes>.+))?\.wav$",
    re.IGNORECASE,
)


def parse_wav_filename(filename: str) -> Dict[str, str]:
    """Parse DCASE 2022 Task 2 filename into structured metadata fields."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        data = match.groupdict()
        return {
            "section": data["section"],
            "domain": data["domain"].lower(),
            "split": data["split"].lower(),
            "condition": data["condition"].lower(),
            "index": data["index"],
            "attributes": data.get("attributes") or "",
        }

    # Fallback heuristic parser if custom naming occurs
    parts = filename.replace(".wav", "").split("_")
    sec = next((p for p in parts if "section" in p.lower() or p.isdigit()), "section_00")
    dom = "target" if "target" in filename.lower() else "source"
    splt = "train" if "train" in filename.lower() else "test"
    cond = "anomaly" if "anomaly" in filename.lower() else "normal"
    return {
        "section": sec if sec.startswith("section_") else f"section_{sec}",
        "domain": dom,
        "split": splt,
        "condition": cond,
        "index": parts[-1] if parts else "0000",
        "attributes": "",
    }


def read_audio_specs(wav_path: Path) -> Tuple[int, float, int]:
    """Fast inspection of audio metadata (sampling_rate, duration_s, channels).

    Uses standard library wave module to read header without loading entire audio stream.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            dur = frames / float(sr) if sr > 0 else 0.0
            return sr, round(dur, 2), ch
    except Exception:
        # Fallback to scipy if wave module encounters non-standard header
        sr, data = scipy.io.wavfile.read(str(wav_path))
        ch = 1 if data.ndim == 1 else data.shape[1]
        dur = len(data) / float(sr) if sr > 0 else 0.0
        return sr, round(dur, 2), ch


def compute_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 128,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Construct Slaney/HTK-style triangular Mel filterbank matrix (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:
            filterbank[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filterbank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)

        # Normalize area
        enorm = 2.0 / (hz_points[m + 1] - hz_points[m - 1])
        filterbank[m - 1] *= enorm

    return filterbank


def compute_log_mel_spectrogram(
    signal: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    win_length: int = 1024,
    f_min: float = 50.0,
    f_max: float = 8000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Log-Mel Spectrogram of 1-D audio signal.

    Returns (times, mel_frequencies, log_mel_spec).
    """
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
        # Normalize int16 scale if needed
        max_val = np.max(np.abs(signal))
        if max_val > 1.0:
            signal /= 32768.0

    # STFT
    window = scipy.signal.windows.hann(win_length)
    f, t, zxx = scipy.signal.stft(
        signal,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
    )
    power_spec = np.abs(zxx) ** 2  # (n_fft//2 + 1, n_frames)

    # Mel projection
    fb = compute_mel_filterbank(sr, n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max)
    mel_spec = np.dot(fb, power_spec)  # (n_mels, n_frames)

    # Log power compression
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_freqs = np.linspace(f_min, f_max, n_mels)
    return t, mel_freqs, log_mel


def scan_dataset(raw_dir: Path) -> Dict[str, Any]:
    """Scan all downloaded machine folders and compute detailed statistics."""
    if not raw_dir.exists():
        return {}

    machine_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    dataset_summary: Dict[str, Any] = {}

    for m_dir in sorted(machine_dirs):
        machine_name = m_dir.name
        wav_files = list(m_dir.rglob("*.wav"))
        if not wav_files:
            continue

        sample_sr, sample_dur, sample_ch = read_audio_specs(wav_files[0])

        machine_stats: Dict[str, Any] = {
            "machine": machine_name,
            "root_path": str(m_dir.resolve()),
            "total_clips": len(wav_files),
            "sampling_rate": sample_sr,
            "duration_seconds": sample_dur,
            "channel_count": sample_ch,
            "sections": {},
        }

        for wf in wav_files:
            meta = parse_wav_filename(wf.name)
            sec = meta["section"]
            dom = meta["domain"]
            splt = meta["split"]
            cond = meta["condition"]

            if sec not in machine_stats["sections"]:
                machine_stats["sections"][sec] = {
                    "train": {"source_normal": 0, "target_normal": 0, "total": 0},
                    "test": {
                        "source_normal": 0,
                        "source_anomaly": 0,
                        "target_normal": 0,
                        "target_anomaly": 0,
                        "total": 0,
                    },
                    "sample_clips": {"normal": None, "anomaly": None},
                }

            sec_dict = machine_stats["sections"][sec]
            if splt == "train":
                key = f"{dom}_{cond}"
                if key in sec_dict["train"]:
                    sec_dict["train"][key] += 1
                sec_dict["train"]["total"] += 1
                if sec_dict["sample_clips"]["normal"] is None and cond == "normal":
                    sec_dict["sample_clips"]["normal"] = str(wf)
            else:
                key = f"{dom}_{cond}"
                if key in sec_dict["test"]:
                    sec_dict["test"][key] += 1
                sec_dict["test"]["total"] += 1
                if sec_dict["sample_clips"]["anomaly"] is None and cond == "anomaly":
                    sec_dict["sample_clips"]["anomaly"] = str(wf)
                elif sec_dict["sample_clips"]["normal"] is None and cond == "normal":
                    sec_dict["sample_clips"]["normal"] = str(wf)

        dataset_summary[machine_name] = machine_stats

    return dataset_summary


def print_summary_table(dataset_summary: Dict[str, Any]) -> None:
    """Print clean ASCII breakdown table per machine and section."""
    print("\n" + "=" * 115)
    print(f"{'Machine':10s} | {'Section':10s} | {'Train Norm (Src/Tgt)':22s} | {'Test Norm (Src/Tgt)':20s} | {'Test Anom (Src/Tgt)':20s} | {'Total':6s} | {'Fs':6s} | {'Dur':5s} | {'Ch':2s}")
    print("-" * 115)

    grand_total = 0
    for m_name, m_stats in dataset_summary.items():
        sr = m_stats["sampling_rate"]
        dur = m_stats["duration_seconds"]
        ch = m_stats["channel_count"]

        for sec_name, sec_stats in sorted(m_stats["sections"].items()):
            tr = sec_stats["train"]
            te = sec_stats["test"]
            train_str = f"{tr['source_normal']:4d} / {tr['target_normal']:3d} ({tr['total']:4d})"
            test_norm_str = f"{te['source_normal']:4d} / {te['target_normal']:3d}"
            test_anom_str = f"{te['source_anomaly']:4d} / {te['target_anomaly']:3d}"
            sec_total = tr["total"] + te["total"]
            grand_total += sec_total

            print(
                f"{m_name:10s} | {sec_name:10s} | {train_str:22s} | {test_norm_str:20s} | {test_anom_str:20s} | {sec_total:6d} | {sr:5d}Hz| {dur:4.1f}s | {ch:2d}"
            )
        print("-" * 115)

    print(f"Total Clips Indexed Across All Machines: {grand_total}")
    print("=" * 115 + "\n")


def generate_example_figures(
    dataset_summary: Dict[str, Any],
    figures_dir: Path,
) -> List[Path]:
    """Generate waveform and log-mel spectrogram comparison for normal vs anomaly per machine."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: List[Path] = []

    for m_name, m_stats in dataset_summary.items():
        normal_path: Optional[str] = None
        anomaly_path: Optional[str] = None

        # Pick representative samples
        for sec_name, sec_dict in m_stats["sections"].items():
            if not normal_path and sec_dict["sample_clips"]["normal"]:
                normal_path = sec_dict["sample_clips"]["normal"]
            if not anomaly_path and sec_dict["sample_clips"]["anomaly"]:
                anomaly_path = sec_dict["sample_clips"]["anomaly"]
            if normal_path and anomaly_path:
                break

        if not normal_path or not anomaly_path:
            continue

        sr_n, sig_n = scipy.io.wavfile.read(normal_path)
        sr_a, sig_a = scipy.io.wavfile.read(anomaly_path)

        tn, fn, log_mel_n = compute_log_mel_spectrogram(sig_n, sr=sr_n)
        ta, fa, log_mel_a = compute_log_mel_spectrogram(sig_a, sr=sr_a)

        time_axis_n = np.linspace(0, len(sig_n) / sr_n, len(sig_n))
        time_axis_a = np.linspace(0, len(sig_a) / sr_a, len(sig_a))

        fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex="col", constrained_layout=True)
        fig.suptitle(f"MIMII DG Acoustic Comparison: {m_name.upper()} (Normal vs Anomaly)", fontsize=14, fontweight="bold")

        # Top row: Waveforms
        axs[0, 0].plot(time_axis_n, sig_n, color="#2b5c8f", alpha=0.8, lw=0.6)
        axs[0, 0].set_title(f"Normal Waveform ({Path(normal_path).name[:35]}...)", fontsize=11)
        axs[0, 0].set_ylabel("Amplitude")
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(time_axis_a, sig_a, color="#c0392b", alpha=0.8, lw=0.6)
        axs[0, 1].set_title(f"Anomaly Waveform ({Path(anomaly_path).name[:35]}...)", fontsize=11)
        axs[0, 1].set_ylabel("Amplitude")
        axs[0, 1].grid(True, alpha=0.3)

        # Bottom row: Log-Mel Spectrograms
        v_min = min(log_mel_n.min(), log_mel_a.min())
        v_max = max(log_mel_n.max(), log_mel_a.max())

        im0 = axs[1, 0].imshow(
            log_mel_n,
            origin="lower",
            aspect="auto",
            extent=[tn[0], tn[-1], fn[0], fn[-1]],
            cmap="viridis",
            vmin=v_min,
            vmax=v_max,
        )
        axs[1, 0].set_title("Normal Log-Mel Spectrogram (dB)", fontsize=11)
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Frequency (Hz)")

        im1 = axs[1, 1].imshow(
            log_mel_a,
            origin="lower",
            aspect="auto",
            extent=[ta[0], ta[-1], fa[0], fa[-1]],
            cmap="viridis",
            vmin=v_min,
            vmax=v_max,
        )
        axs[1, 1].set_title("Anomaly Log-Mel Spectrogram (dB)", fontsize=11)
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Frequency (Hz)")

        cbar = fig.colorbar(im1, ax=axs[1, :], shrink=0.6, pad=0.02)
        cbar.set_label("Log-Mel Energy (dB)")

        save_path = figures_dir / f"{m_name}_example_comparison.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        generated_plots.append(save_path)

    # Class balance bar chart
    if dataset_summary:
        balance_path = generate_class_balance_chart(dataset_summary, figures_dir)
        if balance_path:
            generated_plots.append(balance_path)

    return generated_plots


def generate_class_balance_chart(
    dataset_summary: Dict[str, Any],
    figures_dir: Path,
) -> Optional[Path]:
    """Generate clean stacked/grouped bar chart of class balances."""
    labels: List[str] = []
    train_counts: List[int] = []
    test_norm_counts: List[int] = []
    test_anom_counts: List[int] = []

    for m_name, m_stats in dataset_summary.items():
        for sec_name, sec_dict in sorted(m_stats["sections"].items()):
            labels.append(f"{m_name}\n{sec_name}")
            train_counts.append(sec_dict["train"]["total"])
            test_norm_counts.append(sec_dict["test"]["source_normal"] + sec_dict["test"]["target_normal"])
            test_anom_counts.append(sec_dict["test"]["source_anomaly"] + sec_dict["test"]["target_anomaly"])

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6), constrained_layout=True)
    b1 = ax.bar(x - width, train_counts, width, label="Train Normal (SSL/Baseline Fit)", color="#2ecc71")
    b2 = ax.bar(x, test_norm_counts, width, label="Test Normal (Evaluation)", color="#3498db")
    b3 = ax.bar(x + width, test_anom_counts, width, label="Test Anomaly (Evaluation)", color="#e74c3c")

    ax.set_ylabel("Clip Count", fontsize=11)
    ax.set_title("MIMII DG Dataset Split & Class Balance per Machine / Section", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add count labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    save_path = figures_dir / "dataset_class_balance.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Explore MIMII DG dataset and export summary")
    parser.add_argument("--raw-dir", type=Path, default=RAW_AUDIO_DIR, help="Path to raw audio folder")
    parser.add_argument("--out-json", type=Path, default=METRICS_AUDIO_DIR / "dataset_summary.json", help="Path to output JSON")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_AUDIO_DIR, help="Path to save figures")
    args = parser.parse_args(argv)

    print(f"Scanning MIMII DG audio dataset in: {args.raw_dir.resolve()}")
    summary = scan_dataset(args.raw_dir)

    if not summary:
        print(f"No audio files found in {args.raw_dir}.")
        print("Please run `python -m audio_model.preprocessing.download_mimii_dg --machines fan bearing` first.")
        return 1

    # Save JSON summary
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to: {args.out_json}")

    # Print summary table
    print_summary_table(summary)

    # Generate figures
    print(f"Generating figures in: {args.figures_dir}...")
    fig_paths = generate_example_figures(summary, args.figures_dir)
    for p in fig_paths:
        print(f"  [SAVED] {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
