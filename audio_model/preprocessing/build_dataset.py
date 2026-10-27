"""
Build and cache processed log-Mel and context-stacked datasets for audio anomaly detection.

Processes MIMII DG raw audio into numpy arrays (.npy) stored in:
    data/audio/processed/<machine>/

RULES ENFORCED
--------------
* Normal clips ONLY in training and validation sets.
* Validation split is drawn strictly from TRAIN normal clips (never from test).
* Per-machine AudioNormalizer is fitted strictly on the TRAINING normal split.
* Full test set is cached with ground-truth binary labels (0=normal, 1=anomaly),
  domain tags (source vs target), and section IDs for leak-free, reproducible benchmarking.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from audio_model.config_audio import (
    AUDIO_CFG,
    PROCESSED_AUDIO_DIR,
    RAW_AUDIO_DIR,
    set_seed,
)
from audio_model.preprocessing.explore import parse_wav_filename
from audio_model.preprocessing.features import (
    AudioNormalizer,
    load_audio,
    log_mel,
    stack_frames,
)


def stratified_train_val_split(
    clip_paths: List[Path],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Split normal training clips into train and val sets stratified by (section, domain).

    Ensures both source and target normal clips across all sections are represented
    in validation without leakage.
    """
    random_state = random.Random(seed)
    strata: Dict[Tuple[str, str], List[Path]] = {}

    for p in clip_paths:
        meta = parse_wav_filename(p.name)
        key = (meta["section"], meta["domain"])
        if key not in strata:
            strata[key] = []
        strata[key].append(p)

    train_clips: List[Path] = []
    val_clips: List[Path] = []

    for key, paths in strata.items():
        shuffled = list(paths)
        random_state.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        val_clips.extend(shuffled[:n_val])
        train_clips.extend(shuffled[n_val:])

    # Sort for deterministic processing order
    train_clips.sort()
    val_clips.sort()
    return train_clips, val_clips


def extract_features_for_clips(
    clip_paths: List[Path],
    desc: str = "Extracting",
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Extract log-Mel spectrograms and metadata for a list of audio files."""
    cfg = AUDIO_CFG.feature
    specs: List[np.ndarray] = []
    sections: List[str] = []
    domains: List[str] = []
    filenames: List[str] = []

    for p in tqdm(clip_paths, desc=desc, unit="clip"):
        y = load_audio(
            p,
            sr=cfg.sample_rate,
            duration_s=cfg.clip_duration_s,
        )
        mel = log_mel(
            y,
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            power=cfg.power,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        meta = parse_wav_filename(p.name)
        specs.append(mel)
        sections.append(meta["section"])
        domains.append(meta["domain"])
        filenames.append(p.name)

    spec_array = np.stack(specs, axis=0).astype(np.float32)
    return spec_array, sections, domains, filenames


def process_machine(
    machine: str,
    raw_dir: Path,
    out_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Process all audio for a single machine and save cached .npy datasets."""
    set_seed(seed)
    machine_raw = raw_dir / machine
    train_dir = machine_raw / "train"
    test_dir = machine_raw / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Missing train or test folder in {machine_raw}")

    dest_dir = out_dir / machine
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nProcessing Machine: {machine.upper()}\n{'='*70}")

    # 1. Gather all training clips (strictly normal)
    all_train_wavs = sorted(list(train_dir.glob("*.wav")))
    if not all_train_wavs:
        raise RuntimeError(f"No .wav files found in {train_dir}")

    # Verify all training clips are normal
    for p in all_train_wavs:
        cond = parse_wav_filename(p.name)["condition"]
        if cond != "normal":
            raise ValueError(f"Found non-normal clip in training folder: {p.name}")

    # 2. Split train into train / val
    train_paths, val_paths = stratified_train_val_split(
        all_train_wavs,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(f"Normal clips split: {len(train_paths)} train, {len(val_paths)} val")

    # 3. Gather test clips
    test_paths = sorted(list(test_dir.glob("*.wav")))
    print(f"Test clips found: {len(test_paths)}")

    # 4. Extract raw log-Mel spectrograms
    train_logmel, train_sec, train_dom, train_fn = extract_features_for_clips(
        train_paths, desc=f"{machine} [Train]"
    )
    val_logmel, val_sec, val_dom, val_fn = extract_features_for_clips(
        val_paths, desc=f"{machine} [Val]"
    )
    test_logmel, test_sec, test_dom, test_fn = extract_features_for_clips(
        test_paths, desc=f"{machine} [Test]"
    )

    # 5. Extract ground truth labels for test clips (0=normal, 1=anomaly)
    test_labels = np.array(
        [1 if parse_wav_filename(p.name)["condition"] == "anomaly" else 0 for p in test_paths],
        dtype=np.int64,
    )

    # 6. Per-machine normalization: FIT ON TRAINING NORMAL DATA ONLY
    scaler_path = dest_dir / "scaler.npz"
    normalizer = AudioNormalizer()
    if normalize:
        print("Fitting AudioNormalizer strictly on TRAINING normal clips...")
        normalizer.fit(train_logmel)
        normalizer.save(scaler_path)

        train_logmel = normalizer.transform(train_logmel)
        val_logmel = normalizer.transform(val_logmel)
        test_logmel = normalizer.transform(test_logmel)

    # 7. Compute context-stacked frames (DCASE baseline style)
    cfg = AUDIO_CFG.feature
    print(f"Stacking context frames (context_frames={cfg.context_frames}, hop={cfg.frame_hop})...")
    train_stacked = stack_frames(train_logmel, context_frames=cfg.context_frames, frame_hop=cfg.frame_hop)
    val_stacked = stack_frames(val_logmel, context_frames=cfg.context_frames, frame_hop=cfg.frame_hop)
    test_stacked = stack_frames(test_logmel, context_frames=cfg.context_frames, frame_hop=cfg.frame_hop)

    # 8. Save all arrays to disk
    print(f"Saving .npy arrays to {dest_dir}...")
    np.save(dest_dir / "train_logmel.npy", train_logmel)
    np.save(dest_dir / "train_stacked.npy", train_stacked)
    np.save(dest_dir / "train_sections.npy", np.array(train_sec))
    np.save(dest_dir / "train_domains.npy", np.array(train_dom))

    np.save(dest_dir / "val_logmel.npy", val_logmel)
    np.save(dest_dir / "val_stacked.npy", val_stacked)
    np.save(dest_dir / "val_sections.npy", np.array(val_sec))
    np.save(dest_dir / "val_domains.npy", np.array(val_dom))

    np.save(dest_dir / "test_logmel.npy", test_logmel)
    np.save(dest_dir / "test_stacked.npy", test_stacked)
    np.save(dest_dir / "test_labels.npy", test_labels)
    np.save(dest_dir / "test_sections.npy", np.array(test_sec))
    np.save(dest_dir / "test_domains.npy", np.array(test_dom))
    np.save(dest_dir / "test_filenames.npy", np.array(test_fn))

    summary: Dict[str, Any] = {
        "machine": machine,
        "sample_rate": cfg.sample_rate,
        "n_mels": cfg.n_mels,
        "n_frames": train_logmel.shape[-1],
        "context_frames": cfg.context_frames,
        "vector_dim": train_stacked.shape[-1],
        "train": {
            "n_clips": len(train_paths),
            "logmel_shape": list(train_logmel.shape),
            "stacked_shape": list(train_stacked.shape),
        },
        "val": {
            "n_clips": len(val_paths),
            "logmel_shape": list(val_logmel.shape),
            "stacked_shape": list(val_stacked.shape),
        },
        "test": {
            "n_clips": len(test_paths),
            "n_normal": int((test_labels == 0).sum()),
            "n_anomaly": int((test_labels == 1).sum()),
            "logmel_shape": list(test_logmel.shape),
            "stacked_shape": list(test_stacked.shape),
        },
        "scaler_path": str(scaler_path.resolve()),
    }

    with open(dest_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess and cache MIMII DG log-Mel datasets")
    parser.add_argument(
        "--machines",
        nargs="+",
        default=["fan", "bearing"],
        help="Machines to process (default: fan bearing)",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_AUDIO_DIR, help="Path to raw audio")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_AUDIO_DIR, help="Path to processed audio")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction from normal train clips")
    parser.add_argument("--seed", type=int, default=AUDIO_CFG.seed, help="Random seed for repeatable split")
    args = parser.parse_args(argv)

    print(f"Building processed datasets for machines: {', '.join(args.machines)}")
    overall_summary: Dict[str, Any] = {}

    for machine in args.machines:
        try:
            summ = process_machine(
                machine=machine,
                raw_dir=args.raw_dir,
                out_dir=args.out_dir,
                val_fraction=args.val_fraction,
                seed=args.seed,
            )
            overall_summary[machine] = summ
        except Exception as exc:
            print(f"Error processing {machine}: {exc}")
            raise exc

    print("\n" + "=" * 95)
    print("PROCESSED DATASET REPORT")
    print("=" * 95)
    print(f"{'Machine':10s} | {'Train (Clips/Shape)':24s} | {'Val (Clips/Shape)':22s} | {'Test (Norm/Anom/Shape)':28s}")
    print("-" * 95)
    for m, s in overall_summary.items():
        tr_s = f"{s['train']['n_clips']} clips {tuple(s['train']['logmel_shape'])}"
        val_s = f"{s['val']['n_clips']} clips {tuple(s['val']['logmel_shape'])}"
        te_s = f"{s['test']['n_normal']}N / {s['test']['n_anomaly']}A {tuple(s['test']['logmel_shape'])}"
        print(f"{m:10s} | {tr_s:24s} | {val_s:22s} | {te_s:28s}")
        print(f"  --> Stacked Vector Dimension: {s['vector_dim']} (shape: {tuple(s['train']['stacked_shape'])})")
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
