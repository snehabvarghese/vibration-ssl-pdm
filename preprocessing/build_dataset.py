"""
STEP 2 deliverable: turn raw recordings into a single processed segment array.

WHAT IT DOES
------------
For every record: filter -> downsample -> segment -> z-score, then stack
everything into one `.npz` with parallel metadata arrays:

    segments   (N, window_size) float32
    labels     (N,)  int64   class index
    record_ids (N,)  str     source recording  <- used for leak-free splits
    loads      (N,)  int64   motor load (HP)
    fault_types(N,)  str

WHY ONE FILE
------------
Training, embedding extraction, anomaly detection and the dashboard then all
read exactly the same tensor, so results are directly comparable and the
dashboard needs no preprocessing code of its own.

RUN
---
  python -m preprocessing.build_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

from config import CFG, DATA_PROCESSED_DIR, DATA_RAW_DIR, METRICS_DIR, set_seed
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import SignalLoadError, load_signal
from preprocessing.segmentation import preprocess_record

DEFAULT_OUTPUT = DATA_PROCESSED_DIR / "segments.npz"


def build(
    raw_dir: Path = DATA_RAW_DIR,
    output: Path = DEFAULT_OUTPUT,
    verbose: bool = True,
) -> dict:
    set_seed(CFG.seed)
    records = select_records(subsets=CFG.dataset.subsets, loads=CFG.dataset.loads)

    all_segments: List[np.ndarray] = []
    labels: List[str] = []
    record_ids: List[str] = []
    loads: List[int] = []
    fault_types: List[str] = []
    errors: List[str] = []
    effective_fs = None

    for rec in records:
        try:
            sig = load_signal(rec, raw_dir=raw_dir)
        except (FileNotFoundError, SignalLoadError) as exc:
            errors.append(f"{rec.filename}: {exc}")
            continue

        segs, fs = preprocess_record(sig.signal, sig.sampling_rate, CFG.preprocess)
        if segs.shape[0] == 0:
            errors.append(f"{rec.filename}: produced no usable segments")
            continue
        effective_fs = fs

        all_segments.append(segs)
        labels += [sig.label] * segs.shape[0]
        record_ids += [sig.record_id] * segs.shape[0]
        loads += [sig.load_hp] * segs.shape[0]
        fault_types += [sig.fault_type] * segs.shape[0]

        if verbose:
            print(f"  {rec.filename:>9} {sig.label:<10} -> {segs.shape[0]:4d} segments "
                  f"of {segs.shape[1]} samples")

    if not all_segments:
        raise RuntimeError("no segments produced -- did you download the dataset?")

    segments = np.concatenate(all_segments, axis=0)
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        segments=segments,
        labels=y,
        classes=np.array(classes),
        record_ids=np.array(record_ids),
        loads=np.array(loads, dtype=np.int64),
        fault_types=np.array(fault_types),
        sampling_rate=np.array(effective_fs),
    )

    counts = {c: int((y == i).sum()) for c, i in class_to_idx.items()}
    summary = {
        "output": str(output),
        "n_segments": int(segments.shape[0]),
        "window_size": int(segments.shape[1]),
        "effective_sampling_rate_hz": int(effective_fs),
        "window_seconds": round(segments.shape[1] / float(effective_fs), 4),
        "overlap": CFG.preprocess.overlap,
        "n_classes": len(classes),
        "classes": classes,
        "segments_per_class": counts,
        "n_records": len(set(record_ids)),
        "errors": errors,
        "preprocess_config": {
            "filter_enabled": CFG.preprocess.filter_enabled,
            "filter_type": CFG.preprocess.filter_type,
            "filter_low_hz": CFG.preprocess.filter_low_hz,
            "filter_high_hz": CFG.preprocess.filter_high_hz,
            "downsample_factor": CFG.preprocess.downsample_factor,
            "normalization": CFG.preprocess.normalization,
        },
    }
    (METRICS_DIR / "processed_dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the processed segment dataset")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    summary = build(args.raw_dir, args.output, verbose=not args.quiet)

    print("\n" + "=" * 72)
    print("PROCESSED DATASET -- STEP 2")
    print("=" * 72)
    print(f"Segments      : {summary['n_segments']}")
    print(f"Window        : {summary['window_size']} samples "
          f"= {summary['window_seconds']} s @ {summary['effective_sampling_rate_hz']} Hz "
          f"(overlap {summary['overlap']:.0%})")
    print(f"Classes ({summary['n_classes']:2d})  : {', '.join(summary['classes'])}")
    print(f"Records       : {summary['n_records']}")
    print(f"Saved to      : {summary['output']}")
    if summary["errors"]:
        print("Errors:")
        for e in summary["errors"]:
            print(f"  - {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
