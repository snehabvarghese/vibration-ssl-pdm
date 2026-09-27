"""
Download the CWRU bearing `.mat` files listed in `cwru_manifest.py`.

WHY
---
Phase 1 must be reproducible on any machine: `python -m preprocessing.download_cwru`
fetches exactly the files the pipeline expects into `data/raw/`.

INPUT : the manifest (+ optional filters from `config.CFG.dataset`)
OUTPUT: `.mat` files in `data/raw/`, plus a printed report of what was
        downloaded / skipped / failed.

Downloads are idempotent: an already-present, non-empty file is skipped.
Failures are collected and reported instead of crashing the run, so a single
dead link does not block the whole project.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

from config import CFG, DATA_RAW_DIR
from preprocessing.cwru_manifest import CWRURecord, select_records

_USER_AGENT = "vibration-ssl-pdm/0.1 (academic use)"
_MIN_VALID_BYTES = 10_000  # a real CWRU file is ~1-4 MB; anything tiny is an error page


def download_record(record: CWRURecord, dest_dir: Path, retries: int = 3) -> Tuple[str, str]:
    """Download one record.

    Returns (status, message) where status is 'skipped' | 'ok' | 'failed'.
    """
    dest = dest_dir / record.filename
    if dest.exists() and dest.stat().st_size > _MIN_VALID_BYTES:
        return "skipped", f"{record.filename} already present"

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(record.url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = resp.read()
            if len(payload) < _MIN_VALID_BYTES:
                last_err = f"response too small ({len(payload)} bytes)"
                continue
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return "ok", f"{record.filename} ({len(payload) / 1e6:.1f} MB)"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
            time.sleep(2 * attempt)
    return "failed", f"{record.filename}: {last_err}"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the CWRU bearing dataset")
    parser.add_argument("--out", type=Path, default=DATA_RAW_DIR, help="destination directory")
    parser.add_argument("--all", action="store_true", help="ignore config filters, fetch everything")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.all:
        records = select_records()
    else:
        records = select_records(subsets=CFG.dataset.subsets, loads=CFG.dataset.loads)

    print(f"Downloading {len(records)} CWRU records into {args.out}")
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []
    for i, rec in enumerate(records, 1):
        status, msg = download_record(rec, args.out)
        counts[status] += 1
        if status == "failed":
            failures.append(msg)
        print(f"  [{i:3d}/{len(records)}] {status:7s} {msg}")

    print(f"\nDone: {counts['ok']} downloaded, {counts['skipped']} already present, "
          f"{counts['failed']} failed")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
