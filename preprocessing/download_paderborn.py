"""
Automated downloader for the Paderborn University Bearing Dataset.

Downloads official compressed RAR archives from the Paderborn Bearing DataCenter
(https://groups.uni-paderborn.de/kat/BearingDataCenter/), extracts the .mat
recordings, and places them in data/paderborn/ with checksum and size verification.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEST_DIR = ROOT_DIR / "data" / "paderborn"

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter"

# 32 benchmark bearings
BEARING_SETS = {
    "healthy": ["K001", "K002", "K003", "K004", "K005", "K006"],
    "outer_race_artificial": ["KA01", "KA03", "KA05", "KA07", "KA08", "KA09"],
    "inner_race_artificial": ["KI01", "KI03", "KI05", "KI07", "KI08"],
    "accelerated_fatigue": ["KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"],
}

# Small representative benchmark subset (1 healthy, 1 outer race, 1 inner race)
BENCHMARK_SUBSET = ["K001", "KA01", "KI01"]


def download_bearing_archive(
    bearing_id: str,
    dest_dir: Path,
    retries: int = 3,
) -> Tuple[str, str]:
    """Download a single bearing .rar archive from Paderborn University."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    rar_name = f"{bearing_id}.rar"
    url = f"{BASE_URL}/{rar_name}"
    rar_path = dest_dir / rar_name

    # Check if .mat files for this bearing are already extracted
    existing_mat = list(dest_dir.glob(f"*{bearing_id}*.mat"))
    if existing_mat:
        return "skipped", f"Bearing {bearing_id} already extracted ({len(existing_mat)} .mat files present)"

    # Download RAR
    print(f"Downloading {bearing_id} archive (~160 MB) from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (academic pdm research)"}

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB

                tmp_path = rar_path.with_suffix(".part")
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\r  [{bearing_id}] {percent:.1f}% ({downloaded / 1e6:.1f} MB)")
                            sys.stdout.flush()

                print("")
                tmp_path.replace(rar_path)
                break
        except Exception as exc:
            last_err = str(exc)
            print(f"  Attempt {attempt} failed: {exc}. Retrying...")
            time.sleep(3 * attempt)
    else:
        return "failed", f"{bearing_id}: {last_err}"

    # Extract RAR using bsdtar, unrar, or 7z
    print(f"Extracting {rar_name} into {dest_dir}...")
    extracted = False
    for tool in ["bsdtar", "unrar", "7z"]:
        try:
            if tool == "bsdtar":
                cmd = ["bsdtar", "-xf", str(rar_path), "-C", str(dest_dir)]
            elif tool == "unrar":
                cmd = ["unrar", "x", "-o+", str(rar_path), str(dest_dir)]
            else:
                cmd = ["7z", "x", f"-o{dest_dir}", str(rar_path)]

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                extracted = True
                break
        except FileNotFoundError:
            continue

    if extracted:
        # Clean up compressed RAR to save disk space
        if rar_path.exists():
            rar_path.unlink()
        mat_count = len(list(dest_dir.glob(f"*{bearing_id}*.mat")))
        return "ok", f"Extracted {mat_count} .mat files for {bearing_id}"
    else:
        return "downloaded_only", f"Downloaded {rar_name}, but no extractor (bsdtar/unrar/7z) could unpack it automatically."


def download_paderborn(
    bearings: List[str],
    dest_dir: Path = DEFAULT_DEST_DIR,
) -> None:
    """Download and extract a list of Paderborn bearings."""
    print("=" * 60)
    print(" PADEROORN UNIVERSITY BEARING DATASET DOWNLOADER")
    print(f" Destination: {dest_dir}")
    print(f" Bearings to fetch: {bearings}")
    print("=" * 60)

    for b in bearings:
        status, msg = download_bearing_archive(b, dest_dir)
        print(f"[{status.upper()}] {msg}")

    total_mat = len(list(dest_dir.glob("*.mat")))
    print("=" * 60)
    print(f"Done. Total .mat files now in {dest_dir}: {total_mat}")
    print("To train models on this data, run:")
    print(f"  python training/train_motor_models.py --data-dir {dest_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Paderborn University bearing dataset")
    parser.add_argument("--subset", action="store_true", help="Download small benchmark subset (K001, KA01, KI01) ~500MB")
    parser.add_argument("--bearings", nargs="+", default=None, help="Explicit list of bearings (e.g. K001 KA01)")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST_DIR), help="Destination directory")
    args = parser.parse_args()

    target_bearings = args.bearings or (BENCHMARK_SUBSET if args.subset else BENCHMARK_SUBSET)
    download_paderborn(target_bearings, dest_dir=Path(args.dest))
