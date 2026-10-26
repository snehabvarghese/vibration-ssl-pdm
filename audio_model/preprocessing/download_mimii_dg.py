"""
Download and extract the MIMII DG dataset (DCASE 2022 Task 2 Development Dataset).

Zenodo Record: 10.5281/zenodo.6355122 (https://zenodo.org/record/6355122)

WHY THIS DESIGN
---------------
* Large archives (~700MB - 1GB each) can experience connection drops or IncompleteRead errors.
* This downloader supports HTTP Range resumption, streaming chunked I/O, and automated retries.
* Supports machine-level filtering (--machines fan bearing) to save bandwidth and storage.
"""

from __future__ import annotations

import argparse
import http.client
import os
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from audio_model.config_audio import AUDIO_CFG, RAW_AUDIO_DIR

USER_AGENT = "audio-anomaly-mimii-dg/1.0 (academic research)"
CHUNK_SIZE = 1024 * 1024  # 1 MB chunk

ZENODO_RECORD_ID = 6355122
ZENODO_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}"

# Exact filenames, sizes in bytes, and content endpoints on Zenodo record 6355122
MACHINE_CATALOG: Dict[str, Dict[str, any]] = {
    "bearing": {
        "zip_name": "dev_bearing.zip",
        "size_bytes": 772_253_369,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_bearing.zip/content",
        "target_dir": "bearing",
    },
    "fan": {
        "zip_name": "dev_fan.zip",
        "size_bytes": 928_511_244,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_fan.zip/content",
        "target_dir": "fan",
    },
    "gearbox": {
        "zip_name": "dev_gearbox.zip",
        "size_bytes": 946_394_433,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_gearbox.zip/content",
        "target_dir": "gearbox",
    },
    "slider": {
        "zip_name": "dev_slider.zip",
        "size_bytes": 913_701_334,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_slider.zip/content",
        "target_dir": "slider",
    },
    "valve": {
        "zip_name": "dev_valve.zip",
        "size_bytes": 825_381_222,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_valve.zip/content",
        "target_dir": "valve",
    },
    "ToyTrain": {
        "zip_name": "dev_ToyTrain.zip",
        "size_bytes": 826_939_129,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_ToyTrain.zip/content",
        "target_dir": "ToyTrain",
    },
    "ToyCar": {
        "zip_name": "dev_ToyCar.zip",
        "size_bytes": 981_981_341,
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/dev_ToyCar.zip/content",
        "target_dir": "ToyCar",
    },
}


def is_already_extracted(raw_dir: Path, machine: str) -> bool:
    """Check if the extracted folder exists and has wav files."""
    folder = raw_dir / machine
    if not folder.exists() or not folder.is_dir():
        return False
    train_wavs = list((folder / "train").glob("*.wav")) if (folder / "train").exists() else []
    test_wavs = list((folder / "test").glob("*.wav")) if (folder / "test").exists() else []
    return len(train_wavs) > 10 and len(test_wavs) > 10


def download_with_resume(
    url: str,
    dest_path: Path,
    expected_size: Optional[int] = None,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> bool:
    """Download a file with HTTP Range header resume support and IncompleteRead protection."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_suffix(dest_path.suffix + ".part")

    retries = 0
    while retries < max_retries:
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0
        if expected_size and existing_bytes == expected_size:
            part_path.replace(dest_path)
            return True

        headers = {"User-Agent": USER_AGENT}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing_bytes > 0 else "wb"

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = getattr(resp, "status", 200)
                # If server doesn't support 206 Partial Content, restart from scratch
                if existing_bytes > 0 and status != 206:
                    mode = "wb"
                    existing_bytes = 0

                content_len = resp.headers.get("Content-Length")
                total_bytes = existing_bytes + int(content_len) if content_len else expected_size

                with open(part_path, mode) as f, tqdm(
                    total=total_bytes,
                    initial=existing_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=dest_path.name,
                    leave=True,
                ) as pbar:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Completed stream without error
            final_size = part_path.stat().st_size
            if expected_size and final_size != expected_size:
                print(f"\nWarning: Downloaded {final_size} bytes, expected {expected_size}. Retrying...")
                retries += 1
                time.sleep(backoff_factor * retries)
                continue

            part_path.replace(dest_path)
            return True

        except (
            http.client.IncompleteRead,
            urllib.error.URLError,
            socket.timeout,
            TimeoutError,
            ConnectionResetError,
            OSError,
        ) as exc:
            retries += 1
            wait_s = backoff_factor * (2 ** (retries - 1))
            current_bytes = part_path.stat().st_size if part_path.exists() else 0
            print(
                f"\nNetwork interrupted ({exc.__class__.__name__}: {exc}). "
                f"Resuming from {current_bytes / 1e6:.1f} MB in {wait_s:.1f}s (attempt {retries}/{max_retries})..."
            )
            time.sleep(wait_s)

    return False


def extract_archive(zip_path: Path, extract_dir: Path, remove_zip: bool = False) -> bool:
    """Extract archive into extract_dir."""
    print(f"Extracting {zip_path.name} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            with tqdm(total=len(members), desc=f"Unpacking {zip_path.stem}", unit="files") as pbar:
                for member in members:
                    zf.extract(member, extract_dir)
                    pbar.update(1)

        if remove_zip:
            zip_path.unlink()
        return True
    except Exception as exc:
        print(f"Error extracting {zip_path.name}: {exc}")
        return False


def download_mimii_dg_machine(
    machine: str,
    raw_dir: Path,
    keep_zip: bool = True,
    retries: int = 5,
) -> Tuple[str, str]:
    """Download and extract one machine archive.

    Returns (status, message).
    """
    if machine not in MACHINE_CATALOG:
        return "failed", f"Unknown machine '{machine}'. Available: {list(MACHINE_CATALOG.keys())}"

    if is_already_extracted(raw_dir, machine):
        return "skipped", f"Machine '{machine}' already downloaded and extracted in {raw_dir / machine}"

    entry = MACHINE_CATALOG[machine]
    zip_path = raw_dir / entry["zip_name"]

    # Check if a valid downloaded zip already exists
    if zip_path.exists() and zip_path.stat().st_size == entry["size_bytes"]:
        print(f"Archive {zip_path.name} already fully downloaded. Extracting...")
        ok = extract_archive(zip_path, raw_dir, remove_zip=not keep_zip)
        if ok:
            return "ok", f"Extracted {zip_path.name} successfully"
        return "failed", f"Extraction failed for {zip_path.name}"

    # Download with resume
    print(f"\n--- Downloading MIMII DG [{machine}] ({entry['size_bytes'] / 1e6:.1f} MB) ---")
    success = download_with_resume(
        url=entry["url"],
        dest_path=zip_path,
        expected_size=entry["size_bytes"],
        max_retries=retries,
    )

    if not success:
        return (
            "failed",
            f"Failed to download {entry['zip_name']}. You can download manually from:\n"
            f"  {entry['url']}\n"
            f"  and place/extract into: {raw_dir}",
        )

    # Extract
    ok = extract_archive(zip_path, raw_dir, remove_zip=not keep_zip)
    if ok:
        return "ok", f"Downloaded and extracted {machine} ({entry['size_bytes'] / 1e6:.1f} MB)"
    return "failed", f"Downloaded {zip_path.name} but extraction failed"


def print_manual_instructions(machines: List[str], dest_dir: Path) -> None:
    """Print clear instructions for manual download if needed."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print(f"Zenodo Record URL: {ZENODO_URL}")
    print(f"Target extraction directory: {dest_dir.resolve()}\n")
    for m in machines:
        if m in MACHINE_CATALOG:
            entry = MACHINE_CATALOG[m]
            print(f"Machine: {m}")
            print(f"  File: {entry['zip_name']} ({entry['size_bytes'] / 1e6:.1f} MB)")
            print(f"  URL : {entry['url']}")
            print(f"  Extract folder into: {dest_dir / m}")
            print()
    print("=" * 70 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download & extract MIMII DG (DCASE 2022 Task 2 Dev)")
    parser.add_argument(
        "--machines",
        nargs="+",
        default=["fan", "bearing"],
        help="Machines to download (default: fan bearing). Available: fan bearing gearbox slider valve ToyCar ToyTrain",
    )
    parser.add_argument("--all", action="store_true", help="Download all 7 machines")
    parser.add_argument("--out", type=Path, default=RAW_AUDIO_DIR, help="Destination directory (default: data/audio/raw)")
    parser.add_argument("--no-keep-zip", action="store_true", help="Delete .zip after successful extraction to save disk")
    parser.add_argument("--retries", type=int, default=5, help="Number of download retries on network error")
    parser.add_argument("--manual-help", action="store_true", help="Show manual download URLs and exit")
    args = parser.parse_args(argv)

    selected_machines = list(MACHINE_CATALOG.keys()) if args.all else args.machines

    if args.manual_help:
        print_manual_instructions(selected_machines, args.out)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"MIMII DG Downloader (Zenodo Record {ZENODO_RECORD_ID})")
    print(f"Destination: {args.out.resolve()}")
    print(f"Target machines: {', '.join(selected_machines)}")

    summary: Dict[str, str] = {}
    failed: List[str] = []

    for m in selected_machines:
        status, msg = download_mimii_dg_machine(
            machine=m,
            raw_dir=args.out,
            keep_zip=not args.no_keep_zip,
            retries=args.retries,
        )
        summary[m] = status
        print(f"[{status.upper()}] {msg}")
        if status == "failed":
            failed.append(m)

    print("\nSummary:")
    for m, st in summary.items():
        print(f"  {m:10s} : {st}")

    if failed:
        print(f"\n{len(failed)} machines failed. Showing manual download fallback links:")
        print_manual_instructions(failed, args.out)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
