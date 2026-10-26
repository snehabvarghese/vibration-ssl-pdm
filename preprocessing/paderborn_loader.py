"""
Modular data loader for the Paderborn University Bearing Dataset.

Extracts motor current signals (phase_current_1, phase_current_2) and
temperature (temp_2_bearing_module) from official Paderborn .mat files,
parses operating condition and bearing fault metadata, validates signal integrity,
and segments recordings into analysis windows without cross-record data leakage.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio


@dataclass
class PaderbornRecord:
    """Structured container for a single Paderborn measurement run."""

    filename: str
    bearing_id: str
    condition_label: str  # "Normal" | "Outer_Race" | "Inner_Race" | "Ball_Fault"
    fault_type: str       # "healthy" | "artificial" | "accelerated_lifetime"
    operating_condition: str  # e.g., "N15_M07_F10"
    speed_rpm: int
    load_torque_nm: float
    radial_force_n: float
    sampling_rate: int
    phase_current_1: np.ndarray
    phase_current_2: Optional[np.ndarray] = None
    temperature_celsius: Optional[float] = None
    vibration: Optional[np.ndarray] = None


def parse_paderborn_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata (operating condition, bearing id, label) from Paderborn filename.

    Example formats:
      N15_M07_F10_K001_1.mat
      N09_M07_F10_KA01_2.mat
      N15_M01_F10_KI04_1.mat
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    meta = {
        "operating_condition": "Unknown",
        "speed_rpm": 1500,
        "load_torque_nm": 0.7,
        "radial_force_n": 1000.0,
        "bearing_id": "Unknown",
        "condition_label": "Normal",
        "fault_type": "healthy",
    }

    if len(parts) >= 4:
        # Operating condition code
        meta["operating_condition"] = f"{parts[0]}_{parts[1]}_{parts[2]}"
        
        # Speed: N15 -> 1500 rpm, N09 -> 900 rpm
        if parts[0].startswith("N"):
            try:
                meta["speed_rpm"] = int(parts[0][1:]) * 100
            except ValueError:
                pass
        # Torque: M07 -> 0.7 Nm, M01 -> 0.1 Nm
        if parts[1].startswith("M"):
            try:
                meta["load_torque_nm"] = float(parts[1][1:]) / 10.0
            except ValueError:
                pass
        # Radial force: F10 -> 1000 N, F04 -> 400 N
        if parts[2].startswith("F"):
            try:
                meta["radial_force_n"] = float(parts[2][1:]) * 100.0
            except ValueError:
                pass

        bearing_id = parts[3].upper()
        meta["bearing_id"] = bearing_id

        # Categorize condition label based on bearing code prefix
        if bearing_id.startswith("K0") or bearing_id.startswith("K"):
            if re.match(r"^K00[1-6]$", bearing_id):
                meta["condition_label"] = "Normal"
                meta["fault_type"] = "healthy"
            elif bearing_id.startswith("KA"):
                meta["condition_label"] = "Outer_Race"
                meta["fault_type"] = "artificial" if int(re.sub(r"\D", "", bearing_id) or 0) <= 12 else "accelerated_lifetime"
            elif bearing_id.startswith("KI"):
                meta["condition_label"] = "Inner_Race"
                meta["fault_type"] = "artificial" if int(re.sub(r"\D", "", bearing_id) or 0) <= 12 else "accelerated_lifetime"
            elif bearing_id.startswith("KB"):
                meta["condition_label"] = "Ball_Fault"
                meta["fault_type"] = "artificial"
            else:
                meta["condition_label"] = "Normal"
        elif bearing_id.startswith("KA"):
            meta["condition_label"] = "Outer_Race"
        elif bearing_id.startswith("KI"):
            meta["condition_label"] = "Inner_Race"
        elif bearing_id.startswith("KB"):
            meta["condition_label"] = "Ball_Fault"

    return meta


def load_paderborn_mat(
    file_path_or_bytes: Union[str, Path, bytes, io.BytesIO],
    filename_hint: str = "N15_M07_F10_K001_1.mat",
) -> PaderbornRecord:
    """Parse a Paderborn .mat file into a validated PaderbornRecord.

    Extracts phase_current_1, phase_current_2, temp_2_bearing_module,
    and vibration signals, navigating both standard nested MATLAB structs
    and flat dictionary representations.

    Parameters
    ----------
    file_path_or_bytes : str, Path, bytes, or BytesIO
        Input file location or raw file bytes.
    filename_hint : str, default "N15_M07_F10_K001_1.mat"
        Filename used for metadata parsing if bytes/buffer are provided.

    Returns
    -------
    PaderbornRecord
        Parsed and validated record object.
    """
    filename = filename_hint
    if isinstance(file_path_or_bytes, (str, Path)):
        p = Path(file_path_or_bytes)
        filename = p.name
        mat = sio.loadmat(str(p), squeeze_me=False)
    elif isinstance(file_path_or_bytes, bytes):
        mat = sio.loadmat(io.BytesIO(file_path_or_bytes), squeeze_me=False)
    elif isinstance(file_path_or_bytes, io.BytesIO):
        mat = sio.loadmat(file_path_or_bytes, squeeze_me=False)
    else:
        raise TypeError(f"Unsupported input type: {type(file_path_or_bytes)}")

    meta = parse_paderborn_filename(filename)

    # Search for channels in mat structure
    p1: Optional[np.ndarray] = None
    p2: Optional[np.ndarray] = None
    temp_val: Optional[float] = None
    vib: Optional[np.ndarray] = None
    sampling_rate: int = 64_000

    # 1. Standard Paderborn nested struct layout:
    # Top key is usually the file stem or single non-dunder key
    non_meta_keys = [k for k in mat.keys() if not k.startswith("__")]

    for key in non_meta_keys:
        val = mat[key]
        # In standard PU dataset, mat[key]['Y'][0, 0] holds array of channel structs
        if isinstance(val, np.ndarray) and val.dtype.names and "Y" in val.dtype.names:
            y_struct = val["Y"][0, 0]
            # y_struct is usually a structured array where channels are rows or sub-elements
            for ch in y_struct[0]:
                names = ch.dtype.names or ()
                ch_name = ""
                ch_data = None
                
                for n in names:
                    if n.lower() in ("name", "channel"):
                        ch_name = str(ch[n][0]) if len(ch[n]) > 0 else ""
                    elif n.lower() in ("data", "value"):
                        ch_data = ch[n]

                if ch_name and ch_data is not None:
                    ch_arr = np.asarray(ch_data, dtype=np.float64).ravel()
                    lower_name = ch_name.lower()
                    if "phase_current_1" in lower_name or "current_1" in lower_name or "i_u" in lower_name:
                        p1 = ch_arr
                    elif "phase_current_2" in lower_name or "current_2" in lower_name or "i_v" in lower_name:
                        p2 = ch_arr
                    elif "temp" in lower_name:
                        temp_val = float(np.mean(ch_arr))
                    elif "vibration" in lower_name:
                        vib = ch_arr

    # 2. Check for flat keys if nested extraction didn't resolve channels
    if p1 is None:
        for k in non_meta_keys:
            kl = k.lower()
            arr = np.asarray(mat[k], dtype=np.float64).ravel()
            if "phase_current_1" in kl or "current_1" in kl or kl.endswith("_cu1") or kl.endswith("_iu"):
                p1 = arr
            elif "phase_current_2" in kl or "current_2" in kl or kl.endswith("_cu2") or kl.endswith("_iv"):
                p2 = arr
            elif "temp" in kl:
                temp_val = float(np.mean(arr))
            elif "vibration" in kl:
                vib = arr

    # If still none, check if there is an array named 'current' or 'signal'
    if p1 is None and "current" in mat:
        p1 = np.asarray(mat["current"], dtype=np.float64).ravel()
    if p1 is None and "signal" in mat:
        p1 = np.asarray(mat["signal"], dtype=np.float64).ravel()

    if p1 is None or len(p1) == 0:
        raise ValueError(
            f"Could not locate motor current channel (phase_current_1) in {filename}. "
            f"Available keys: {non_meta_keys}"
        )

    # Signal validation: reject dead or NaN-dominated signals
    nan_ratio = np.isnan(p1).mean()
    if nan_ratio > 0.01:
        raise ValueError(f"Signal rejected: {nan_ratio*100:.1f}% NaN values in current channel")
    p1 = np.nan_to_num(p1, nan=0.0)

    if np.std(p1) < 1e-6:
        raise ValueError("Signal rejected: motor current channel is flat/dead (< 1e-6 std)")

    # Default thermal state for Paderborn test rig if not recorded in file
    if temp_val is None:
        temp_val = 45.0  # nominal regulated lubricant temperature in PU bench

    return PaderbornRecord(
        filename=filename,
        bearing_id=meta["bearing_id"],
        condition_label=meta["condition_label"],
        fault_type=meta["fault_type"],
        operating_condition=meta["operating_condition"],
        speed_rpm=meta["speed_rpm"],
        load_torque_nm=meta["load_torque_nm"],
        radial_force_n=meta["radial_force_n"],
        sampling_rate=sampling_rate,
        phase_current_1=p1,
        phase_current_2=p2,
        temperature_celsius=temp_val,
        vibration=vib,
    )


def segment_record(
    record: PaderbornRecord,
    window_seconds: float = 0.5,
    overlap: float = 0.5,
) -> List[Dict[str, Any]]:
    """Segment a PaderbornRecord into analysis windows with strictly preserved record metadata.

    Parameters
    ----------
    record : PaderbornRecord
        Source record.
    window_seconds : float, default 0.5
        Length of window in seconds (at 64 kHz, 0.5 s = 32,000 samples).
    overlap : float, default 0.5
        Fractional overlap between consecutive windows.

    Returns
    -------
    list of dict
        Each dict contains windowed phase_current_1, phase_current_2,
        temperature, labels, and original bearing/recording identifiers.
    """
    fs = record.sampling_rate
    win_len = int(round(window_seconds * fs))
    step = int(round(win_len * (1.0 - overlap)))
    step = max(1, step)

    sig1 = record.phase_current_1
    sig2 = record.phase_current_2
    n_samples = len(sig1)

    segments: List[Dict[str, Any]] = []
    idx = 0
    seg_idx = 0
    while idx + win_len <= n_samples:
        w1 = sig1[idx : idx + win_len]
        w2 = sig2[idx : idx + win_len] if sig2 is not None else None

        segments.append({
            "segment_id": f"{Path(record.filename).stem}_win{seg_idx:04d}",
            "record_filename": record.filename,
            "bearing_id": record.bearing_id,
            "condition_label": record.condition_label,
            "fault_type": record.fault_type,
            "operating_condition": record.operating_condition,
            "phase_current_1": w1,
            "phase_current_2": w2,
            "temperature_celsius": record.temperature_celsius,
            "sampling_rate": fs,
        })
        idx += step
        seg_idx += 1

    return segments
