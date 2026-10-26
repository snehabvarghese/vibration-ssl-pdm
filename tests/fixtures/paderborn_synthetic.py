"""
SYNTHETIC TEST FIXTURE (FOR SOFTWARE & CI TESTING ONLY).

CRITICAL NOTICE:
This module generates SYNTHETIC motor current and temperature data solely for
software testing, pipeline verification, and unit test suites when full multi-gigabyte
Paderborn RAR files are not present in CI environments.
DO NOT USE SYNTHETIC DATA TO REPORT REAL MODEL PERFORMANCE OR RESEARCH RESULTS.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio


def generate_synthetic_motor_current(
    condition: str = "Normal",
    duration_seconds: float = 2.0,
    sampling_rate: int = 64_000,
    fund_freq_hz: float = 50.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate physically-grounded synthetic 2-phase motor currents and temperature.

    Parameters
    ----------
    condition : str
        "Normal", "Outer_Race", or "Inner_Race".
    duration_seconds : float, default 2.0
        Duration in seconds.
    sampling_rate : int, default 64000
        Sampling frequency in Hz.
    fund_freq_hz : float, default 50.0
        Fundamental electrical supply frequency.
    seed : int, default 42
        Random seed for repeatability.

    Returns
    -------
    phase1 : np.ndarray
        Synthetic I_u current signal.
    phase2 : np.ndarray
        Synthetic I_v current signal (shifted by -120 degrees).
    temperature : float
        Synthetic oil temperature in Celsius (around 45-50 deg C).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)

    # 1. Fundamental electrical supply current (~1.5 A peak)
    i1 = 1.5 * np.cos(2 * np.pi * fund_freq_hz * t)
    i2 = 1.5 * np.cos(2 * np.pi * fund_freq_hz * t - (2 * np.pi / 3))

    # 2. Harmonics (3rd, 5th from inverter switching)
    i1 += 0.05 * np.cos(2 * np.pi * (3 * fund_freq_hz) * t)
    i2 += 0.05 * np.cos(2 * np.pi * (3 * fund_freq_hz) * t - np.pi)

    temp_val = 46.5 + rng.normal(0, 0.4)

    # 3. Add fault-induced modulation sidebands
    # For a 6203 bearing at 1500 rpm (25 Hz shaft):
    # BPFO ~ 76 Hz, BPFI ~ 124 Hz
    if condition == "Outer_Race":
        f_fault = 76.0
        # Torque oscillation modulates motor current: (f_e +/- f_fault)
        mod = 0.15 * np.cos(2 * np.pi * f_fault * t)
        # Periodic impulse bursts
        impulses = np.exp(-((t % (1.0 / f_fault)) / 0.002) ** 2) * 0.1
        i1 = i1 * (1.0 + mod) + impulses
        i2 = i2 * (1.0 + mod) + impulses
        temp_val += 1.2  # Slight thermal elevation under sustained degradation
    elif condition == "Inner_Race":
        f_fault = 124.0
        mod = 0.18 * np.cos(2 * np.pi * f_fault * t)
        impulses = np.exp(-((t % (1.0 / f_fault)) / 0.0015) ** 2) * 0.12
        i1 = i1 * (1.0 + mod) + impulses
        i2 = i2 * (1.0 + mod) + impulses
        temp_val += 1.8

    # 4. Add sensor noise
    i1 += rng.normal(0, 0.02, size=n_samples)
    i2 += rng.normal(0, 0.02, size=n_samples)

    return i1, i2, float(temp_val)


def create_synthetic_paderborn_mat(
    filename: str = "N15_M07_F10_K001_1.mat",
    condition: str = "Normal",
    duration_seconds: float = 1.0,
    sampling_rate: int = 64_000,
    seed: int = 42,
) -> bytes:
    """Pack synthetic motor data into a Paderborn-compliant MATLAB .mat byte buffer."""
    i1, i2, temp_val = generate_synthetic_motor_current(
        condition=condition,
        duration_seconds=duration_seconds,
        sampling_rate=sampling_rate,
        seed=seed,
    )

    stem = Path(filename).stem

    # Construct nested structure mirroring Paderborn format:
    # record['Y'][0, 0] with structured array containing 'Name' and 'Data'
    dt_channel = np.dtype([
        ("Name", "O"),
        ("Data", "O"),
        ("Unit", "O"),
        ("Raster", "O"),
    ])

    channels = np.empty((1, 4), dtype=dt_channel)
    channels[0, 0] = ("phase_current_1", i1.reshape(-1, 1), "A", 64000)
    channels[0, 1] = ("phase_current_2", i2.reshape(-1, 1), "A", 64000)
    channels[0, 2] = ("temp_2_bearing_module", np.array([[temp_val]]), "C", 1)
    channels[0, 3] = ("vibration_1", np.zeros((len(i1), 1)), "g", 64000)

    dt_y = np.dtype([("Y", "O")])
    main_struct = np.empty((1, 1), dtype=dt_y)
    main_struct[0, 0] = (channels,)

    buf = io.BytesIO()
    sio.savemat(buf, {stem: main_struct})
    return buf.getvalue()


def create_synthetic_dataset_manifest() -> List[Dict[str, Any]]:
    """Return a list of synthetic metadata records covering healthy, outer-race, and inner-race."""
    records = [
        # Healthy bearings (K001 - K003)
        {"filename": "N15_M07_F10_K001_1.mat", "condition": "Normal", "bearing_id": "K001", "seed": 101},
        {"filename": "N15_M07_F10_K001_2.mat", "condition": "Normal", "bearing_id": "K001", "seed": 102},
        {"filename": "N15_M07_F10_K002_1.mat", "condition": "Normal", "bearing_id": "K002", "seed": 103},
        {"filename": "N15_M07_F10_K003_1.mat", "condition": "Normal", "bearing_id": "K003", "seed": 104},
        # Outer race faults (KA01 - KA03)
        {"filename": "N15_M07_F10_KA01_1.mat", "condition": "Outer_Race", "bearing_id": "KA01", "seed": 201},
        {"filename": "N15_M07_F10_KA01_2.mat", "condition": "Outer_Race", "bearing_id": "KA01", "seed": 202},
        {"filename": "N15_M07_F10_KA02_1.mat", "condition": "Outer_Race", "bearing_id": "KA02", "seed": 203},
        {"filename": "N15_M07_F10_KA03_1.mat", "condition": "Outer_Race", "bearing_id": "KA03", "seed": 204},
        # Inner race faults (KI01 - KI03)
        {"filename": "N15_M07_F10_KI01_1.mat", "condition": "Inner_Race", "bearing_id": "KI01", "seed": 301},
        {"filename": "N15_M07_F10_KI01_2.mat", "condition": "Inner_Race", "bearing_id": "KI01", "seed": 302},
        {"filename": "N15_M07_F10_KI02_1.mat", "condition": "Inner_Race", "bearing_id": "KI02", "seed": 303},
        {"filename": "N15_M07_F10_KI03_1.mat", "condition": "Inner_Race", "bearing_id": "KI03", "seed": 304},
    ]
    return records
