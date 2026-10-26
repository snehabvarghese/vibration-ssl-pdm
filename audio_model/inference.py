"""
Standalone sound anomaly inference module.

Exposes analyse_audio_clip() for evaluating audio files or raw audio arrays.
Decoupled from web frameworks for clean downstream API and dashboard integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def analyse_audio_clip(
    audio_input: Union[str, Path, np.ndarray],
    sampling_rate: int = 16_000,
    machine_type: str = "fan",
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyse a sound clip for acoustic anomalies.

    Parameters
    ----------
    audio_input : Union[str, Path, np.ndarray]
        Path to a .wav file or 1-D numpy array of raw audio samples.
    sampling_rate : int
        Sampling frequency in Hz (defaults to 16,000 Hz).
    machine_type : str
        Machine category (e.g. 'fan', 'bearing').
    section : Optional[str]
        Optional section identifier (e.g. 'section_00').

    Returns
    -------
    Dict[str, Any]
        Dictionary with status, anomaly score, threshold, and diagnostics.
    """
    # Placeholder contract to be implemented in downstream steps
    return {
        "status": "HEALTHY",
        "machine_type": machine_type,
        "section": section,
        "anomaly_score": 0.0,
        "threshold": 1.0,
        "is_anomaly": False,
        "sampling_rate": sampling_rate,
    }
