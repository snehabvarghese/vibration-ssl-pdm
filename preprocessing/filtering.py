"""
Filtering and downsampling (first two stages of the preprocessing pipeline).

WHY FILTER
----------
Raw accelerometer data contains (a) a DC/very-low-frequency component from
mounting and thermal drift, and (b) high-frequency content that carries little
bearing information but a lot of noise. A zero-phase Butterworth band-pass
keeps the band where bearing fault impulses and their harmonics live, without
shifting the impulses in time (zero-phase = no group delay, which matters
because the fault signature *is* the impulse timing).

WHY DOWNSAMPLE
--------------
A 1 s window at 12 kHz is 12 000 samples; at 3 kHz it is 3 000. Shorter inputs
mean a smaller, faster 1-D CNN that trains on a CPU, at the cost of discarding
content above 1.5 kHz. `scipy.signal.decimate` low-pass filters before
dropping samples, so this does not alias.

INPUTS  : 1-D float array + sampling rate
OUTPUTS : 1-D float array (same rate for `filter_signal`, reduced rate for
          `downsample_signal`)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

from config import CFG, PreprocessConfig


def design_filter(
    sampling_rate: int,
    filter_type: str,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Design a Butterworth filter as second-order sections (numerically stable).

    Cutoffs are clipped just below Nyquist so that a configuration that is too
    aggressive for a given sampling rate degrades gracefully instead of raising.
    """
    nyq = sampling_rate / 2.0
    if filter_type == "bandpass":
        low = max(low_hz, 1e-3) / nyq
        high = min(high_hz, nyq * 0.99) / nyq
        if not 0 < low < high < 1:
            raise ValueError(
                f"invalid band {low_hz}-{high_hz} Hz for fs={sampling_rate} Hz"
            )
        return butter(order, [low, high], btype="bandpass", output="sos")
    if filter_type == "highpass":
        return butter(order, max(low_hz, 1e-3) / nyq, btype="highpass", output="sos")
    if filter_type == "lowpass":
        return butter(order, min(high_hz, nyq * 0.99) / nyq, btype="lowpass", output="sos")
    raise ValueError(f"unknown filter_type {filter_type!r}")


def filter_signal(
    signal: np.ndarray,
    sampling_rate: int,
    cfg: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Apply the configured zero-phase band-pass filter.

    Returns the signal unchanged when filtering is disabled or when the signal
    is too short for the filter's transient length (filtfilt requirement).
    """
    cfg = cfg or CFG.preprocess
    if not cfg.filter_enabled:
        return signal

    sos = design_filter(
        sampling_rate, cfg.filter_type, cfg.filter_low_hz, cfg.filter_high_hz, cfg.filter_order
    )
    # sosfiltfilt needs the signal to be longer than ~3 * the filter order.
    if signal.size <= 3 * (2 * cfg.filter_order + 1):
        return signal
    return np.asarray(sosfiltfilt(sos, signal))


def downsample_signal(
    signal: np.ndarray,
    sampling_rate: int,
    factor: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Anti-aliased decimation by an integer factor.

    Returns (downsampled_signal, new_sampling_rate).
    """
    factor = factor if factor is not None else CFG.preprocess.downsample_factor
    if factor <= 1:
        return signal, sampling_rate
    out = np.asarray(decimate(signal, factor, ftype="fir", zero_phase=True))
    return out, sampling_rate // factor
