"""
Segmentation and normalization.

SEGMENTATION
------------
A CNN needs fixed-length inputs, and a 10 s recording is both too long and too
few examples. We cut each recording into overlapping windows:

    |-------- window --------|
              |-------- window --------|
              ^ hop = window * (1 - overlap)

50 % overlap (the paper's setting) doubles the number of training segments and
makes the model robust to where a fault impulse falls inside the window.

IMPORTANT CONSEQUENCE: neighbouring windows share half their samples, so they
are *not* independent. Every segment therefore carries its `record_id`, and
train/val/test splits are made at record level (see `training/splits.py`).

NORMALIZATION
-------------
Per-window z-score: x -> (x - mean) / std.

Per-window (not global) because absolute vibration amplitude depends on sensor
gain, mounting and load; the *shape* of the signal carries the fault
information. Normalizing per window also means a Phase 2 live stream needs no
dataset-wide statistics to run inference.

INPUTS  : 1-D signal
OUTPUTS : (n_segments, window_size) float32 array
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import CFG, PreprocessConfig


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    hop_size: Optional[int] = None,
    overlap: Optional[float] = None,
    drop_last: bool = True,
) -> np.ndarray:
    """Cut a 1-D signal into overlapping fixed-length windows.

    Parameters
    ----------
    window_size  samples per window
    hop_size     step between window starts; if None it is derived from `overlap`
    overlap      fraction in [0, 1); ignored when `hop_size` is given
    drop_last    discard a trailing partial window (True) or zero-pad it (False)

    Returns
    -------
    (n_windows, window_size) array. Empty array with the right shape if the
    signal is shorter than one window.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if hop_size is None:
        ov = CFG.preprocess.overlap if overlap is None else overlap
        if not 0.0 <= ov < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {ov}")
        hop_size = max(1, int(round(window_size * (1.0 - ov))))

    if signal.size < window_size:
        if drop_last:
            return np.empty((0, window_size), dtype=np.float64)
        padded = np.zeros(window_size, dtype=np.float64)
        padded[: signal.size] = signal
        return padded[None, :]

    n = 1 + (signal.size - window_size) // hop_size
    # as_strided gives a zero-copy view; copy() makes the result contiguous and safe.
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        signal, shape=(n, window_size), strides=strides, writeable=False
    ).copy()

    if not drop_last:
        consumed = (n - 1) * hop_size + window_size
        remainder = signal.size - consumed
        if remainder > 0:
            tail = np.zeros((1, window_size), dtype=np.float64)
            tail[0, :remainder] = signal[consumed:]
            windows = np.vstack([windows, tail])
    return windows


def normalize_signal(
    x: np.ndarray,
    method: Optional[str] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize a single window or a batch of windows (last axis = time).

    method="zscore"  -> zero mean, unit variance per window
    method="minmax"  -> scaled to [-1, 1] per window
    method="none"    -> unchanged
    """
    method = method or CFG.preprocess.normalization
    x = np.asarray(x, dtype=np.float64)
    if method == "none":
        return x
    if method == "zscore":
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    if method == "minmax":
        lo = x.min(axis=-1, keepdims=True)
        hi = x.max(axis=-1, keepdims=True)
        return 2.0 * (x - lo) / (hi - lo + eps) - 1.0
    raise ValueError(f"unknown normalization {method!r}")


def drop_degenerate(
    segments: np.ndarray,
    min_std: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove flat/dead windows (sensor dropout) before normalization.

    Returns (kept_segments, kept_mask). Dividing a constant window by its std
    would produce NaNs, so these are dropped rather than normalized.
    """
    min_std = CFG.preprocess.min_segment_std if min_std is None else min_std
    stds = segments.std(axis=-1)
    mask = np.isfinite(segments).all(axis=-1) & (stds > min_std)
    return segments[mask], mask


def preprocess_record(
    signal: np.ndarray,
    sampling_rate: int,
    cfg: Optional[PreprocessConfig] = None,
) -> tuple[np.ndarray, int]:
    """Full per-record pipeline: filter -> downsample -> segment -> normalize.

    Returns (segments float32 of shape (n, window_size), effective sampling rate).
    """
    from preprocessing.filtering import downsample_signal, filter_signal

    cfg = cfg or CFG.preprocess
    filtered = filter_signal(signal, sampling_rate, cfg)
    reduced, fs = downsample_signal(filtered, sampling_rate, cfg.downsample_factor)

    window = cfg.window_size(sampling_rate)  # already accounts for the downsampling
    hop = cfg.hop_size(sampling_rate)
    segments = segment_signal(reduced, window_size=window, hop_size=hop)

    segments, _ = drop_degenerate(segments, cfg.min_segment_std)
    segments = normalize_signal(segments, cfg.normalization)
    return segments.astype(np.float32), fs
