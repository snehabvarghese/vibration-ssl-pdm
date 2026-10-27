"""
Feature extraction and signal preprocessing for sound anomaly detection.

Includes:
  * load_audio: loads, resamples to 16 kHz, converts to mono, pads/trims to fixed 10 s, returns float32.
  * log_mel: computes log-Mel spectrogram in dB using librosa.
  * stack_frames: DCASE-baseline style frame stacking into vectors of size n_mels * context_frames.
  * AudioNormalizer: per-machine standard scaler fitted strictly on normal training data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import scipy.io.wavfile

from audio_model.config_audio import AUDIO_CFG


def load_audio(
    path: Union[str, Path],
    sr: int = 16_000,
    duration_s: float = 10.0,
) -> np.ndarray:
    """Load audio file as mono, resampled to sr, fixed duration (pad/trim), float32.

    Parameters
    ----------
    path : Union[str, Path]
        Path to audio file (.wav).
    sr : int
        Target sampling rate (default: 16,000 Hz).
    duration_s : float
        Target clip duration in seconds (default: 10.0 s).

    Returns
    -------
    np.ndarray
        1-D float32 audio signal of length int(sr * duration_s).
    """
    path_str = str(path)
    target_len = int(round(sr * duration_s))

    try:
        # Fast path: direct read via scipy for native 16kHz PCM wav files
        file_sr, data = scipy.io.wavfile.read(path_str)
        if file_sr == sr and (data.ndim == 1 or data.shape[1] == 1):
            if data.ndim == 2:
                data = data[:, 0]
            if data.dtype == np.int16:
                y = (data / 32768.0).astype(np.float32)
            elif data.dtype == np.int32:
                y = (data / 2147483648.0).astype(np.float32)
            else:
                y = data.astype(np.float32)
        else:
            y, _ = librosa.load(path_str, sr=sr, mono=True, dtype=np.float32)
    except Exception:
        # Fallback to librosa
        y, _ = librosa.load(path_str, sr=sr, mono=True, dtype=np.float32)

    cur_len = len(y)
    if cur_len < target_len:
        y = np.pad(y, (0, target_len - cur_len), mode="constant", constant_values=0.0)
    elif cur_len > target_len:
        y = y[:target_len]

    return y.astype(np.float32)


def log_mel(
    y: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    power: float = 2.0,
    f_min: float = 50.0,
    f_max: float = 8000.0,
    ref: float = 1.0,
) -> np.ndarray:
    """Compute Log-Mel Spectrogram in dB.

    Parameters
    ----------
    y : np.ndarray
        1-D audio signal.
    sr : int
        Audio sampling rate.
    n_mels : int
        Number of Mel filter bands.
    n_fft : int
        FFT window length.
    hop_length : int
        Hop length between STFT frames.
    power : float
        Exponent for magnitude spectrogram (2.0 = power, 1.0 = energy).
    f_min : float
        Minimum frequency in Hz.
    f_max : float
        Maximum frequency in Hz.
    ref : float
        Reference power for dB calculation (ref=1.0 preserves absolute signal power).

    Returns
    -------
    np.ndarray
        2-D array of shape (n_mels, n_frames), float32.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected 1-D audio array, got shape {y.shape}")

    mels = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        power=power,
        fmin=f_min,
        fmax=f_max,
    )
    # Convert to decibels (log-power): 10 * log10(mels / ref)
    logmel = librosa.power_to_db(mels, ref=ref)
    return logmel.astype(np.float32)


def stack_frames(
    logmel: np.ndarray,
    context_frames: int = 5,
    frame_hop: int = 1,
) -> np.ndarray:
    """Stack adjacent temporal frames into vectors of size n_mels * context_frames.

    Follows the standard DCASE Task 2 baseline representation:
    Each time index t captures [frame_t, frame_{t+1}, ..., frame_{t+context_frames-1}].

    Parameters
    ----------
    logmel : np.ndarray
        Log-Mel spectrogram of shape (n_mels, n_frames) or (N, n_mels, n_frames).
    context_frames : int
        Number of consecutive frames to stack (default: 5).
    frame_hop : int
        Hop stride between consecutive stacked vectors (default: 1).

    Returns
    -------
    np.ndarray
        If input is (n_mels, n_frames):
            Output is (n_vectors, n_mels * context_frames).
        If input is (N, n_mels, n_frames):
            Output is (N, n_vectors, n_mels * context_frames).
    """
    if logmel.ndim == 2:
        n_mels, n_frames = logmel.shape
        if n_frames < context_frames:
            raise ValueError(
                f"Number of frames ({n_frames}) must be >= context_frames ({context_frames})"
            )
        n_vectors = (n_frames - context_frames) // frame_hop + 1
        stacked = np.empty((n_vectors, n_mels * context_frames), dtype=np.float32)
        for i in range(n_vectors):
            start = i * frame_hop
            end = start + context_frames
            # Slices (n_mels, context_frames) and flattens
            stacked[i] = logmel[:, start:end].flatten(order="C")
        return stacked

    elif logmel.ndim == 3:
        N, n_mels, n_frames = logmel.shape
        if n_frames < context_frames:
            raise ValueError(
                f"Number of frames ({n_frames}) must be >= context_frames ({context_frames})"
            )
        n_vectors = (n_frames - context_frames) // frame_hop + 1
        stacked = np.empty((N, n_vectors, n_mels * context_frames), dtype=np.float32)
        for b in range(N):
            for i in range(n_vectors):
                start = i * frame_hop
                end = start + context_frames
                stacked[b, i] = logmel[b, :, start:end].flatten(order="C")
        return stacked

    else:
        raise ValueError(f"Expected 2-D or 3-D logmel array, got shape {logmel.shape}")


class AudioNormalizer:
    """Per-machine standard scaler fitted strictly on normal training data.

    Computes feature-wise mean and std across training normal clips and
    normalizes validation and test features without distribution leakage.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> AudioNormalizer:
        """Fit mean and std on normal training array.

        Parameters
        ----------
        X : np.ndarray
            Features array. Supported shapes:
              * (N, n_mels, n_frames) -> computes mean/std across N and n_frames per mel band,
              * (N, feature_dim) or (N, T, feature_dim) -> computes mean/std per feature dimension.
        """
        if X.ndim == 3 and X.shape[1] == AUDIO_CFG.feature.n_mels:
            # (N, n_mels, n_frames) -> normalize per mel band: (1, n_mels, 1)
            self.mean = np.mean(X, axis=(0, 2), keepdims=True).astype(np.float32)
            self.std = (np.std(X, axis=(0, 2), keepdims=True) + self.eps).astype(np.float32)
        elif X.ndim == 2:
            # (N, D) -> normalize per column feature: (1, D)
            self.mean = np.mean(X, axis=0, keepdims=True).astype(np.float32)
            self.std = (np.std(X, axis=0, keepdims=True) + self.eps).astype(np.float32)
        elif X.ndim == 3:
            # (N, T, D) -> normalize across samples and time steps: (1, 1, D)
            self.mean = np.mean(X, axis=(0, 1), keepdims=True).astype(np.float32)
            self.std = (np.std(X, axis=(0, 1), keepdims=True) + self.eps).astype(np.float32)
        else:
            raise ValueError(f"Unsupported array shape for normalization: {X.shape}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize features using fitted statistics."""
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("AudioNormalizer must be fitted on training data before calling transform()")
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted normalizer statistics to .npz file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted AudioNormalizer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, eps=self.eps)

    @classmethod
    def load(cls, path: Union[str, Path]) -> AudioNormalizer:
        """Load fitted normalizer statistics from .npz file."""
        data = np.load(str(path))
        normalizer = cls(eps=float(data.get("eps", 1e-8)))
        normalizer.mean = data["mean"]
        normalizer.std = data["std"]
        normalizer.is_fitted = True
        return normalizer
