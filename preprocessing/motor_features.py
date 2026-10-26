"""
Feature extraction module for Motor Current Signature Analysis (MCSA) and Temperature.

Extracts time-domain statistical features, spectral MCSA features, Park's vector
transform representations (when two phase currents are available), and thermal features.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats


def compute_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute standard statistical time-domain features from a 1-D signal array.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series array (e.g. motor current phase).

    Returns
    -------
    dict
        Dictionary containing RMS, peak, peak-to-peak, crest factor, kurtosis,
        skewness, shape factor, impulse factor, and margin factor.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) == 0:
        return {
            "rms": 0.0,
            "peak": 0.0,
            "peak_to_peak": 0.0,
            "crest_factor": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "shape_factor": 0.0,
            "impulse_factor": 0.0,
            "margin_factor": 0.0,
        }

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.ptp(x))
    abs_mean = float(np.mean(np.abs(x)))
    square_root_mean = float(np.mean(np.sqrt(np.abs(x))) ** 2)

    eps = 1e-12
    crest_factor = peak / (rms + eps)
    shape_factor = rms / (abs_mean + eps)
    impulse_factor = peak / (abs_mean + eps)
    margin_factor = peak / (square_root_mean + eps)

    # Scipy moments
    kurt = float(stats.kurtosis(x, fisher=True))  # 0 for normal distribution
    skew = float(stats.skew(x))

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "skewness": skew,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor,
    }


def compute_spectral_mcsa_features(
    signal: np.ndarray,
    sampling_rate: int = 64_000,
    expected_fundamental_hz: float = 50.0,
) -> Dict[str, float]:
    """Compute spectral features characteristic of Motor Current Signature Analysis.

    Identifies the fundamental electrical frequency, harmonic distortion,
    sideband energy (where bearing faults modulate the stator current),
    spectral centroid, and spectral entropy.

    Parameters
    ----------
    signal : np.ndarray
        1-D motor current signal.
    sampling_rate : int, default 64000
        Sampling frequency in Hz.
    expected_fundamental_hz : float, default 50.0
        Nominal grid / drive frequency in Hz.

    Returns
    -------
    dict
        Dictionary of spectral features.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = len(x)
    if n < 64:
        return {
            "fundamental_freq": expected_fundamental_hz,
            "fundamental_amp": 0.0,
            "thd": 0.0,
            "sideband_energy_ratio": 0.0,
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
        }

    # Hanning window to reduce spectral leakage
    window = np.hanning(n)
    fft_vals = np.fft.rfft((x - np.mean(x)) * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    win_sum = float(np.sum(window))
    # Single-sided peak amplitude scaled by coherent gain of window
    mag = (2.0 * np.abs(fft_vals)) / (win_sum + 1e-12)
    power = mag ** 2
    total_power = np.sum(power) + 1e-12

    # Locate fundamental frequency near expected_fundamental_hz (+/- 10 Hz search band)
    fund_band_mask = (freqs >= max(5.0, expected_fundamental_hz - 10.0)) & (
        freqs <= expected_fundamental_hz + 10.0
    )
    if np.any(fund_band_mask):
        sub_indices = np.where(fund_band_mask)[0]
        fund_idx = sub_indices[np.argmax(mag[fund_band_mask])]
        fundamental_freq = float(freqs[fund_idx])
        fundamental_amp = float(mag[fund_idx])
    else:
        fundamental_freq = expected_fundamental_hz
        fund_idx = int(np.argmin(np.abs(freqs - expected_fundamental_hz)))
        fundamental_amp = float(mag[fund_idx])

    # Harmonics: 2nd, 3rd, 4th, 5th
    harmonics_power = 0.0
    for h in [2, 3, 4, 5]:
        target_f = fundamental_freq * h
        h_mask = (freqs >= target_f - 4.0) & (freqs <= target_f + 4.0)
        if np.any(h_mask):
            harmonics_power += float(np.max(power[h_mask]))

    thd = float(np.sqrt(harmonics_power) / (fundamental_amp + 1e-12))

    # Sideband energy ratio: energy in [f_e - 150 Hz, f_e + 150 Hz] excluding [f_e - 5 Hz, f_e + 5 Hz]
    sideband_mask = (
        (freqs >= max(1.0, fundamental_freq - 150.0))
        & (freqs <= fundamental_freq + 150.0)
        & ((freqs < fundamental_freq - 5.0) | (freqs > fundamental_freq + 5.0))
    )
    sideband_power = float(np.sum(power[sideband_mask]))
    sideband_ratio = float(sideband_power / total_power)

    # Spectral centroid
    spectral_centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))

    # Spectral entropy
    p_norm = power / total_power
    p_norm = p_norm[p_norm > 0]
    spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm)) / np.log2(len(freqs)))

    return {
        "fundamental_freq": fundamental_freq,
        "fundamental_amp": fundamental_amp,
        "thd": thd,
        "sideband_energy_ratio": sideband_ratio,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
    }


def compute_parks_vector_features(
    phase1: np.ndarray,
    phase2: np.ndarray,
) -> Dict[str, float]:
    """Compute Park's Vector Transform features from two phase currents (I_u, I_v).

    For balanced three-phase systems where i_u + i_v + i_w = 0:
      i_alpha = sqrt(3/2) * i_u
      i_beta  = 1/sqrt(2) * (i_u + 2 * i_v)
      i_park  = sqrt(i_alpha^2 + i_beta^2)

    Mechanical bearing defects cause characteristic radius modulations and
    elliptical distortions in the Park's vector locus.

    Parameters
    ----------
    phase1 : np.ndarray
        Phase current 1 (I_u).
    phase2 : np.ndarray
        Phase current 2 (I_v).

    Returns
    -------
    dict
        Park's modulus mean, std, peak-to-peak, and eccentricity ratio.
    """
    u = np.asarray(phase1, dtype=np.float64).ravel()
    v = np.asarray(phase2, dtype=np.float64).ravel()
    length = min(len(u), len(v))
    if length == 0:
        return {
            "park_mean": 0.0,
            "park_std": 0.0,
            "park_ptp": 0.0,
            "park_eccentricity": 0.0,
        }

    u = u[:length]
    v = v[:length]

    alpha = np.sqrt(1.5) * u
    beta = (u + 2.0 * v) / np.sqrt(2.0)
    radius = np.sqrt(alpha ** 2 + beta ** 2)

    park_mean = float(np.mean(radius))
    park_std = float(np.std(radius))
    park_ptp = float(np.ptp(radius))
    eps = 1e-12
    park_eccentricity = park_std / (park_mean + eps)

    return {
        "park_mean": park_mean,
        "park_std": park_std,
        "park_ptp": park_ptp,
        "park_eccentricity": park_eccentricity,
    }


def compute_temperature_features(temp_signal: Union[np.ndarray, float]) -> Dict[str, float]:
    """Extract thermal features preserving units (Celsius) and physical meaning.

    Parameters
    ----------
    temp_signal : np.ndarray or float
        Scalar temperature or 1-D temperature series.

    Returns
    -------
    dict
        Dictionary with temp_mean, temp_std, temp_min, temp_max, temp_slope.
    """
    if isinstance(temp_signal, (int, float, np.floating, np.integer)):
        val = float(temp_signal)
        return {
            "temp_mean": val,
            "temp_std": 0.0,
            "temp_min": val,
            "temp_max": val,
            "temp_slope": 0.0,
        }

    t = np.asarray(temp_signal, dtype=np.float64).ravel()
    if len(t) == 0:
        return {
            "temp_mean": 45.0,  # nominal Paderborn bench temperature
            "temp_std": 0.0,
            "temp_min": 45.0,
            "temp_max": 45.0,
            "temp_slope": 0.0,
        }

    mean_val = float(np.mean(t))
    std_val = float(np.std(t))
    min_val = float(np.min(t))
    max_val = float(np.max(t))

    slope = 0.0
    if len(t) > 1:
        x_axis = np.linspace(0, 1, len(t))
        # Simple linear regression slope
        cov = np.cov(x_axis, t)
        if cov.shape == (2, 2) and cov[0, 0] > 0:
            slope = float(cov[0, 1] / cov[0, 0])

    return {
        "temp_mean": mean_val,
        "temp_std": std_val,
        "temp_min": min_val,
        "temp_max": max_val,
        "temp_slope": slope,
    }


def extract_multimodal_features(
    current_phase1: np.ndarray,
    current_phase2: Optional[np.ndarray] = None,
    temperature: Optional[Union[np.ndarray, float]] = None,
    sampling_rate: int = 64_000,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Extract and combine motor current and temperature features into a standardized vector.

    Parameters
    ----------
    current_phase1 : np.ndarray
        Phase 1 motor current signal.
    current_phase2 : np.ndarray, optional
        Phase 2 motor current signal (if available).
    temperature : np.ndarray or float, optional
        Recorded temperature readings.
    sampling_rate : int, default 64000
        Current sampling rate in Hz.

    Returns
    -------
    feature_vector : np.ndarray
        1-D numpy array of extracted features.
    feature_names : list of str
        List of feature names matching feature_vector.
    features_dict : dict
        Flat dictionary of key-value pairs for all extracted features.
    """
    features: Dict[str, float] = {}

    # Phase 1 time & spectral features
    t1 = compute_time_domain_features(current_phase1)
    for k, v in t1.items():
        features[f"p1_{k}"] = v

    s1 = compute_spectral_mcsa_features(current_phase1, sampling_rate=sampling_rate)
    for k, v in s1.items():
        features[f"p1_{k}"] = v

    # Phase 2 if present
    if current_phase2 is not None and len(current_phase2) > 0:
        t2 = compute_time_domain_features(current_phase2)
        for k, v in t2.items():
            features[f"p2_{k}"] = v

        s2 = compute_spectral_mcsa_features(current_phase2, sampling_rate=sampling_rate)
        for k, v in s2.items():
            features[f"p2_{k}"] = v

        p_feat = compute_parks_vector_features(current_phase1, current_phase2)
        for k, v in p_feat.items():
            features[k] = v
    else:
        # Zero-pad Phase 2 / Park fields so vector length is fixed and stable
        for k in t1.keys():
            features[f"p2_{k}"] = 0.0
        for k in s1.keys():
            features[f"p2_{k}"] = 0.0
        for k in ["park_mean", "park_std", "park_ptp", "park_eccentricity"]:
            features[k] = 0.0

    # Temperature features
    t_feat = compute_temperature_features(temperature if temperature is not None else 45.0)
    for k, v in t_feat.items():
        features[k] = v

    names = sorted(features.keys())
    vec = np.array([features[name] for name in names], dtype=np.float64)
    return vec, names, features
