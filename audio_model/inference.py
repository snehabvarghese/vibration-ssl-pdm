"""
Standalone inference module for acoustic anomaly detection (Dense Autoencoder).

Provides:
  * predict_audio(audio, sampling_rate=None, machine_type="fan", threshold_percentile=95) -> dict
  * CLI interface: python -m audio_model.inference <path/to/clip.wav> --machine fan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import torch

from audio_model.config_audio import (
    AUDIO_CFG,
    CHECKPOINTS_AUDIO_DIR,
    get_device,
)
from audio_model.models.autoencoder import DenseAutoencoder
from audio_model.preprocessing.features import log_mel, stack_frames

# --------------------------------------------------------------------------
# Model & Scaler Cache
# --------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def get_cached_model(
    machine_type: str,
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load model checkpoint, normalization stats, and thresholds once, caching in memory.

    Parameters
    ----------
    machine_type : str
        Machine category (e.g. 'fan', 'bearing').
    checkpoints_dir : Optional[Path]
        Directory holding checkpoint files. Defaults to CHECKPOINTS_AUDIO_DIR.

    Returns
    -------
    Dict[str, Any]
        Dictionary holding model, scaler_mean, scaler_std, thresholds, and device.
    """
    key = machine_type.strip().lower()
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_AUDIO_DIR
    ckpt_path = ckpt_dir / f"{key}_ae.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for machine type '{machine_type}'. "
            f"Expected checkpoint at: {ckpt_path.resolve()}. "
            f"Please ensure the model has been trained first."
        )

    device_str = get_device()
    device = torch.device(device_str)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    input_dim = ckpt.get("input_dim", 640)
    hidden_dim = ckpt.get("hidden_dim", 128)
    bottleneck_dim = ckpt.get("bottleneck_dim", 8)

    model = DenseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_mean = ckpt.get("scaler_mean")
    scaler_std = ckpt.get("scaler_std")
    if scaler_mean is None or scaler_std is None:
        raise ValueError(f"Checkpoint at {ckpt_path} is missing scaler_mean or scaler_std statistics.")

    thresholds = ckpt.get("thresholds", {})
    if not thresholds:
        raise ValueError(f"Checkpoint at {ckpt_path} contains no decision thresholds.")

    cached = {
        "model": model,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "thresholds": thresholds,
        "device": device,
    }
    _MODEL_CACHE[key] = cached
    return cached


def predict_audio(
    audio: Union[str, Path, np.ndarray],
    sampling_rate: Optional[int] = None,
    machine_type: str = "fan",
    threshold_percentile: int = 95,
) -> Dict[str, Any]:
    """Score audio clip for anomalies using trained Dense Autoencoder.

    Parameters
    ----------
    audio : Union[str, Path, np.ndarray]
        File path to a .wav recording or 1-D numpy array of raw audio samples.
    sampling_rate : Optional[int]
        Sampling rate in Hz if `audio` is a numpy array. If None and `audio` is an array,
        assumed to be 16,000 Hz. If `audio` is a file, the file's native sampling rate is detected.
    machine_type : str
        Machine category (e.g. 'fan', 'bearing').
    threshold_percentile : int
        Training normal score percentile used as the decision threshold (e.g. 90, 95, 99).
        Defaults to 95.

    Returns
    -------
    Dict[str, Any]
        {
            "machine_type": str,
            "anomaly_score": float,
            "threshold": float,
            "is_anomaly": bool,
            "status": "NORMAL" or "ANOMALOUS",
            "frame_scores": list of float,
            "sampling_rate_used": 16000,
        }

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist or if no model checkpoint exists for `machine_type`.
    ValueError
        If the audio is empty, contains NaNs/Infs, or is silent.
    """
    target_sr = 16_000
    target_samples = int(target_sr * AUDIO_CFG.feature.clip_duration_s)  # 160,000 samples (10 s)

    # 1. Load Audio
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        if not p.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {p}")
        try:
            y, file_sr = librosa.load(str(p), sr=None, mono=True, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load audio file {p}: {e}") from e
        src_sr = file_sr if sampling_rate is None else sampling_rate
    elif isinstance(audio, np.ndarray):
        y = np.asarray(audio, dtype=np.float32)
        src_sr = target_sr if sampling_rate is None else int(sampling_rate)
    else:
        raise TypeError(f"Expected audio as str, Path, or np.ndarray; got {type(audio).__name__}")

    # Convert multi-channel to mono
    if y.ndim == 2:
        y = np.mean(y, axis=1) if y.shape[0] > y.shape[1] else np.mean(y, axis=0)
    elif y.ndim > 2:
        raise ValueError(f"Unsupported audio array dimension: {y.ndim} (must be 1-D or 2-D)")

    # 2. Validation: Empty, NaN/Inf, Silence
    if y.size == 0 or len(y) == 0:
        raise ValueError("Audio input is empty (0 samples).")

    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("Audio contains NaN or Inf values.")

    peak_amplitude = float(np.max(np.abs(y)))
    std_amplitude = float(np.std(y))
    if peak_amplitude < 1e-5 or std_amplitude < 1e-6:
        raise ValueError(
            f"Audio input is silent (peak amplitude: {peak_amplitude:.2e}, std: {std_amplitude:.2e})."
        )

    # 3. Resample to 16,000 Hz
    if src_sr != target_sr:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=target_sr)

    # 4. Handle Duration: pad if shorter than 10s, window/average if longer
    windows: List[np.ndarray] = []
    if len(y) <= target_samples:
        if len(y) < target_samples:
            y_padded = np.pad(y, (0, target_samples - len(y)), mode="constant", constant_values=0.0)
        else:
            y_padded = y
        windows.append(y_padded)
    else:
        # For longer audio, extract contiguous 10 s windows, zero-padding the final window
        for start_idx in range(0, len(y), target_samples):
            chunk = y[start_idx : start_idx + target_samples]
            if len(chunk) < target_samples:
                chunk = np.pad(chunk, (0, target_samples - len(chunk)), mode="constant", constant_values=0.0)
            windows.append(chunk)

    # 5. Retrieve Cached Model, Scaler, and Threshold
    cached = get_cached_model(machine_type)
    model: DenseAutoencoder = cached["model"]
    scaler_mean: np.ndarray = cached["scaler_mean"]
    scaler_std: np.ndarray = cached["scaler_std"]
    thresholds: Dict[str, float] = cached["thresholds"]
    device: torch.device = cached["device"]

    th_key = f"p{int(threshold_percentile)}"
    if th_key in thresholds:
        threshold = float(thresholds[th_key])
    else:
        threshold = float(thresholds.get("p95", 0.5))

    # 6. Feature Extraction & Reconstruction Scoring
    cfg = AUDIO_CFG.feature
    all_frame_scores: List[float] = []

    for w_audio in windows:
        mel = log_mel(
            w_audio,
            sr=target_sr,
            n_mels=cfg.n_mels,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            power=cfg.power,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        # Normalize: shape (128, n_frames) -> (1, 128, n_frames)
        mel_norm = (mel[np.newaxis, :, :] - scaler_mean) / scaler_std
        mel_norm = mel_norm[0]

        stacked = stack_frames(mel_norm, context_frames=cfg.context_frames, frame_hop=cfg.frame_hop)

        frames_t = torch.from_numpy(stacked).to(device)
        with torch.no_grad():
            recon = model(frames_t)
            frame_errors = torch.mean((frames_t - recon) ** 2, dim=-1).cpu().numpy()

        all_frame_scores.extend([float(x) for x in frame_errors])

    anomaly_score = float(np.mean(all_frame_scores))
    is_anomaly = bool(anomaly_score > threshold)
    status = "ANOMALOUS" if is_anomaly else "NORMAL"

    return {
        "machine_type": machine_type.strip().lower(),
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
        "status": status,
        "frame_scores": all_frame_scores,
        "sampling_rate_used": target_sr,
    }


# Backwards compatibility alias
analyse_audio_clip = predict_audio


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Predict acoustic anomaly from sound clip")
    parser.add_argument("clip_path", type=str, help="Path to input audio .wav clip")
    parser.add_argument("--machine", type=str, default="fan", help="Machine type (e.g. fan, bearing)")
    parser.add_argument(
        "--threshold-percentile",
        type=int,
        default=95,
        help="Decision threshold percentile (90, 95, 99)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    args = parser.parse_args(argv)

    try:
        res = predict_audio(
            audio=args.clip_path,
            machine_type=args.machine,
            threshold_percentile=args.threshold_percentile,
        )
    except Exception as exc:
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 1

    if args.json:
        # Frame scores can be long; output a truncated preview in CLI json
        preview_res = dict(res)
        preview_res["frame_scores"] = f"[{len(res['frame_scores'])} frames computed]"
        print(json.dumps(preview_res, indent=2))
    else:
        status_color = "\033[91m" if res["is_anomaly"] else "\033[92m"
        reset_color = "\033[0m"

        print("\n" + "=" * 60)
        print("ACOUSTIC ANOMALY INFERENCE REPORT")
        print("=" * 60)
        print(f"Clip File          : {args.clip_path}")
        print(f"Machine Type       : {res['machine_type'].upper()}")
        print(f"Sampling Rate Used : {res['sampling_rate_used']} Hz")
        print(f"Reconstruction MSE : {res['anomaly_score']:.6f}")
        print(f"Decision Threshold : {res['threshold']:.6f} (P{args.threshold_percentile})")
        print(f"Classification     : {status_color}{res['status']}{reset_color} (is_anomaly={res['is_anomaly']})")
        print(f"Evaluated Frames   : {len(res['frame_scores'])}")
        print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
