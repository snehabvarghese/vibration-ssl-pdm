"""
Core inference module for Motor Current and Temperature analysis.

Provides a decoupled, cached inference interface designed for easy integration
by teammate application and API layers (FastAPI / Streamlit / Celery).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np

from models.motor_temperature_model import MotorPdMModel
from preprocessing.motor_features import extract_multimodal_features

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints" / "motor_temp"
METADATA_PATH = CHECKPOINTS_DIR / "metadata.json"

MODEL_VERSION = "1.0.0"


@lru_cache(maxsize=3)
def get_motor_model(model_mode: str = "combined") -> Optional[MotorPdMModel]:
    """Load and cache the trained MotorPdMModel instance.

    Parameters
    ----------
    model_mode : str, default "combined"
        One of "combined", "current_only", "temperature_only".

    Returns
    -------
    MotorPdMModel or None
        Cached model instance if checkpoint exists, else None.
    """
    ckpt_path = CHECKPOINTS_DIR / f"{model_mode}_model.joblib"
    if not ckpt_path.exists():
        return None
    return joblib.load(ckpt_path)


def analyse_motor_signal(
    current_phase1: np.ndarray,
    current_phase2: Optional[np.ndarray] = None,
    temperature: Optional[Union[float, np.ndarray]] = None,
    sampling_rate: int = 64_000,
    model_mode: str = "combined",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute end-to-end inference on motor current and temperature data.

    Parameters
    ----------
    current_phase1 : np.ndarray
        Phase 1 motor current readings (1-D array).
    current_phase2 : np.ndarray, optional
        Phase 2 motor current readings (1-D array).
    temperature : float or np.ndarray, optional
        Bearing housing/oil temperature in Celsius. Defaults to 45.0 (PU regulated bench nominal)
        if not supplied.
    sampling_rate : int, default 64000
        Sampling frequency in Hz.
    model_mode : str, default "combined"
        Model selection: "combined", "current_only", or "temperature_only".
    timestamp : str, optional
        Acquisition ISO timestamp from original record metadata (if available).

    Returns
    -------
    dict
        Structured output compliant with team integration contract:
        - model_name (str)
        - model_version (str)
        - status (str: "HEALTHY" or "ANOMALOUS")
        - predicted_condition (str: e.g. "Normal", "Outer_Race", "Inner_Race")
        - anomaly_score (float in [0.0, 1.0])
        - confidence (float in [0.0, 1.0])
        - class_probabilities (dict of str -> float)
        - timestamp (str or None)
        - features (dict of extracted MCSA & thermal metrics)
        - input_diagnostics (dict)
    """
    p1 = np.asarray(current_phase1, dtype=np.float64).ravel()
    if len(p1) == 0:
        raise ValueError("Cannot analyse empty motor current signal.")

    p2 = np.asarray(current_phase2, dtype=np.float64).ravel() if current_phase2 is not None else None

    # Extract features matching the 0.5s training window length
    win_len = int(round(0.5 * sampling_rate))
    if len(p1) > win_len:
        step = max(1, win_len // 2)
        all_vecs = []
        last_dict = {}
        for start in range(0, len(p1) - win_len + 1, step):
            w1 = p1[start : start + win_len]
            w2 = p2[start : start + win_len] if p2 is not None else None
            v, feature_names, last_dict = extract_multimodal_features(
                current_phase1=w1,
                current_phase2=w2,
                temperature=temperature,
                sampling_rate=sampling_rate,
            )
            all_vecs.append(v)
        vec = np.mean(all_vecs, axis=0)
        feat_dict = {name: float(vec[i]) for i, name in enumerate(feature_names)}
    else:
        vec, feature_names, feat_dict = extract_multimodal_features(
            current_phase1=p1,
            current_phase2=p2,
            temperature=temperature,
            sampling_rate=sampling_rate,
        )
    X = vec.reshape(1, -1)

    model = get_motor_model(model_mode=model_mode)
    if model is None:
        raise RuntimeError(
            f"Trained model '{model_mode}' not found in {CHECKPOINTS_DIR}. "
            "Please run `python training/train_motor_models.py` first."
        )

    # Classification inference
    pred_class = str(model.predict(X, feature_names)[0])
    probas = model.predict_proba(X, feature_names)[0]
    classes = [str(c) for c in model.classes_]
    class_probs = {c: float(p) for c, p in zip(classes, probas)}
    conf = float(np.max(probas))

    # Anomaly scoring
    anomaly_score = float(model.score_anomaly(X, feature_names)[0])
    status = "ANOMALOUS" if anomaly_score >= 0.5 or pred_class != "Normal" else "HEALTHY"

    return {
        "model_name": f"motor_{model_mode}_pdm",
        "model_version": MODEL_VERSION,
        "status": status,
        "predicted_condition": pred_class,
        "anomaly_score": round(anomaly_score, 4),
        "confidence": round(conf, 4),
        "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
        "timestamp": timestamp,  # None if not available; never fabricated
        "features": {k: round(v, 4) for k, v in feat_dict.items()},
        "input_diagnostics": {
            "samples": len(p1),
            "duration_seconds": round(len(p1) / sampling_rate, 4),
            "sampling_rate_hz": sampling_rate,
            "two_phase_available": p2 is not None and len(p2) > 0,
            "temperature_input_available": temperature is not None,
        },
    }
