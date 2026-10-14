"""
FastAPI backend service for Vibration SSL Predictive Maintenance.

Endpoints:
  GET  /health   - Service health status and compute device.
  POST /analyse  - Accepts vibration files (.mat, .npy, .csv) with sampling_rate
                   and optional channel; returns complete analysis metrics as JSON.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Root path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import CFG, DATA_RAW_DIR, get_device
from core.analysis import analyse_signal, get_map
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import _find_channel_key, _resample_to, load_signal

app = FastAPI(
    title="Vibration SSL PdM API",
    description="FastAPI service for vibration analysis, anomaly detection, and fault classification.",
    version="1.0.0",
)

# Enable CORS for frontend clients (e.g., Next.js / React running on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy arrays and scalars to native Python types.

    Also replaces NaN / Infinity with None so the result is valid JSON.
    """
    import math

    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _extract_signal_from_mat(content: bytes, channel: Optional[str]) -> np.ndarray:
    """Extract 1-D vibration signal from a .mat file buffer."""
    try:
        mat = sio.loadmat(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable file: failed to parse MATLAB (.mat) format: {exc}",
        ) from exc

    if channel:
        from preprocessing.loader import _CHANNEL_SUFFIX
        if channel in _CHANNEL_SUFFIX:
            key = _find_channel_key(mat, channel)
        elif channel in mat:
            key = channel
        else:
            candidates = [
                k for k in mat
                if isinstance(k, str) and (k.endswith(f"_{channel}_time") or k.endswith(f"_{channel}"))
            ]
            key = candidates[0] if candidates else None

        if key is None:
            available = sorted(k for k in mat if isinstance(k, str) and not k.startswith("__"))
            raise HTTPException(
                status_code=400,
                detail=f"Channel '{channel}' not found in .mat file. Available keys: {available}",
            )
    else:
        # Prioritize Drive End accelerometer keys matching CWRU convention
        keys = [k for k in mat if isinstance(k, str) and k.endswith("_time")] or [
            k for k, v in mat.items() if isinstance(k, str) and not k.startswith("__") and getattr(v, "size", 0) > 1000
        ]
        if not keys:
            raise HTTPException(
                status_code=400,
                detail="Unreadable file: no 1-D vibration signal variable found in .mat file.",
            )
        de_keys = [k for k in keys if k.endswith("_DE_time")]
        key = de_keys[0] if de_keys else keys[0]

    try:
        sig = np.asarray(mat[key], dtype=float).ravel()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable file: variable '{key}' could not be converted to 1-D float array: {exc}",
        ) from exc
    return sig


def _extract_signal_from_npy(content: bytes) -> np.ndarray:
    """Extract 1-D vibration signal from a .npy buffer."""
    try:
        arr = np.load(io.BytesIO(content))
        return np.asarray(arr, dtype=float).ravel()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable file: failed to load numpy (.npy) format: {exc}",
        ) from exc


def _extract_signal_from_csv(content: bytes, channel: Optional[str]) -> np.ndarray:
    """Extract 1-D vibration signal from a CSV buffer."""
    try:
        # Try reading with pandas
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable file: failed to parse CSV format: {exc}",
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Empty file: CSV contains no rows.")

    if channel:
        if channel in df.columns:
            col = df[channel]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Channel '{channel}' not found in CSV. Available columns: {list(df.columns)}",
            )
    else:
        if df.shape[1] == 1:
            col = df.iloc[:, 0]
        else:
            de_cols = [c for c in df.columns if "DE" in str(c).upper()]
            if de_cols:
                col = df[de_cols[0]]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = df[numeric_cols[0]]
                else:
                    col = df.iloc[:, 0]

    try:
        sig = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float).ravel()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unreadable file: could not parse numeric column from CSV: {exc}",
        ) from exc
    return sig


@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint returning system status and compute device."""
    return {
        "status": "healthy",
        "device": get_device(),
        "model_loaded": True,
    }


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(..., description="Vibration file (.mat, .npy, or .csv)"),
    sampling_rate: int = Form(12000, description="Vibration signal sampling rate in Hz"),
    channel: Optional[str] = Form(None, description="Optional channel/variable name (e.g. 'DE')"),
    clf_name: Optional[str] = Form("svm_rbf", description="Classifier to use (e.g. 'svm_rbf', 'mlp')"),
) -> Dict[str, Any]:
    """Analyse an uploaded vibration file and return PdM metrics, fault prediction, and anomaly scores."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in uploaded file.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    ext = Path(file.filename).suffix.lower()

    if ext == ".mat":
        sig = _extract_signal_from_mat(content, channel)
    elif ext == ".npy":
        sig = _extract_signal_from_npy(content)
    elif ext == ".csv":
        sig = _extract_signal_from_csv(content, channel)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: .mat, .npy, .csv",
        )

    if sig.size == 0:
        raise HTTPException(status_code=400, detail="Unreadable file: extracted signal contains zero samples.")

    # Validate finite values (reject NaNs and Infs)
    if not np.isfinite(sig).all():
        raise HTTPException(status_code=400, detail="Signal contains invalid values (NaN or Inf).")

    # Resample to model's expected sampling rate (12 kHz) if required
    effective_rate = sampling_rate
    if sampling_rate != CFG.dataset.sampling_rate:
        try:
            sig = _resample_to(sig, src_rate=int(sampling_rate), dst_rate=CFG.dataset.sampling_rate)
            effective_rate = CFG.dataset.sampling_rate
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to resample signal from {sampling_rate} Hz to {CFG.dataset.sampling_rate} Hz: {exc}",
            ) from exc

    # Validate signal duration
    min_window_samples = CFG.preprocess.window_size(effective_rate)
    if len(sig) < min_window_samples:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Signal is too short to produce an analysis window. "
                f"Requires at least {min_window_samples} samples ({CFG.preprocess.window_seconds} s), "
                f"got {len(sig)} samples."
            ),
        )

    # Run core analysis pipeline
    result = analyse_signal(sig, effective_rate, clf_name=clf_name or "svm_rbf")
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Signal is too short to produce an analysis window.",
        )

    # Waveform downsampled to <= 2000 points
    step = max(1, len(sig) // 2000)
    result["waveform"] = {
        "samples": sig[::step].tolist(),
        "sampling_rate": int(effective_rate),
    }

    # Transform embeddings to 2-D PCA space
    _, _, _, pca = get_map()
    result["map_xy"] = pca.transform(np.asarray(result["embeddings"])).tolist()

    return _to_serializable(result)


@app.get("/map")
def get_embedding_map(max_points: int = 1500) -> Dict[str, Any]:
    """Return background embedding map (2-D PCA coordinates and class labels)."""
    xy, labels, classes, _ = get_map(max_points)
    points = [
        {
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
            "class_label": str(classes[labels[i]]),
        }
        for i in range(len(xy))
    ]
    return {
        "points": points,
        "classes": [str(c) for c in classes],
        "total_points": len(points),
    }


@app.get("/recordings")
def list_recordings() -> List[Dict[str, Any]]:
    """List raw CWRU recordings available on disk."""
    records = [r for r in select_records() if (DATA_RAW_DIR / f"{r.file_id}.mat").exists()]
    return [
        {
            "id": str(r.file_id),
            "label": r.label(),
            "fault_type": r.fault_type,
            "load_hp": r.load_hp,
            "rpm": r.rpm,
            "sampling_rate": r.sampling_rate,
        }
        for r in records
    ]


@app.post("/analyse/recording/{file_id}")
def analyse_recording(
    file_id: str,
    clf_name: str = "svm_rbf",
) -> Dict[str, Any]:
    """Analyse a CWRU dataset recording by file ID without uploading."""
    records = {str(r.file_id): r for r in select_records() if (DATA_RAW_DIR / f"{r.file_id}.mat").exists()}
    if file_id not in records:
        raise HTTPException(status_code=404, detail=f"Recording '{file_id}' not found in data/raw.")

    rec = records[file_id]
    try:
        loaded = load_signal(rec)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load recording {file_id}: {exc}",
        ) from exc

    sig = loaded.signal
    rate = loaded.sampling_rate

    result = analyse_signal(sig, rate, clf_name=clf_name or "svm_rbf")
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Signal is too short to produce an analysis window.",
        )

    # Waveform downsampled to <= 2000 points
    step = max(1, len(sig) // 2000)
    result["waveform"] = {
        "samples": sig[::step].tolist(),
        "sampling_rate": int(rate),
    }

    # Transform embeddings to 2-D PCA space
    _, _, _, pca = get_map()
    result["map_xy"] = pca.transform(np.asarray(result["embeddings"])).tolist()

    # Metadata & ground truth
    result["recording_id"] = file_id
    result["true_label"] = loaded.label
    result["load_hp"] = loaded.load_hp
    result["rpm"] = loaded.rpm

    return _to_serializable(result)
