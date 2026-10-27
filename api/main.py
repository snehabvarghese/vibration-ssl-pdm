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
from core.motor_temperature import analyse_motor_signal
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import _find_channel_key, _resample_to, load_signal
from preprocessing.paderborn_loader import load_paderborn_mat, parse_paderborn_filename

app = FastAPI(
    title="Vibration SSL PdM API",
    description="FastAPI service for vibration analysis, anomaly detection, and fault classification.",
    version="1.0.0",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include unified prediction and model inspection endpoints
from api.routes_predict import router as predict_router
app.include_router(predict_router)



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


# --------------------------------------------------------------------------
# Motor Current & Temperature PdM Endpoints
# --------------------------------------------------------------------------
@app.get("/recordings/motor")
def list_motor_recordings() -> List[Dict[str, Any]]:
    """List available Paderborn motor recordings on disk."""
    paderborn_dir = ROOT_DIR / "data" / "paderborn"
    if not paderborn_dir.exists():
        return []
    mat_files = sorted(list(paderborn_dir.rglob("*.mat")))
    records = []
    for p in mat_files:
        meta = parse_paderborn_filename(p.name)
        records.append({
            "filename": p.name,
            "bearing_id": meta["bearing_id"],
            "condition_label": meta["condition_label"],
            "fault_type": meta["fault_type"],
            "operating_condition": meta["operating_condition"],
            "speed_rpm": meta["speed_rpm"],
            "load_torque_nm": meta["load_torque_nm"],
            "file_size_mb": round(p.stat().st_size / 1e6, 2),
        })
    return records


@app.post("/analyse/motor")
async def analyse_motor_upload(
    file: UploadFile = File(...),
    temperature: Optional[float] = Form(None),
    sampling_rate: int = Form(64_000),
    model_mode: str = Form("combined"),
) -> Dict[str, Any]:
    """Analyse uploaded motor current and temperature file (.mat, .npy, .csv)."""
    content = await file.read()
    filename = file.filename or "motor_signal.mat"

    p1: Optional[np.ndarray] = None
    p2: Optional[np.ndarray] = None
    temp_val = temperature
    fs = sampling_rate

    lower = filename.lower()
    if lower.endswith(".mat"):
        try:
            rec = load_paderborn_mat(content, filename_hint=filename)
            p1 = rec.phase_current_1
            p2 = rec.phase_current_2
            fs = rec.sampling_rate
            if temp_val is None:
                temp_val = rec.temperature_celsius
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse MATLAB motor file: {exc}") from exc
    elif lower.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(content))
            if arr.ndim == 2:
                if arr.shape[0] == 2:
                    p1, p2 = arr[0], arr[1]
                elif arr.shape[1] >= 2:
                    p1, p2 = arr[:, 0], arr[:, 1]
                else:
                    p1 = arr.ravel()
            else:
                p1 = arr.ravel()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse numpy array: {exc}") from exc
    elif lower.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content))
            c_cols = [c for c in df.columns if "current" in c.lower() or "i_" in c.lower()]
            if c_cols:
                p1 = df[c_cols[0]].values.astype(np.float64)
                if len(c_cols) > 1:
                    p2 = df[c_cols[1]].values.astype(np.float64)
            else:
                p1 = df.iloc[:, 0].values.astype(np.float64)
                if df.shape[1] > 1:
                    p2 = df.iloc[:, 1].values.astype(np.float64)
            t_cols = [c for c in df.columns if "temp" in c.lower()]
            if t_cols and temp_val is None:
                temp_val = float(df[t_cols[0]].mean())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Upload .mat, .npy, or .csv")

    if p1 is None or len(p1) == 0:
        raise HTTPException(status_code=400, detail="Could not extract motor current signal from uploaded file.")

    try:
        result = analyse_motor_signal(
            current_phase1=p1,
            current_phase2=p2,
            temperature=temp_val,
            sampling_rate=fs,
            model_mode=model_mode,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Motor analysis error: {exc}") from exc

    step = max(1, len(p1) // 1500)
    result["waveform_p1"] = p1[::step].tolist()
    if p2 is not None:
        result["waveform_p2"] = p2[::step].tolist()

    n_fft = min(len(p1), fs)
    fft_vals = np.abs(np.fft.rfft(p1[:n_fft])) / (n_fft / 2.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spec_mask = freqs <= 1000.0
    s_step = max(1, int(np.sum(spec_mask) // 400))
    result["spectrum"] = {
        "freqs": freqs[spec_mask][::s_step].tolist(),
        "magnitudes": fft_vals[spec_mask][::s_step].tolist(),
    }
    result["metadata"] = {
        "filename": filename,
        "temperature_celsius": temp_val if temp_val is not None else 45.0,
        "sampling_rate": fs,
    }
    return _to_serializable(result)


@app.post("/analyse/motor/recording")
def analyse_motor_preset(
    filename: str = Form(...),
    temperature: Optional[float] = Form(None),
    model_mode: str = Form("combined"),
) -> Dict[str, Any]:
    """Analyse a Paderborn benchmark recording stored locally on disk."""
    paderborn_dir = ROOT_DIR / "data" / "paderborn"
    matches = list(paderborn_dir.rglob(filename))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Recording '{filename}' not found in data/paderborn.")

    rec = load_paderborn_mat(matches[0])
    temp = temperature if temperature is not None else rec.temperature_celsius

    result = analyse_motor_signal(
        current_phase1=rec.phase_current_1,
        current_phase2=rec.phase_current_2,
        temperature=temp,
        sampling_rate=rec.sampling_rate,
        model_mode=model_mode,
    )

    step = max(1, len(rec.phase_current_1) // 1500)
    result["waveform_p1"] = rec.phase_current_1[::step].tolist()
    if rec.phase_current_2 is not None:
        result["waveform_p2"] = rec.phase_current_2[::step].tolist()

    n_fft = min(len(rec.phase_current_1), rec.sampling_rate)
    fft_vals = np.abs(np.fft.rfft(rec.phase_current_1[:n_fft])) / (n_fft / 2.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / rec.sampling_rate)
    spec_mask = freqs <= 1000.0
    s_step = max(1, int(np.sum(spec_mask) // 400))
    result["spectrum"] = {
        "freqs": freqs[spec_mask][::s_step].tolist(),
        "magnitudes": fft_vals[spec_mask][::s_step].tolist(),
    }
    result["metadata"] = {
        "filename": rec.filename,
        "bearing_id": rec.bearing_id,
        "true_label": rec.condition_label,
        "operating_condition": rec.operating_condition,
        "speed_rpm": rec.speed_rpm,
        "load_torque_nm": rec.load_torque_nm,
        "temperature_celsius": temp,
    }
    return _to_serializable(result)
