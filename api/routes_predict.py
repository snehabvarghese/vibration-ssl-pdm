"""
Unified prediction and model inspection endpoints for Vibration, Audio, and Motor PdM models.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from config import CFG, DATA_RAW_DIR
from core.fusion.fusion import FusionConfig, FusionResult, fuse
from core.models_api import get_registry, ModelResult
from preprocessing.cwru_manifest import select_records
from preprocessing.loader import _resample_to, load_signal
from preprocessing.paderborn_loader import load_paderborn_mat
from tests.fixtures.paderborn_synthetic import generate_synthetic_motor_current

router = APIRouter(tags=["Unified PdM Predictions"])

# Maximum allowable upload size (50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class FusionResultResponse(BaseModel):
    """Schema for multimodal fusion verdict."""

    fused_score: float
    fused_status: Literal["NORMAL", "WARNING", "ALARM", "UNAVAILABLE"]
    method: Literal["max", "weighted", "vote"]
    per_model_ratios: Dict[str, float]
    missing_models: List[str]
    reasons: List[str]
    confidence: float
    is_replay: bool = False
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)


class FusionRequest(BaseModel):
    """Request schema for /fusion endpoint."""

    results: Optional[List[Dict[str, Any]]] = None
    sample_ids: Optional[Dict[str, str]] = None
    method: Literal["max", "weighted", "vote"] = "max"
    config: Optional[Dict[str, Any]] = None


class ScenarioMetadataResponse(BaseModel):
    """Schema for replay scenario metadata."""

    id: str
    name: str
    description: str
    total_steps: int = 60
    duration_seconds: int = 60
    is_replay: bool = True


class ModelInfoResponse(BaseModel):
    """Schema for individual model metadata."""

    model_name: str
    modality: str
    is_available: bool
    threshold: float
    threshold_metric: Optional[str] = None
    training_dataset: Optional[str] = None
    checkpoint_path: str
    monitor_path: Optional[str] = None
    machine_type: Optional[str] = None
    sub_type: Optional[str] = None


class ModelResultResponse(BaseModel):
    """Schema for unified inference result across all modalities."""

    model_name: str
    modality: Literal["vibration", "audio", "motor"]
    status: Literal["NORMAL", "ANOMALOUS", "UNAVAILABLE"]
    anomaly_score: float
    threshold: float
    score_ratio: float
    prediction_label: Optional[str] = None
    confidence: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    input_info: Dict[str, Any] = Field(default_factory=dict)
    is_replay: bool = False
    timestamp: str
    error: Optional[str] = None


class SampleRecordResponse(BaseModel):
    """Schema for demo / replay sample recording description."""

    sample_id: str
    modality: Literal["vibration", "audio", "motor"]
    label: Optional[str] = None
    description: Optional[str] = None
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    sampling_rate: Optional[int] = None
    is_replay: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/models", response_model=List[ModelInfoResponse])
def get_models() -> List[Dict[str, Any]]:
    """List all registered models across Vibration, Audio, and Motor with availability and thresholds."""
    registry = get_registry()
    return registry.list_models()


@router.post("/predict/vibration", response_model=ModelResultResponse)
async def predict_vibration(
    file: UploadFile = File(..., description="Vibration signal file (.mat, .npy, .csv)"),
    sampling_rate: int = Form(12000, description="Sampling frequency in Hz"),
    channel: Optional[str] = Form(None, description="Optional channel/variable name (e.g. 'DE')"),
    clf_name: Optional[str] = Form("svm_rbf", description="Classifier variant (e.g. 'svm_rbf')"),
) -> Dict[str, Any]:
    """Execute vibration SSL anomaly detection and fault classification on an uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in uploaded file.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )

    ext = Path(file.filename).suffix.lower()
    from api.main import _extract_signal_from_csv, _extract_signal_from_mat, _extract_signal_from_npy

    if ext == ".mat":
        sig = _extract_signal_from_mat(content, channel)
    elif ext == ".npy":
        sig = _extract_signal_from_npy(content)
    elif ext == ".csv":
        sig = _extract_signal_from_csv(content, channel)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Upload .mat, .npy, or .csv",
        )

    if sig.size == 0:
        raise HTTPException(status_code=400, detail="Extracted signal contains zero samples.")
    if not np.isfinite(sig).all():
        raise HTTPException(status_code=400, detail="Signal contains invalid values (NaN or Inf).")

    effective_rate = sampling_rate
    if sampling_rate != CFG.dataset.sampling_rate:
        try:
            sig = _resample_to(sig, src_rate=int(sampling_rate), dst_rate=CFG.dataset.sampling_rate)
            effective_rate = CFG.dataset.sampling_rate
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Resampling failed: {exc}") from exc

    min_samples = CFG.preprocess.window_size(effective_rate)
    if len(sig) < min_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Signal too short. Requires >= {min_samples} samples ({CFG.preprocess.window_seconds}s), got {len(sig)}.",
        )

    adapter = get_registry().get_adapter("vibration")
    if adapter is None:
        raise HTTPException(status_code=400, detail="Vibration adapter is not registered.")

    res = adapter.predict(
        sig,
        sampling_rate=effective_rate,
        clf_name=clf_name or "svm_rbf",
        source_name=file.filename,
        is_replay=False,
    )
    return res.to_dict()


@router.post("/predict/audio", response_model=ModelResultResponse)
async def predict_audio(
    file: UploadFile = File(..., description="Audio recording (.wav)"),
    machine_type: str = Form("fan", description="Machine category: 'fan' or 'bearing'"),
    threshold_percentile: int = Form(95, description="Decision threshold percentile (default: 95)"),
) -> Dict[str, Any]:
    """Execute acoustic anomaly detection using trained Dense Autoencoder on uploaded .wav clip."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing in uploaded file.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext != ".wav":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Only .wav files are supported.",
        )

    mt = machine_type.strip().lower()
    if mt not in ("fan", "bearing"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown machine_type '{machine_type}'. Must be 'fan' or 'bearing'.",
        )

    try:
        audio_arr, sr = sf.read(io.BytesIO(content))
        if audio_arr.ndim > 1:
            audio_arr = np.mean(audio_arr, axis=1)  # stereo to mono
        audio_arr = np.asarray(audio_arr, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable audio file: {exc}") from exc

    if audio_arr.size == 0:
        raise HTTPException(status_code=400, detail="Audio file contains no samples.")
    if not np.isfinite(audio_arr).all():
        raise HTTPException(status_code=400, detail="Audio signal contains invalid values (NaN or Inf).")

    adapter = get_registry().get_adapter(f"audio_{mt}") or get_registry().get_adapter("audio")
    if adapter is None:
        raise HTTPException(status_code=400, detail=f"Audio adapter not registered for machine '{mt}'.")

    res = adapter.predict(
        audio_arr,
        sampling_rate=sr,
        machine_type=mt,
        threshold_percentile=threshold_percentile,
        source_name=file.filename,
        is_replay=False,
    )
    return res.to_dict()


@router.post("/predict/motor", response_model=ModelResultResponse)
async def predict_motor(
    file: Optional[UploadFile] = File(None, description="Motor recording (.mat, .npy, .csv)"),
    temperature: Optional[float] = Form(None, description="Bearing housing temperature in Celsius (e.g. 45.0)"),
    sampling_rate: int = Form(64_000, description="Sampling frequency in Hz (default: 64,000)"),
    model_mode: str = Form("combined", description="Model mode: 'combined', 'current_only', or 'temperature_only'"),
    current_phase1: Optional[str] = Form(None, description="Optional comma-separated phase 1 current numbers"),
    current_phase2: Optional[str] = Form(None, description="Optional comma-separated phase 2 current numbers"),
) -> Dict[str, Any]:
    """Execute motor current and temperature predictive maintenance diagnosis."""
    mm = model_mode.strip().lower()
    if mm not in ("combined", "current_only", "temperature_only"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_mode '{model_mode}'. Must be 'combined', 'current_only', or 'temperature_only'.",
        )

    p1: Optional[np.ndarray] = None
    p2: Optional[np.ndarray] = None
    temp_val: Optional[float] = temperature
    fs: int = sampling_rate
    source_name = "motor_input"

    if file is not None and file.filename:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded motor file is empty.")
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
            )
        source_name = file.filename
        ext = Path(file.filename).suffix.lower()

        if ext == ".mat":
            try:
                rec = load_paderborn_mat(content, filename_hint=file.filename)
                p1 = rec.phase_current_1
                p2 = rec.phase_current_2
                fs = rec.sampling_rate
                if temp_val is None:
                    temp_val = rec.temperature_celsius
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to parse MATLAB motor file: {exc}") from exc
        elif ext == ".npy":
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
        elif ext == ".csv":
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
            raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Upload .mat, .npy, or .csv")

    elif current_phase1:
        try:
            cleaned = current_phase1.strip("[] \r\n\t")
            p1 = np.array([float(x.strip()) for x in cleaned.split(",") if x.strip()], dtype=np.float64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse current_phase1 array: {exc}") from exc
        if current_phase2:
            try:
                cleaned2 = current_phase2.strip("[] \r\n\t")
                p2 = np.array([float(x.strip()) for x in cleaned2.split(",") if x.strip()], dtype=np.float64)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to parse current_phase2 array: {exc}") from exc
        source_name = "form_data_current"

    if p1 is None or len(p1) == 0:
        if mm == "temperature_only" and temp_val is not None:
            source_name = "temperature_scalar"
        else:
            raise HTTPException(
                status_code=400,
                detail="Missing motor input: provide a motor recording file, 'current_phase1', or 'temperature'.",
            )

    adapter = get_registry().get_adapter(f"motor_{mm}") or get_registry().get_adapter("motor")
    if adapter is None:
        raise HTTPException(status_code=400, detail=f"Motor adapter for mode '{mm}' not found.")

    res = adapter.predict(
        input_data=p1 if p1 is not None else temp_val,
        current_phase2=p2,
        temperature=temp_val,
        sampling_rate=fs,
        source_name=source_name,
        is_replay=False,
    )
    return res.to_dict()


@router.get("/samples/{modality}", response_model=List[SampleRecordResponse])
def list_samples(modality: str) -> List[Dict[str, Any]]:
    """List recorded benchmark demo files for the specified modality (all labelled is_replay=true)."""
    mod = modality.strip().lower()

    if mod == "vibration":
        records = [r for r in select_records() if (DATA_RAW_DIR / f"{r.file_id}.mat").exists()]
        return [
            {
                "sample_id": str(r.file_id),
                "modality": "vibration",
                "label": r.label(),
                "description": f"CWRU Fault {r.fault_type} ({r.load_hp} HP, {r.rpm} RPM)",
                "filename": f"{r.file_id}.mat",
                "file_size_bytes": (DATA_RAW_DIR / f"{r.file_id}.mat").stat().st_size,
                "sampling_rate": r.sampling_rate,
                "is_replay": True,
            }
            for r in records
        ]

    elif mod == "audio":
        audio_root = Path(__file__).resolve().parent.parent / "data" / "audio" / "raw"
        samples = []
        for machine in ("fan", "bearing"):
            test_dir = audio_root / machine / "test"
            if test_dir.exists():
                wavs = sorted(list(test_dir.glob("*.wav")))[:20]  # First 20 per machine
                for p in wavs:
                    is_anom = "anomaly" in p.name.lower()
                    samples.append({
                        "sample_id": f"{machine}_{p.stem}",
                        "modality": "audio",
                        "label": "Anomaly" if is_anom else "Normal",
                        "description": f"MIMII DG {machine.capitalize()} Test Clip ({'Anomaly' if is_anom else 'Normal'})",
                        "filename": p.name,
                        "file_size_bytes": p.stat().st_size,
                        "duration_seconds": 10.0,
                        "sampling_rate": 16000,
                        "is_replay": True,
                    })
        return samples

    elif mod == "motor":
        samples = []
        paderborn_dir = Path(__file__).resolve().parent.parent / "data" / "paderborn"
        if paderborn_dir.exists():
            for p in sorted(list(paderborn_dir.rglob("*.mat"))):
                from preprocessing.paderborn_loader import parse_paderborn_filename
                meta = parse_paderborn_filename(p.name)
                samples.append({
                    "sample_id": p.name,
                    "modality": "motor",
                    "label": meta["condition_label"],
                    "description": f"Paderborn {meta['bearing_id']} ({meta['speed_rpm']} RPM, {meta['load_torque_nm']} Nm)",
                    "filename": p.name,
                    "file_size_bytes": p.stat().st_size,
                    "sampling_rate": 64000,
                    "is_replay": True,
                })

        # Reliable built-in demo replay presets
        demo_presets = [
            ("paderborn_demo_normal", "Normal", "Paderborn Stator Current & Temp - Normal Baseline"),
            ("paderborn_demo_outer_race", "Outer_Race", "Paderborn Stator Current & Temp - Outer Race Fault"),
            ("paderborn_demo_inner_race", "Inner_Race", "Paderborn Stator Current & Temp - Inner Race Fault"),
        ]
        for sid, lbl, desc in demo_presets:
            samples.append({
                "sample_id": sid,
                "modality": "motor",
                "label": lbl,
                "description": desc,
                "filename": f"{sid}.mat",
                "file_size_bytes": 1024000,
                "duration_seconds": 2.0,
                "sampling_rate": 64000,
                "is_replay": True,
            })
        return samples

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported modality '{modality}'. Supported modalities: vibration, audio, motor",
        )


@router.post("/predict/{modality}/sample/{sample_id}", response_model=ModelResultResponse)
def predict_sample(modality: str, sample_id: str) -> Dict[str, Any]:
    """Execute prediction on a recorded demo sample file (always marked is_replay=true)."""
    mod = modality.strip().lower()
    sid = sample_id.strip()

    if mod == "vibration":
        records = {str(r.file_id): r for r in select_records() if (DATA_RAW_DIR / f"{r.file_id}.mat").exists()}
        if sid not in records:
            raise HTTPException(status_code=404, detail=f"Vibration sample recording '{sid}' not found.")
        rec = records[sid]
        try:
            loaded = load_signal(rec)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load sample '{sid}': {exc}") from exc

        adapter = get_registry().get_adapter("vibration")
        if adapter is None:
            raise HTTPException(status_code=400, detail="Vibration adapter not found.")
        res = adapter.predict(
            loaded.signal,
            sampling_rate=loaded.sampling_rate,
            clf_name="svm_rbf",
            source_name=f"cwru_sample_{sid}",
            is_replay=True,
        )
        return res.to_dict()

    elif mod == "audio":
        audio_root = Path(__file__).resolve().parent.parent / "data" / "audio" / "raw"
        found_file = None
        machine_type = "fan"

        target_name = sid if sid.endswith(".wav") else f"{sid}.wav"
        for p in audio_root.rglob("*.wav"):
            if p.name == target_name or p.stem == sid or f"fan_{p.stem}" == sid or f"bearing_{p.stem}" == sid:
                found_file = p
                machine_type = "bearing" if "bearing" in str(p).lower() else "fan"
                break

        if found_file is None:
            raise HTTPException(status_code=404, detail=f"Audio sample '{sid}' not found in data/audio/raw.")

        adapter = get_registry().get_adapter(f"audio_{machine_type}") or get_registry().get_adapter("audio")
        if adapter is None:
            raise HTTPException(status_code=400, detail="Audio adapter not found.")

        res = adapter.predict(
            found_file,
            machine_type=machine_type,
            source_name=found_file.name,
            is_replay=True,
        )
        return res.to_dict()

    elif mod == "motor":
        if sid in ("paderborn_demo_normal", "demo_normal", "normal"):
            p1, p2, temp = generate_synthetic_motor_current(condition="Normal", seed=42)
        elif sid in ("paderborn_demo_outer_race", "demo_outer_race", "outer_race"):
            p1, p2, temp = generate_synthetic_motor_current(condition="Outer_Race", seed=42)
        elif sid in ("paderborn_demo_inner_race", "demo_inner_race", "inner_race"):
            p1, p2, temp = generate_synthetic_motor_current(condition="Inner_Race", seed=42)
        else:
            paderborn_dir = Path(__file__).resolve().parent.parent / "data" / "paderborn"
            matches = list(paderborn_dir.rglob(sid if sid.endswith(".mat") else f"{sid}.mat"))
            if not matches:
                raise HTTPException(status_code=404, detail=f"Motor sample '{sid}' not found.")
            rec = load_paderborn_mat(matches[0])
            p1 = rec.phase_current_1
            p2 = rec.phase_current_2
            temp = rec.temperature_celsius

        adapter = get_registry().get_adapter("motor")
        if adapter is None:
            raise HTTPException(status_code=400, detail="Motor adapter not found.")

        res = adapter.predict(
            p1,
            current_phase2=p2,
            temperature=temp,
            sampling_rate=64000,
            source_name=sid,
            is_replay=True,
        )
        return res.to_dict()

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported modality '{modality}'. Supported modalities: vibration, audio, motor",
        )


# ---------------------------------------------------------------------------
# Fusion Endpoints
# ---------------------------------------------------------------------------

@router.get("/fusion/config")
def get_fusion_config() -> Dict[str, Any]:
    """Return default fusion configuration and decision boundary parameters."""
    cfg = FusionConfig()
    return asdict(cfg)


@router.post("/fusion", response_model=FusionResultResponse)
def run_fusion(req: FusionRequest) -> Dict[str, Any]:
    """Execute multimodal late fusion over a list of ModelResults or sample IDs."""
    model_results: List[ModelResult] = []

    # Case 1: Direct list of ModelResult dicts
    if req.results:
        for d in req.results:
            mr = ModelResult(
                model_name=str(d.get("model_name", "model")),
                modality=d.get("modality", "vibration"),
                status=d.get("status", "NORMAL"),
                anomaly_score=float(d.get("anomaly_score", 0.0)),
                threshold=float(d.get("threshold", 1.0)),
                score_ratio=float(d.get("score_ratio", 0.0)),
                prediction_label=d.get("prediction_label"),
                confidence=float(d.get("confidence", 0.0)) if d.get("confidence") is not None else None,
                details=d.get("details", {}),
                input_info=d.get("input_info", {}),
                is_replay=bool(d.get("is_replay", False)),
                timestamp=str(d.get("timestamp", "")),
                error=d.get("error"),
            )
            model_results.append(mr)

    # Case 2: Dictionary of sample IDs per modality
    elif req.sample_ids:
        for mod, sid in req.sample_ids.items():
            res_dict = predict_sample(modality=mod, sample_id=sid)
            mr = ModelResult(
                model_name=str(res_dict.get("model_name", f"{mod}_sample")),
                modality=res_dict.get("modality", mod),  # type: ignore[arg-type]
                status=res_dict.get("status", "NORMAL"),  # type: ignore[arg-type]
                anomaly_score=float(res_dict.get("anomaly_score", 0.0)),
                threshold=float(res_dict.get("threshold", 1.0)),
                score_ratio=float(res_dict.get("score_ratio", 0.0)),
                prediction_label=res_dict.get("prediction_label"),
                confidence=float(res_dict.get("confidence", 0.0)) if res_dict.get("confidence") is not None else None,
                details=res_dict.get("details", {}),
                input_info=res_dict.get("input_info", {}),
                is_replay=bool(res_dict.get("is_replay", True)),
                timestamp=str(res_dict.get("timestamp", "")),
                error=res_dict.get("error"),
            )
            model_results.append(mr)

    else:
        raise HTTPException(
            status_code=400,
            detail="Missing input: please provide either 'results' (list of ModelResult dicts) or 'sample_ids' (mapping of modality -> sample_id).",
        )

    # Configure fusion parameters
    cfg = FusionConfig()
    if req.config:
        for k, v in req.config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    res = fuse(model_results, method=req.method, config=cfg)
    return res.to_dict()


# ---------------------------------------------------------------------------
# Replay Streaming Endpoints
# ---------------------------------------------------------------------------

@router.get("/replay/scenarios", response_model=List[ScenarioMetadataResponse])
def get_replay_scenarios() -> List[Dict[str, Any]]:
    """List available 60-second replay scenarios for streaming."""
    from core.replay import list_replay_scenarios
    return list_replay_scenarios()


@router.websocket("/ws/replay/{scenario_id}")
async def websocket_replay_stream(websocket: WebSocket, scenario_id: str):
    """Stream 60-second telemetry replay sequence with pause, stop, and speed controls."""
    await websocket.accept()
    from core.replay import get_scenario

    try:
        steps = get_scenario(scenario_id)
    except KeyError:
        await websocket.send_json({"error": f"Scenario '{scenario_id}' not found."})
        await websocket.close(code=1008)
        return

    # Playback control state
    state = {
        "step_index": 0,
        "is_paused": False,
        "speed": 1.0,
        "is_stopped": False,
    }

    async def listen_commands():
        try:
            while not state["is_stopped"]:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    action = msg.get("action", "").lower()
                    if action == "pause":
                        state["is_paused"] = True
                    elif action == "resume":
                        state["is_paused"] = False
                    elif action == "stop":
                        state["is_stopped"] = True
                        break
                    elif action == "set_speed":
                        new_spd = float(msg.get("speed", 1.0))
                        if new_spd > 0:
                            state["speed"] = new_spd
                except Exception:
                    pass
        except Exception:
            state["is_stopped"] = True

    listener_task = asyncio.create_task(listen_commands())

    try:
        while state["step_index"] < len(steps) and not state["is_stopped"]:
            if state["is_paused"]:
                await asyncio.sleep(0.05)
                continue

            step_data = steps[state["step_index"]]
            await websocket.send_json(step_data)
            state["step_index"] += 1

            delay = max(0.005, 1.0 / state["speed"])
            elapsed = 0.0
            while elapsed < delay and not state["is_stopped"]:
                chunk = min(0.05, delay - elapsed)
                await asyncio.sleep(chunk)
                elapsed += chunk

        if not state["is_stopped"]:
            await websocket.send_json({
                "type": "completed",
                "scenario_id": scenario_id,
                "total_steps": len(steps),
                "is_replay": True,
            })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        state["is_stopped"] = True
        listener_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass

