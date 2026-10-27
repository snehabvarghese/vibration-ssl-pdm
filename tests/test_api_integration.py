"""
Integration tests for the extended FastAPI backend endpoints:
- GET  /models
- POST /predict/vibration
- POST /predict/audio
- POST /predict/motor
- GET  /samples/{modality}
- POST /predict/{modality}/sample/{sample_id}
- Error handling (empty files, invalid formats, NaN/Inf, missing checkpoints returning UNAVAILABLE)
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api.main import app
from core.models_api import get_registry

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET /models
# ---------------------------------------------------------------------------

def test_get_models_list():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 4

    names = [m["model_name"] for m in models]
    assert "vibration_cwru_ssl" in names
    assert "audio_mimii_dg_ae" in names
    assert "motor_current_temperature_combined" in names

    for m in models:
        assert "model_name" in m
        assert "modality" in m
        assert "is_available" in m
        assert "threshold" in m
        assert "checkpoint_path" in m


# ---------------------------------------------------------------------------
# 2. POST /predict/vibration
# ---------------------------------------------------------------------------

def test_predict_vibration_valid_npy():
    # 3000 samples @ 12 kHz (0.25s)
    arr = np.sin(np.linspace(0, 50, 3000)).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    files = {"file": ("test_signal.npy", buf, "application/octet-stream")}
    data = {"sampling_rate": 12000, "clf_name": "svm_rbf"}
    response = client.post("/predict/vibration", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["model_name"] == "vibration_cwru_ssl"
    assert res["modality"] == "vibration"
    assert res["status"] in ("NORMAL", "ANOMALOUS")
    assert res["is_replay"] is False
    assert "edr_mean" in res["details"]
    assert res["input_info"]["samples"] == 3000


def test_predict_vibration_empty_file():
    files = {"file": ("empty.npy", io.BytesIO(b""), "application/octet-stream")}
    response = client.post("/predict/vibration", files=files)
    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]


def test_predict_vibration_invalid_format():
    files = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/predict/vibration", files=files)
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_predict_vibration_signal_too_short():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    files = {"file": ("short.npy", buf, "application/octet-stream")}
    response = client.post("/predict/vibration", files=files)
    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. POST /predict/audio
# ---------------------------------------------------------------------------

def test_predict_audio_valid_wav():
    # 1 second of random noise at 16 kHz
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32) * 0.05
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    files = {"file": ("test_fan.wav", buf, "audio/wav")}
    data = {"machine_type": "fan", "threshold_percentile": 95}
    response = client.post("/predict/audio", files=files, data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["modality"] == "audio"
    assert res["status"] in ("NORMAL", "ANOMALOUS")
    assert res["is_replay"] is False
    assert "frame_scores" in res["details"]


def test_predict_audio_invalid_extension():
    files = {"file": ("test.mp3", io.BytesIO(b"dummy mp3 data"), "audio/mp3")}
    response = client.post("/predict/audio", files=files)
    assert response.status_code == 400
    assert "Only .wav files are supported" in response.json()["detail"]


def test_predict_audio_invalid_machine_type():
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)

    files = {"file": ("test.wav", buf, "audio/wav")}
    data = {"machine_type": "jet_engine"}
    response = client.post("/predict/audio", files=files, data=data)
    assert response.status_code == 400
    assert "Unknown machine_type" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 4. POST /predict/motor
# ---------------------------------------------------------------------------

def test_predict_motor_valid_current_array_form():
    # Generate 640 samples representing stator current
    current_vals = [f"{5.0 * np.sin(2 * np.pi * 50 * (i / 64000)):.4f}" for i in range(640)]
    current_str = ",".join(current_vals)

    data = {
        "current_phase1": current_str,
        "temperature": 45.0,
        "sampling_rate": 64000,
        "model_mode": "combined",
    }
    response = client.post("/predict/motor", data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["modality"] == "motor"
    assert res["status"] in ("NORMAL", "ANOMALOUS")
    assert res["is_replay"] is False
    assert "features" in res["details"]


def test_predict_motor_temperature_only_scalar():
    data = {
        "temperature": 52.5,
        "model_mode": "temperature_only",
    }
    response = client.post("/predict/motor", data=data)

    assert response.status_code == 200
    res = response.json()
    assert res["modality"] == "motor"
    assert res["status"] in ("NORMAL", "ANOMALOUS")


def test_predict_motor_missing_inputs():
    response = client.post("/predict/motor", data={"model_mode": "combined"})
    assert response.status_code == 400
    assert "Missing motor input" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 5. GET /samples/{modality}
# ---------------------------------------------------------------------------

def test_list_samples_vibration():
    response = client.get("/samples/vibration")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    if samples:
        s0 = samples[0]
        assert s0["modality"] == "vibration"
        assert s0["is_replay"] is True
        assert "sample_id" in s0


def test_list_samples_audio():
    response = client.get("/samples/audio")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    if samples:
        s0 = samples[0]
        assert s0["modality"] == "audio"
        assert s0["is_replay"] is True
        assert s0["sampling_rate"] == 16000


def test_list_samples_motor():
    response = client.get("/samples/motor")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 3  # Built-in demo presets
    s_ids = [s["sample_id"] for s in samples]
    assert "paderborn_demo_normal" in s_ids


def test_list_samples_invalid_modality():
    response = client.get("/samples/radiation")
    assert response.status_code == 400
    assert "Unsupported modality" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 6. POST /predict/{modality}/sample/{sample_id}
# ---------------------------------------------------------------------------

def test_predict_motor_sample_replay():
    response = client.post("/predict/motor/sample/paderborn_demo_normal")
    assert response.status_code == 200
    res = response.json()
    assert res["modality"] == "motor"
    assert res["is_replay"] is True
    assert res["status"] in ("NORMAL", "ANOMALOUS")


def test_predict_sample_not_found():
    response = client.post("/predict/vibration/sample/non_existent_sample_99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 7. Mocked missing checkpoint returning UNAVAILABLE
# ---------------------------------------------------------------------------

def test_predict_audio_mocked_missing_checkpoint():
    adapter = get_registry().get_adapter("audio")
    assert adapter is not None

    with patch.object(adapter, "is_available", return_value=False):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        buf.seek(0)

        files = {"file": ("test.wav", buf, "audio/wav")}
        data = {"machine_type": "fan"}

        response = client.post("/predict/audio", files=files, data=data)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "UNAVAILABLE"
        assert res["error"] is not None
        assert "missing" in res["error"].lower()


def test_predict_vibration_mocked_missing_checkpoint():
    adapter = get_registry().get_adapter("vibration")
    assert adapter is not None

    with patch.object(adapter, "is_available", return_value=False):
        arr = np.sin(np.linspace(0, 50, 3000)).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        files = {"file": ("test.npy", buf, "application/octet-stream")}
        response = client.post("/predict/vibration", files=files)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "UNAVAILABLE"
        assert res["error"] is not None
        assert "missing" in res["error"].lower()
