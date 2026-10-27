"""
End-to-end integration tests for the unified Multimodal PdM System.

Validates:
1. GET /models: registry metadata and availability across all modalities.
2. All /predict/* endpoints with recorded held-out samples and uploads.
3. POST /fusion: multimodal late fusion with max, weighted, and vote methods.
4. Short replay stream of each scripted scenario over WebSocket.
5. Strict contract enforcement: is_replay=True for all replayed benchmark data.
6. Graceful error handling: invalid inputs return 4xx, never raw 500s.
7. Resilience to unavailable models: fusion continues operating without failure.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app
from core.replay import SCENARIO_METADATA

client = TestClient(app)


def _generate_test_wav_bytes(duration_s: float = 1.0, sr: int = 16000) -> bytes:
    """Generate in-memory mono PCM 16-bit WAV audio bytes."""
    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    # 440 Hz sine tone
    samples = (np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()


# ---------------------------------------------------------------------------
# 1. Models Catalogue Endpoint
# ---------------------------------------------------------------------------

def test_e2e_get_models():
    """Verify GET /models returns model info for vibration, audio, and motor."""
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) >= 3

    modalities = {m["modality"] for m in models}
    assert "vibration" in modalities
    assert "audio" in modalities
    assert "motor" in modalities

    for m in models:
        assert "model_name" in m
        assert "is_available" in m
        assert "threshold" in m
        assert "checkpoint_path" in m
        assert "training_dataset" in m


# ---------------------------------------------------------------------------
# 2. All /predict/* Endpoints with Recorded Benchmark Samples
# ---------------------------------------------------------------------------

def test_e2e_predict_vibration_sample():
    """Verify vibration prediction on recorded CWRU held-out sample."""
    samples_resp = client.get("/samples/vibration")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()
    assert len(samples) > 0
    sample_id = samples[0]["sample_id"]

    resp = client.post(f"/predict/vibration/sample/{sample_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["modality"] == "vibration"
    assert data["status"] in ("NORMAL", "ANOMALOUS", "UNAVAILABLE")
    assert "anomaly_score" in data
    assert "threshold" in data
    assert "score_ratio" in data
    assert data["is_replay"] is True


def test_e2e_predict_audio_sample():
    """Verify acoustic prediction on recorded MIMII DG held-out sample."""
    samples_resp = client.get("/samples/audio")
    assert samples_resp.status_code == 200
    samples = samples_resp.json()

    if len(samples) > 0:
        sample_id = samples[0]["sample_id"]
        resp = client.post(f"/predict/audio/sample/{sample_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["modality"] == "audio"
        assert data["status"] in ("NORMAL", "ANOMALOUS", "UNAVAILABLE")
        assert data["is_replay"] is True


def test_e2e_predict_motor_sample():
    """Verify motor prediction on recorded Paderborn held-out sample."""
    resp = client.post("/predict/motor/sample/paderborn_demo_normal")
    assert resp.status_code == 200
    data = resp.json()

    assert data["modality"] == "motor"
    assert data["status"] in ("NORMAL", "ANOMALOUS", "UNAVAILABLE")
    assert "anomaly_score" in data
    assert "threshold" in data
    assert "score_ratio" in data
    assert data["is_replay"] is True


def test_e2e_predict_file_uploads():
    """Verify file upload predictions across all modalities."""
    # 1. Vibration CSV upload
    t = np.linspace(0, 0.5, 6000)
    sig = np.sin(2 * np.pi * 50 * t)
    csv_bytes = "\n".join(str(v) for v in sig).encode("utf-8")
    resp_vib = client.post(
        "/predict/vibration",
        files={"file": ("test_signal.csv", csv_bytes, "text/csv")},
        data={"sampling_rate": 12000},
    )
    assert resp_vib.status_code == 200
    vib_data = resp_vib.json()
    assert vib_data["modality"] == "vibration"
    assert vib_data["is_replay"] is False

    # 2. Audio WAV upload
    wav_bytes = _generate_test_wav_bytes(duration_s=1.0, sr=16000)
    resp_audio = client.post(
        "/predict/audio",
        files={"file": ("test_audio.wav", wav_bytes, "audio/wav")},
        data={"machine_type": "fan"},
    )
    assert resp_audio.status_code == 200
    audio_data = resp_audio.json()
    assert audio_data["modality"] == "audio"
    assert audio_data["is_replay"] is False

    # 3. Motor CSV upload
    motor_csv = "current_p1,current_p2,temperature\n" + "\n".join(
        f"{np.sin(2 * np.pi * 50 * x)},{np.sin(2 * np.pi * 50 * x + 2.0)},45.2"
        for x in np.linspace(0, 0.1, 1000)
    )
    resp_motor = client.post(
        "/predict/motor",
        files={"file": ("test_motor.csv", motor_csv.encode("utf-8"), "text/csv")},
        data={"temperature": 45.2},
    )
    assert resp_motor.status_code == 200
    motor_data = resp_motor.json()
    assert motor_data["modality"] == "motor"
    assert motor_data["is_replay"] is False


# ---------------------------------------------------------------------------
# 3. Multimodal Fusion Endpoint
# ---------------------------------------------------------------------------

def test_e2e_post_fusion_methods():
    """Verify POST /fusion across max, weighted, and vote decision rules."""
    sample_results = [
        {
            "model_name": "vib_cwru",
            "modality": "vibration",
            "status": "NORMAL",
            "anomaly_score": 0.4,
            "threshold": 1.0,
            "score_ratio": 0.4,
            "is_replay": True,
        },
        {
            "model_name": "audio_ae",
            "modality": "audio",
            "status": "ANOMALOUS",
            "anomaly_score": 1.4,
            "threshold": 1.0,
            "score_ratio": 1.4,
            "is_replay": True,
        },
        {
            "model_name": "motor_mcsa",
            "modality": "motor",
            "status": "NORMAL",
            "anomaly_score": 0.3,
            "threshold": 0.5,
            "score_ratio": 0.6,
            "is_replay": True,
        },
    ]

    for method in ("max", "weighted", "vote"):
        resp = client.post("/fusion", json={"results": sample_results, "method": method})
        assert resp.status_code == 200
        fusion = resp.json()

        assert fusion["method"] == method
        assert fusion["fused_status"] in ("NORMAL", "WARNING", "ALARM", "UNAVAILABLE")
        assert "fused_score" in fusion
        assert "confidence" in fusion
        assert fusion["is_replay"] is True
        assert len(fusion["reasons"]) > 0


# ---------------------------------------------------------------------------
# 4. Short Replay of Each Scripted Scenario over WebSocket
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIO_METADATA)
def test_e2e_short_replay_scenarios(scenario):
    """Verify short replay streaming over WebSocket for every scenario in catalog."""
    sc_id = scenario["id"]
    with client.websocket_connect(f"/ws/replay/{sc_id}") as websocket:
        # Increase speed to avoid real-time delay
        websocket.send_json({"action": "set_speed", "speed": 100.0})

        # Receive first 3 steps
        for step_num in range(1, 4):
            msg = websocket.receive_json()
            assert msg["step"] == step_num
            assert msg["scenario_id"] == sc_id
            assert msg["is_replay"] is True

            # All model outputs must be tagged is_replay=True
            for mod in ("vibration", "audio", "motor"):
                assert mod in msg["models"]
                assert msg["models"][mod]["is_replay"] is True

            # Fusion verdict must be tagged is_replay=True
            assert msg["fusion"]["is_replay"] is True
            assert msg["fusion"]["fused_status"] in ("NORMAL", "WARNING", "ALARM", "UNAVAILABLE")

        # Graceful stop
        websocket.send_json({"action": "stop"})


# ---------------------------------------------------------------------------
# 5. Contract: is_replay True on All Replayed Benchmark Data
# ---------------------------------------------------------------------------

def test_e2e_is_replay_strict_contract():
    """Verify is_replay flag is strictly True across all replay endpoints and samples."""
    # 1. Scenario catalogue
    scenarios_resp = client.get("/replay/scenarios")
    assert scenarios_resp.status_code == 200
    for sc in scenarios_resp.json():
        assert sc["is_replay"] is True

    # 2. Sample listings
    for mod in ("vibration", "audio", "motor"):
        s_resp = client.get(f"/samples/{mod}")
        assert s_resp.status_code == 200
        for s in s_resp.json():
            assert s["is_replay"] is True

    # 3. Fusion via sample IDs
    resp = client.post(
        "/fusion",
        json={"sample_ids": {"vibration": "105", "motor": "paderborn_demo_normal"}},
    )
    assert resp.status_code == 200
    assert resp.json()["is_replay"] is True


# ---------------------------------------------------------------------------
# 6. Error Handling: No Raw 500s on Invalid Inputs
# ---------------------------------------------------------------------------

def test_e2e_no_raw_500_on_bad_inputs():
    """Ensure invalid parameters, corrupt payloads, and missing IDs yield 4xx, never raw 500."""
    # 1. Corrupt audio file type
    resp1 = client.post(
        "/predict/audio",
        files={"file": ("corrupt.txt", b"plain text content", "text/plain")},
        data={"machine_type": "fan"},
    )
    assert resp1.status_code == 400
    assert "Unsupported audio format" in resp1.json()["detail"]

    # 2. Unsupported machine category
    wav_bytes = _generate_test_wav_bytes(duration_s=0.5)
    resp2 = client.post(
        "/predict/audio",
        files={"file": ("sound.wav", wav_bytes, "audio/wav")},
        data={"machine_type": "turbine_unknown"},
    )
    assert resp2.status_code == 400
    assert "Unknown machine_type" in resp2.json()["detail"]

    # 3. Non-existent sample ID
    resp3 = client.post("/predict/vibration/sample/9999999_non_existent")
    assert resp3.status_code == 404

    # 4. Unknown modality in /samples
    resp4 = client.get("/samples/infrared_unknown")
    assert resp4.status_code == 400

    # 5. Missing fusion parameters
    resp5 = client.post("/fusion", json={})
    assert resp5.status_code == 400

    # 6. WebSocket non-existent scenario
    with client.websocket_connect("/ws/replay/invalid_scenario_id_xyz") as ws:
        err_msg = ws.receive_json()
        assert "error" in err_msg


# ---------------------------------------------------------------------------
# 7. Resilient Fusion with an Unavailable Model
# ---------------------------------------------------------------------------

def test_e2e_fusion_resilient_to_unavailable_model():
    """Verify fusion continues executing successfully when a model is marked UNAVAILABLE."""
    results_with_failure = [
        {
            "model_name": "audio_ae",
            "modality": "audio",
            "status": "UNAVAILABLE",
            "anomaly_score": 0.0,
            "threshold": 0.0,
            "score_ratio": 0.0,
            "error": "Acoustic sensor disconnected or checkpoint offline.",
            "is_replay": False,
        },
        {
            "model_name": "vib_cwru",
            "modality": "vibration",
            "status": "ANOMALOUS",
            "anomaly_score": 1.8,
            "threshold": 1.0,
            "score_ratio": 1.8,
            "is_replay": False,
        },
        {
            "model_name": "motor_mcsa",
            "modality": "motor",
            "status": "NORMAL",
            "anomaly_score": 0.2,
            "threshold": 0.5,
            "score_ratio": 0.4,
            "is_replay": False,
        },
    ]

    resp = client.post(
        "/fusion",
        json={"results": results_with_failure, "method": "max"},
    )
    assert resp.status_code == 200
    fusion = resp.json()

    # System did not crash; produces valid verdict from active sensors
    assert fusion["fused_status"] in ("WARNING", "ALARM")
    assert fusion["fused_score"] == 1.8
    assert "audio" in fusion["missing_models"]
    assert any("audio" in r.lower() for r in fusion["reasons"])
    # Confidence is penalized due to missing sensor modality
    assert fusion["confidence"] < 1.0
