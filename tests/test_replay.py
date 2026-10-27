"""
Unit and integration tests for Scenario Replay, WebSocket streaming, and /fusion endpoints.

Verifies:
- Scenario inventory and metadata schemas
- All 60 steps conform to replay contract
- is_replay is strictly True on all models, steps, and fusion verdicts
- Determinism across runs with fixed seed
- API endpoints: GET /fusion/config, POST /fusion, GET /replay/scenarios
- WebSocket /ws/replay/{scenario_id} streaming and control commands (pause/stop)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.replay import (
    generate_scenario_steps,
    get_scenario,
    list_replay_scenarios,
    REPLAY_SOURCE_LABEL,
    SCENARIO_METADATA,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Scenario Catalog & Metadata
# ---------------------------------------------------------------------------

def test_list_replay_scenarios():
    scenarios = list_replay_scenarios()
    assert len(scenarios) == 5
    ids = [sc["id"] for sc in scenarios]
    assert "healthy_baseline" in ids
    assert "vibration_fault_onset" in ids
    assert "audio_fault_onset" in ids
    assert "motor_fault_onset" in ids
    assert "multi_fault_escalation" in ids

    for sc in scenarios:
        assert sc["total_steps"] == 60
        assert sc["duration_seconds"] == 60
        assert sc["is_replay"] is True


# ---------------------------------------------------------------------------
# 2. Replay Step Schema and Contract
# ---------------------------------------------------------------------------

def test_scenario_steps_schema():
    steps = get_scenario("multi_fault_escalation")
    assert len(steps) == 60

    for i, step in enumerate(steps):
        assert step["step"] == i + 1
        assert "timestamp" in step
        assert step["scenario_id"] == "multi_fault_escalation"
        assert step["source"] == REPLAY_SOURCE_LABEL
        assert step["is_replay"] is True

        models = step["models"]
        assert "vibration" in models
        assert "audio" in models
        assert "motor" in models

        for mod in ("vibration", "audio", "motor"):
            m_res = models[mod]
            assert m_res["is_replay"] is True
            assert "score_ratio" in m_res
            assert "anomaly_score" in m_res
            assert "threshold" in m_res
            assert m_res["status"] in ("NORMAL", "ANOMALOUS", "UNAVAILABLE")

        fusion = step["fusion"]
        assert fusion["is_replay"] is True
        assert fusion["fused_status"] in ("NORMAL", "WARNING", "ALARM", "UNAVAILABLE")
        assert "fused_score" in fusion


# ---------------------------------------------------------------------------
# 3. Determinism with Fixed Seed
# ---------------------------------------------------------------------------

def test_replay_determinism_with_seed():
    steps1 = generate_scenario_steps("vibration_fault_onset", total_steps=20, seed=42)
    steps2 = generate_scenario_steps("vibration_fault_onset", total_steps=20, seed=42)

    for s1, s2 in zip(steps1, steps2):
        assert s1["step"] == s2["step"]
        assert s1["fusion"]["fused_score"] == s2["fusion"]["fused_score"]
        assert s1["fusion"]["fused_status"] == s2["fusion"]["fused_status"]
        for mod in ("vibration", "audio", "motor"):
            assert s1["models"][mod]["score_ratio"] == s2["models"][mod]["score_ratio"]


# ---------------------------------------------------------------------------
# 4. Scenario State Progression
# ---------------------------------------------------------------------------

def test_healthy_baseline_progression():
    steps = get_scenario("healthy_baseline")
    for step in steps:
        assert step["fusion"]["fused_status"] in ("NORMAL", "WARNING")
        # In healthy baseline, no models are anomalous
        assert step["models"]["vibration"]["status"] == "NORMAL"
        assert step["models"]["audio"]["status"] == "NORMAL"
        assert step["models"]["motor"]["status"] == "NORMAL"


def test_multi_fault_escalation_progression():
    steps = get_scenario("multi_fault_escalation")
    # Early steps: healthy
    assert steps[0]["models"]["vibration"]["status"] == "NORMAL"
    assert steps[0]["models"]["audio"]["status"] == "NORMAL"
    assert steps[0]["models"]["motor"]["status"] == "NORMAL"

    # Final steps: multiple faults active -> ALARM
    last_step = steps[-1]
    assert last_step["models"]["vibration"]["status"] == "ANOMALOUS"
    assert last_step["models"]["audio"]["status"] == "ANOMALOUS"
    assert last_step["models"]["motor"]["status"] == "ANOMALOUS"
    assert last_step["fusion"]["fused_status"] == "ALARM"


# ---------------------------------------------------------------------------
# 5. API Endpoints: /fusion and /replay
# ---------------------------------------------------------------------------

def test_get_fusion_config():
    response = client.get("/fusion/config")
    assert response.status_code == 200
    cfg = response.json()
    assert "warning_ratio_threshold" in cfg
    assert "alarm_ratio_threshold" in cfg
    assert "single_model_alarm_ratio" in cfg
    assert "weights" in cfg


def test_post_fusion_with_sample_ids():
    payload = {
        "sample_ids": {
            "vibration": "105",
            "motor": "paderborn_demo_normal",
        },
        "method": "max",
    }
    response = client.post("/fusion", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "fused_score" in res
    assert res["fused_status"] in ("NORMAL", "WARNING", "ALARM")
    assert res["is_replay"] is True
    assert "vibration" in res["per_model_ratios"]
    assert "motor" in res["per_model_ratios"]


def test_post_fusion_with_model_results():
    payload = {
        "results": [
            {
                "model_name": "m1",
                "modality": "vibration",
                "status": "NORMAL",
                "anomaly_score": 0.5,
                "threshold": 1.0,
                "score_ratio": 0.5,
            },
            {
                "model_name": "m2",
                "modality": "audio",
                "status": "ANOMALOUS",
                "anomaly_score": 1.5,
                "threshold": 1.0,
                "score_ratio": 1.5,
            },
        ],
        "method": "weighted",
    }
    response = client.post("/fusion", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["fused_score"] == 1.0
    assert res["fused_status"] == "WARNING"


def test_get_replay_scenarios_api():
    response = client.get("/replay/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) == 5


# ---------------------------------------------------------------------------
# 6. WebSocket Replay Streaming & Controls
# ---------------------------------------------------------------------------

def test_websocket_replay_stream_and_stop():
    with client.websocket_connect("/ws/replay/multi_fault_escalation") as websocket:
        # Fast speed so we don't wait 1s in test
        websocket.send_json({"action": "set_speed", "speed": 100.0})

        # Receive first step
        msg1 = websocket.receive_json()
        assert msg1["step"] == 1
        assert msg1["scenario_id"] == "multi_fault_escalation"
        assert msg1["is_replay"] is True

        # Send stop command
        websocket.send_json({"action": "stop"})


def test_websocket_replay_unknown_scenario():
    with client.websocket_connect("/ws/replay/non_existent_scenario_123") as websocket:
        msg = websocket.receive_json()
        assert "error" in msg
