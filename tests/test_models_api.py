"""
Unit tests for the Unified Model API and Adapters (core/models_api).

Verifies compliance with the integration contract:
- ModelResult schema, types, and to_dict() serialization
- BaseModelAdapter abstract interface
- VibrationAdapter (CWRU SSL)
- AudioAdapter (MIMII DG autoencoder)
- Motor adapters (Paderborn Current, Temperature, Combined)
- ModelRegistry lazy lookup, aliases, and resilient error handling
- score_ratio >= 1.0 contract for all anomalous predictions
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from core.models_api.interface import BaseModelAdapter, ModelResult
from core.models_api.vibration_adapter import VibrationAdapter
from core.models_api.audio_adapter import AudioAdapter
from core.models_api.motor_adapter import (
    BaseMotorAdapter,
    MotorCombinedAdapter,
    MotorCurrentAdapter,
    MotorTemperatureAdapter,
)
from core.models_api.registry import ModelRegistry, get_registry


# ---------------------------------------------------------------------------
# 1. ModelResult Contract Tests
# ---------------------------------------------------------------------------

def test_model_result_dataclass_fields():
    result = ModelResult(
        model_name="test_model",
        modality="vibration",
        status="NORMAL",
        anomaly_score=0.12345,
        threshold=0.5,
        score_ratio=0.2469,
        prediction_label="Normal",
        confidence=0.98,
        details={"extra_info": 42},
        input_info={"source": "test_signal"},
        is_replay=True,
    )

    assert result.model_name == "test_model"
    assert result.modality == "vibration"
    assert result.status == "NORMAL"
    assert result.anomaly_score == 0.12345
    assert result.threshold == 0.5
    assert result.score_ratio == 0.2469
    assert result.prediction_label == "Normal"
    assert result.confidence == 0.98
    assert result.is_replay is True
    assert result.error is None
    assert isinstance(result.timestamp, str)


def test_model_result_to_dict_json_serializable():
    result = ModelResult(
        model_name="test_model",
        modality="audio",
        status="ANOMALOUS",
        anomaly_score=np.float32(0.85),
        threshold=0.45,
        score_ratio=1.8889,
        details={
            "numpy_array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "numpy_int": np.int64(10),
            "numpy_bool": np.bool_(True),
        },
        input_info={"samples": np.int32(16000)},
        is_replay=False,
    )

    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert isinstance(res_dict["details"]["numpy_array"], list)
    assert isinstance(res_dict["details"]["numpy_int"], int)
    assert isinstance(res_dict["details"]["numpy_bool"], bool)

    # Must serialize to valid JSON without error
    serialized = json.dumps(res_dict)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert deserialized["status"] == "ANOMALOUS"


def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModelAdapter(name="dummy", modality="vibration")


# ---------------------------------------------------------------------------
# 2. VibrationAdapter Tests
# ---------------------------------------------------------------------------

def test_vibration_adapter_info():
    adapter = VibrationAdapter()
    info = adapter.info()
    assert "model_name" in info
    assert info["modality"] == "vibration"
    assert "checkpoint_path" in info
    assert "threshold" in info
    assert "training_dataset" in info


def test_vibration_adapter_missing_checkpoint_handling():
    dummy_path = Path("non_existent_encoder_checkpoint_12345.pt")
    adapter = VibrationAdapter(encoder_path=dummy_path)
    assert not adapter.is_available()

    # Predict must return UNAVAILABLE and not raise
    sig = np.zeros(3000, dtype=np.float32)
    res = adapter.predict(sig, sampling_rate=12000, is_replay=True)
    assert res.status == "UNAVAILABLE"
    assert res.is_replay is True
    assert res.error is not None
    assert "missing" in res.error.lower()


def test_vibration_adapter_predict_valid_signal():
    adapter = VibrationAdapter()
    if not adapter.is_available():
        pytest.skip("Vibration checkpoints not present.")

    # 3000 samples at 12 kHz = 0.25 s (1 window)
    sig = np.sin(2 * np.pi * 50 * np.linspace(0, 0.25, 3000)) * 0.1
    res = adapter.predict(sig, sampling_rate=12000, is_replay=True)

    assert res.modality == "vibration"
    assert res.status in ("NORMAL", "ANOMALOUS")
    assert res.is_replay is True
    assert isinstance(res.anomaly_score, float)
    assert isinstance(res.threshold, float)
    assert isinstance(res.score_ratio, float)

    if res.status == "ANOMALOUS":
        assert res.score_ratio >= 1.0


# ---------------------------------------------------------------------------
# 3. AudioAdapter Tests
# ---------------------------------------------------------------------------

def test_audio_adapter_info():
    adapter = AudioAdapter(machine_type="fan")
    info = adapter.info()
    assert info["modality"] == "audio"
    assert info["machine_type"] == "fan"
    assert "threshold" in info
    assert "checkpoint_path" in info


def test_audio_adapter_missing_checkpoint():
    dummy_dir = Path("non_existent_checkpoints_dir_audio")
    adapter = AudioAdapter(machine_type="fan", checkpoints_dir=dummy_dir)
    assert not adapter.is_available()

    res = adapter.predict(np.zeros(16000), is_replay=True)
    assert res.status == "UNAVAILABLE"
    assert res.is_replay is True
    assert res.error is not None


def test_audio_adapter_predict_valid_signal():
    adapter = AudioAdapter(machine_type="fan")
    if not adapter.is_available():
        pytest.skip("Audio fan autoencoder checkpoint not found.")

    # 1 second of normal audio noise
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.05, 16000).astype(np.float32)

    res = adapter.predict(audio, sampling_rate=16000, is_replay=False)
    assert res.modality == "audio"
    assert res.status in ("NORMAL", "ANOMALOUS")
    assert res.is_replay is False
    assert res.error is None

    if res.status == "ANOMALOUS":
        assert res.score_ratio >= 1.0


# ---------------------------------------------------------------------------
# 4. Motor Adapters Tests (Paderborn)
# ---------------------------------------------------------------------------

def test_motor_adapters_info():
    comb = MotorCombinedAdapter()
    curr = MotorCurrentAdapter()
    temp = MotorTemperatureAdapter()

    assert comb.info()["sub_type"] == "combined"
    assert curr.info()["sub_type"] == "current"
    assert temp.info()["sub_type"] == "temperature"


def test_motor_adapter_predict():
    comb = MotorCombinedAdapter()
    if not comb.is_available():
        pytest.skip("Motor combined checkpoint not found.")

    # 64,000 samples of 50 Hz stator current at 64 kHz
    t = np.linspace(0, 1.0, 64000, endpoint=False)
    p1 = 5.0 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(64000)

    res = comb.predict(p1, temperature=48.0, sampling_rate=64000, is_replay=True)
    assert res.modality == "motor"
    assert res.status in ("NORMAL", "ANOMALOUS")
    assert res.is_replay is True
    assert res.threshold == 0.5

    if res.status == "ANOMALOUS":
        assert res.score_ratio >= 1.0


def test_motor_adapter_dict_and_scalar_input():
    temp_adapter = MotorTemperatureAdapter()
    if not temp_adapter.is_available():
        pytest.skip("Motor temperature checkpoint not found.")

    # Predict using temperature scalar input
    res = temp_adapter.predict(45.0, is_replay=False)
    assert res.modality == "motor"
    assert res.status in ("NORMAL", "ANOMALOUS")
    assert res.input_info["temperature_deg_c"] == 45.0


def test_motor_adapter_missing_checkpoint():
    class DummyMotorAdapter(BaseMotorAdapter):
        @property
        def checkpoint_path(self) -> Path:
            return Path("non_existent_motor_model.joblib")

    adapter = DummyMotorAdapter("dummy_motor", "dummy_mode", "combined")
    assert not adapter.is_available()

    res = adapter.predict(np.zeros(100))
    assert res.status == "UNAVAILABLE"
    assert "Checkpoint not found" in str(res.error)


# ---------------------------------------------------------------------------
# 5. ModelRegistry Tests
# ---------------------------------------------------------------------------

def test_registry_registration_and_aliases():
    registry = ModelRegistry()

    # Vibration lookup
    assert registry.get_adapter("vibration") is not None
    assert registry.get_adapter("vibration_cwru_ssl") is not None
    assert registry.get_adapter("cwru") is not None

    # Audio lookup
    assert registry.get_adapter("audio") is not None
    assert registry.get_adapter("audio_fan") is not None
    assert registry.get_adapter("audio_bearing") is not None

    # Motor lookup
    assert registry.get_adapter("motor") is not None
    assert registry.get_adapter("motor_current") is not None
    assert registry.get_adapter("motor_temperature") is not None

    # Unknown model
    assert registry.get_adapter("non_existent_model") is None


def test_registry_list_models():
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) >= 4
    for m in models:
        assert "model_name" in m
        assert "modality" in m
        assert "is_available" in m


def test_registry_predict_unknown_model_returns_unavailable():
    registry = ModelRegistry()
    res = registry.predict("unregistered_model_xyz", input_data=np.zeros(10))

    assert res.status == "UNAVAILABLE"
    assert "not registered" in str(res.error)


def test_global_get_registry():
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2
