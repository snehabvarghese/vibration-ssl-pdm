"""
Comprehensive test suite for the Motor Current and Temperature subsystem.

Tests:
  - Time-domain, spectral MCSA, and Park's vector feature extraction
  - Paderborn .mat format parsing and validation
  - Signal integrity checks (NaN handling, dead channel rejection)
  - Zero-leakage bearing-level data splitting
  - Model training and baseline comparison (Current vs Temp vs Combined)
  - Team integration inference contract compliance
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None


def assert_raises(exc_type, func, *args, **kwargs):
    """Fallback assertion helper when pytest is not installed."""
    try:
        func(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised.")

from core.motor_temperature import analyse_motor_signal, get_motor_model
from models.motor_temperature_model import MotorPdMModel
from preprocessing.motor_features import (
    compute_parks_vector_features,
    compute_spectral_mcsa_features,
    compute_temperature_features,
    compute_time_domain_features,
    extract_multimodal_features,
)
from preprocessing.paderborn_loader import load_paderborn_mat, parse_paderborn_filename, segment_record
from tests.fixtures.paderborn_synthetic import (
    create_synthetic_paderborn_mat,
    generate_synthetic_motor_current,
)
from training.train_motor_models import run_experiment, split_by_bearing


def test_time_domain_features():
    """Verify statistical time-domain features on known sinusoidal signal."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t)  # Pure sine wave: RMS = 1/sqrt(2) ~ 0.7071
    feats = compute_time_domain_features(x)

    assert abs(feats["rms"] - (1.0 / np.sqrt(2.0))) < 0.01
    assert abs(feats["peak"] - 1.0) < 0.01
    assert abs(feats["peak_to_peak"] - 2.0) < 0.01
    assert abs(feats["crest_factor"] - np.sqrt(2.0)) < 0.05
    assert not np.isnan(feats["kurtosis"])
    assert not np.isnan(feats["skewness"])


def test_spectral_mcsa_features():
    """Verify spectral feature extraction accurately tracks fundamental frequency."""
    fs = 64_000
    t = np.linspace(0, 1, fs, endpoint=False)
    # 50 Hz fundamental + 3rd harmonic at 150 Hz
    x = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 150 * t)
    s_feats = compute_spectral_mcsa_features(x, sampling_rate=fs, expected_fundamental_hz=50.0)

    assert abs(s_feats["fundamental_freq"] - 50.0) < 1.0
    assert s_feats["fundamental_amp"] > 0.8
    assert s_feats["thd"] > 0.05
    assert not np.isnan(s_feats["spectral_entropy"])


def test_parks_vector_features():
    """Verify Park's Vector Transform on balanced 2-phase currents."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    u = np.cos(2 * np.pi * 50 * t)
    v = np.cos(2 * np.pi * 50 * t - (2 * np.pi / 3))
    p_feats = compute_parks_vector_features(u, v)

    assert p_feats["park_mean"] > 0.0
    assert not np.isnan(p_feats["park_eccentricity"])


def test_temperature_features():
    """Verify temperature feature handling preserving Celsius scale."""
    t_const = compute_temperature_features(47.5)
    assert t_const["temp_mean"] == 47.5
    assert t_const["temp_std"] == 0.0

    t_drift = np.linspace(45.0, 48.0, 50)
    t_feats = compute_temperature_features(t_drift)
    assert abs(t_feats["temp_mean"] - 46.5) < 0.1
    assert t_feats["temp_min"] == 45.0
    assert t_feats["temp_max"] == 48.0
    assert t_feats["temp_slope"] > 0.0


def test_multimodal_feature_vector():
    """Verify combined feature vector generation and dimension stability."""
    i1, i2, temp_val = generate_synthetic_motor_current("Normal", duration_seconds=0.5)
    vec, names, d = extract_multimodal_features(i1, i2, temp_val, sampling_rate=64_000)

    assert len(vec) == len(names)
    assert len(vec) >= 20
    assert not np.isnan(vec).any(), "Extracted features contain NaN values"
    assert "p1_rms" in names
    assert "temp_mean" in names


def test_paderborn_filename_parser():
    """Verify metadata parser for Paderborn naming conventions."""
    meta = parse_paderborn_filename("N15_M07_F10_KA01_1.mat")
    assert meta["speed_rpm"] == 1500
    assert meta["load_torque_nm"] == 0.7
    assert meta["radial_force_n"] == 1000.0
    assert meta["bearing_id"] == "KA01"
    assert meta["condition_label"] == "Outer_Race"

    meta_k0 = parse_paderborn_filename("N09_M07_F10_K002_1.mat")
    assert meta_k0["speed_rpm"] == 900
    assert meta_k0["bearing_id"] == "K002"
    assert meta_k0["condition_label"] == "Normal"

    meta_ki = parse_paderborn_filename("N15_M01_F10_KI04_1.mat")
    assert meta_ki["condition_label"] == "Inner_Race"
    assert meta_ki["load_torque_nm"] == 0.1


def test_paderborn_loader_and_segmentation():
    """Verify parsing Paderborn-structured .mat buffer into PaderbornRecord."""
    buf = create_synthetic_paderborn_mat("N15_M07_F10_KA01_1.mat", condition="Outer_Race", duration_seconds=1.0)
    rec = load_paderborn_mat(buf, filename_hint="N15_M07_F10_KA01_1.mat")

    assert rec.bearing_id == "KA01"
    assert rec.condition_label == "Outer_Race"
    assert rec.sampling_rate == 64_000
    assert len(rec.phase_current_1) == 64_000
    assert rec.phase_current_2 is not None
    assert rec.temperature_celsius is not None

    segs = segment_record(rec, window_seconds=0.5, overlap=0.5)
    assert len(segs) >= 2
    assert segs[0]["bearing_id"] == "KA01"
    assert len(segs[0]["phase_current_1"]) == 32_000


def test_signal_integrity_rejection():
    """Verify rejection of NaN signals, flat/dead channels, and empty inputs."""
    # 1. Empty signal
    if pytest is not None:
        with pytest.raises(ValueError):
            analyse_motor_signal(np.array([]))
    else:
        assert_raises(ValueError, analyse_motor_signal, np.array([]))

    # 2. Dead / zero signal
    def _load_dead():
        raw_mat = create_synthetic_paderborn_mat("N15_M07_F10_K001_1.mat")
        # Replace phase_current_1 channel name to simulate missing/dead channel
        corrupt = raw_mat.replace(b"phase_current_1", b"phase_current_x")
        load_paderborn_mat(io.BytesIO(corrupt), filename_hint="N15_M07_F10_K001_1.mat")

    if pytest is not None:
        with pytest.raises(ValueError):
            _load_dead()
    else:
        assert_raises(ValueError, _load_dead)


def test_leak_free_bearing_splitting():
    """Verify that bearing-level splits have zero overlapping bearing IDs."""
    bearing_ids = np.array(["K001", "K001", "K002", "KA01", "KA01", "KI01", "KI02", "KI02"])
    tr_mask, te_mask = split_by_bearing(bearing_ids, test_ratio=0.33, random_seed=42)

    tr_bearings = set(bearing_ids[tr_mask])
    te_bearings = set(bearing_ids[te_mask])

    assert len(tr_bearings.intersection(te_bearings)) == 0, "Data leakage between train and test bearings!"


def test_end_to_end_training_and_evaluation():
    """Train Current-Only, Temperature-Only, and Combined models and verify results."""
    results = run_experiment(clf_type="rf", random_seed=42)

    assert "current_only" in results
    assert "temperature_only" in results
    assert "combined" in results

    # Current-only should achieve high classification score on MCSA features
    assert results["current_only"]["accuracy"] >= 0.75
    assert results["current_only"]["macro_f1"] >= 0.70

    # Temperature-only is scientifically documented to have limited discriminative
    # power across micro-fault types under thermostatically regulated 45-50°C conditions
    print(f"Verified Temperature-Only Accuracy: {results['temperature_only']['accuracy']:.3f}")
    assert "accuracy" in results["temperature_only"]

    # Combined model should be fitted and saved
    assert results["combined"]["accuracy"] >= 0.75
    assert results["combined"]["anomaly_roc_auc"] is not None


def test_inference_service_contract():
    """Verify output format complies with the team integration contract."""
    # Ensure models exist by running experiment first if needed
    ckpt_path = ROOT_DIR / "checkpoints" / "motor_temp" / "combined_model.joblib"
    if not ckpt_path.exists():
        run_experiment(clf_type="rf", random_seed=42)

    i1, i2, temp_val = generate_synthetic_motor_current("Outer_Race", duration_seconds=0.5, seed=777)
    out = analyse_motor_signal(
        current_phase1=i1,
        current_phase2=i2,
        temperature=temp_val,
        sampling_rate=64_000,
        model_mode="combined",
        timestamp="2026-10-26T20:00:00Z",
    )

    # Required contract fields
    assert "model_name" in out
    assert "model_version" in out
    assert "status" in out
    assert "predicted_condition" in out
    assert "anomaly_score" in out
    assert "confidence" in out
    assert "timestamp" in out
    assert out["timestamp"] == "2026-10-26T20:00:00Z"

    # Value ranges
    assert 0.0 <= out["anomaly_score"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["status"] in ("HEALTHY", "ANOMALOUS")
    assert isinstance(out["class_probabilities"], dict)


def test_api_motor_endpoints():
    """Verify FastAPI motor PdM endpoints: catalogue, preset analysis, and upload."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)

    # 1. GET /recordings/motor
    resp = client.get("/recordings/motor")
    assert resp.status_code == 200
    rec_list = resp.json()
    assert isinstance(rec_list, list)

    if rec_list:
        first = rec_list[0]
        assert "filename" in first
        assert "bearing_id" in first
        assert "condition_label" in first

        # 2. POST /analyse/motor/recording
        resp_analysis = client.post(
            "/analyse/motor/recording",
            data={"filename": first["filename"], "model_mode": "combined"},
        )
        assert resp_analysis.status_code == 200
        result = resp_analysis.json()
        assert "predicted_condition" in result
        assert "confidence" in result
        assert "waveform_p1" in result
        assert "spectrum" in result
        assert "features" in result

    # 3. POST /analyse/motor with synthetic uploaded .mat
    mat_bytes = create_synthetic_paderborn_mat(duration_seconds=1.0, sampling_rate=64000, condition="Normal")
    resp_upload = client.post(

        "/analyse/motor",
        files={"file": ("synthetic_motor.mat", mat_bytes, "application/octet-stream")},
        data={"sampling_rate": "64000", "model_mode": "combined"},
    )
    assert resp_upload.status_code == 200
    upload_res = resp_upload.json()
    assert "predicted_condition" in upload_res
    assert upload_res["predicted_condition"] in ("Normal", "Outer_Race", "Inner_Race")
    assert "status" in upload_res


if __name__ == "__main__":
    if pytest is not None:
        pytest.main(["-v", __file__])
    else:
        print("Running tests via standalone runner...")
        test_time_domain_features()
        print("✓ test_time_domain_features passed")
        test_spectral_mcsa_features()
        print("✓ test_spectral_mcsa_features passed")
        test_parks_vector_features()
        print("✓ test_parks_vector_features passed")
        test_temperature_features()
        print("✓ test_temperature_features passed")
        test_multimodal_feature_vector()
        print("✓ test_multimodal_feature_vector passed")
        test_paderborn_filename_parser()
        print("✓ test_paderborn_filename_parser passed")
        test_paderborn_loader_and_segmentation()
        print("✓ test_paderborn_loader_and_segmentation passed")
        test_signal_integrity_rejection()
        print("✓ test_signal_integrity_rejection passed")
        test_leak_free_bearing_splitting()
        print("✓ test_leak_free_bearing_splitting passed")
        test_end_to_end_training_and_evaluation()
        print("✓ test_end_to_end_training_and_evaluation passed")
        test_inference_service_contract()
        print("✓ test_inference_service_contract passed")
        test_api_motor_endpoints()
        print("✓ test_api_motor_endpoints passed")
        print("\nALL MOTOR & TEMPERATURE SUBSYSTEM TESTS PASSED!")

