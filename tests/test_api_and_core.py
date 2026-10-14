"""
Unit and integration tests for core.analysis and the FastAPI backend.

Tests:
  - core.analysis loaders (encoder, monitor, embeddings, classifier, map)
  - core.analysis analyse_signal (valid signal, too short signal)
  - FastAPI GET /health
  - FastAPI POST /analyse with Normal CWRU recording (97.mat)
  - FastAPI POST /analyse with Faulty CWRU recording (105.mat)
  - FastAPI validation 400 errors (empty file, unsupported format, NaNs, too short, bad channel)
  - FastAPI CORS configuration for http://localhost:3000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app
from config import DATA_RAW_DIR
from core.analysis import (
    analyse_signal,
    get_classifier,
    get_embeddings,
    get_encoder,
    get_map,
    get_monitor,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# 1. core.analysis tests
# --------------------------------------------------------------------------
def test_core_loaders():
    """Verify all cached loaders return initialized artefacts."""
    encoder = get_encoder()
    assert encoder is not None, "Encoder failed to load"

    monitor = get_monitor()
    assert monitor is not None, "Health monitor failed to load"

    embeddings = get_embeddings()
    assert "embeddings" in embeddings and "labels" in embeddings
    assert len(embeddings["classes"]) == 16

    clf = get_classifier("svm_rbf")
    assert hasattr(clf, "predict_proba"), "Classifier must support predict_proba"

    xy, labels, classes, pca = get_map(max_points=500)
    assert xy.shape[1] == 2
    assert len(labels) == xy.shape[0]


def test_core_analyse_signal_synthetic():
    """Verify analyse_signal works on synthetic vibration input."""
    sr = 12000
    signal = np.random.randn(sr * 2)  # 2 seconds
    res = analyse_signal(signal, sr, clf_name="svm_rbf")

    assert res is not None
    expected_keys = {
        "n_segments",
        "embeddings",
        "predicted_class",
        "confidence",
        "class_probabilities",
        "edr_scores",
        "edr_mean",
        "edr_threshold",
        "anomalous",
        "segment_scores",
        "segment_threshold",
        "fraction_segments_flagged",
    }
    assert expected_keys.issubset(res.keys())
    assert isinstance(res["predicted_class"], str)
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["anomalous"], bool)


def test_core_analyse_signal_too_short():
    """Verify analyse_signal returns None for signals that cannot form a segment."""
    signal = np.random.randn(100)  # only 100 samples (< 0.25 s)
    res = analyse_signal(signal, 12000)
    assert res is None


# --------------------------------------------------------------------------
# 2. FastAPI API tests
# --------------------------------------------------------------------------
def test_api_health(client: TestClient):
    """Verify GET /health returns 200 with system metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["model_loaded"] is True


def test_api_cors_header(client: TestClient):
    """Verify CORS allow-origin header for http://localhost:3000."""
    resp = client.options(
        "/analyse",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_api_cors_header_vite(client: TestClient):
    """Verify CORS allow-origin header for http://localhost:5173 (Vite dev server)."""
    resp = client.options(
        "/analyse",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_api_analyse_normal_recording(client: TestClient):
    """POST /analyse with Normal CWRU recording (97.mat, 48 kHz)."""
    normal_path = DATA_RAW_DIR / "97.mat"
    if not normal_path.exists():
        pytest.skip("97.mat not found in data/raw")

    with open(normal_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/analyse",
        files={"file": ("97.mat", file_bytes, "application/octet-stream")},
        data={"sampling_rate": "48000", "channel": "DE", "clf_name": "svm_rbf"},
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()

    assert data["n_segments"] > 0
    assert isinstance(data["embeddings"], list)  # verify converted to list
    assert isinstance(data["edr_scores"], list)
    assert isinstance(data["segment_scores"], list)
    assert isinstance(data["anomalous"], bool)
    assert "Normal" in data["class_probabilities"]
    assert "waveform" in data and "samples" in data["waveform"]
    assert "map_xy" in data and isinstance(data["map_xy"], list)
    assert len(data["map_xy"]) == data["n_segments"]


def test_api_analyse_faulty_recording(client: TestClient):
    """POST /analyse with Faulty CWRU recording (105.mat, 12 kHz, IR007)."""
    fault_path = DATA_RAW_DIR / "105.mat"
    if not fault_path.exists():
        pytest.skip("105.mat not found in data/raw")

    with open(fault_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/analyse",
        files={"file": ("105.mat", file_bytes, "application/octet-stream")},
        data={"sampling_rate": "12000", "channel": "DE", "clf_name": "svm_rbf"},
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    data = resp.json()

    assert data["n_segments"] > 0
    assert isinstance(data["embeddings"], list)
    assert isinstance(data["predicted_class"], str)
    assert isinstance(data["anomalous"], bool)
    assert "waveform" in data and "samples" in data["waveform"]
    assert "map_xy" in data and len(data["map_xy"]) == data["n_segments"]


def test_api_get_map(client: TestClient):
    """GET /map returns background points and class list."""
    resp = client.get("/map")
    assert resp.status_code == 200
    data = resp.json()
    assert "points" in data and "classes" in data
    assert len(data["points"]) > 0
    assert "x" in data["points"][0] and "y" in data["points"][0] and "class_label" in data["points"][0]
    assert len(data["classes"]) == 16


def test_api_get_recordings(client: TestClient):
    """GET /recordings returns list of dataset files."""
    resp = client.get("/recordings")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0] and "label" in data[0] and "load_hp" in data[0]


def test_api_analyse_recording_endpoint(client: TestClient):
    """POST /analyse/recording/{file_id} analyses dataset file directly."""
    resp = client.post("/analyse/recording/105")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording_id"] == "105"
    assert data["true_label"] == "IR007"
    assert "waveform" in data
    assert "map_xy" in data
    assert data["n_segments"] > 0


def test_api_analyse_npy_and_csv(client: TestClient):
    """POST /analyse with .npy and .csv formatted vibration data."""
    # .npy upload test
    sig = np.random.randn(12000)
    buf_npy = io.BytesIO()
    np.save(buf_npy, sig)
    resp_npy = client.post(
        "/analyse",
        files={"file": ("test.npy", buf_npy.getvalue(), "application/octet-stream")},
        data={"sampling_rate": "12000"},
    )
    assert resp_npy.status_code == 200

    # .csv upload test
    import pandas as pd
    df = pd.DataFrame({"DE": sig})
    buf_csv = io.BytesIO()
    df.to_csv(buf_csv, index=False)
    resp_csv = client.post(
        "/analyse",
        files={"file": ("test.csv", buf_csv.getvalue(), "text/csv")},
        data={"sampling_rate": "12000", "channel": "DE"},
    )
    assert resp_csv.status_code == 200


def test_api_validation_errors(client: TestClient):
    """Verify 400 validation errors for bad inputs."""
    # 1. Empty file
    resp = client.post(
        "/analyse",
        files={"file": ("empty.mat", b"", "application/octet-stream")},
        data={"sampling_rate": "12000"},
    )
    assert resp.status_code == 400
    assert "Empty file" in resp.json()["detail"]

    # 2. Unsupported format
    resp = client.post(
        "/analyse",
        files={"file": ("bad.txt", b"hello world", "text/plain")},
        data={"sampling_rate": "12000"},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]

    # 3. NaNs in signal (.npy)
    sig_nan = np.array([1.0, np.nan, 3.0] * 2000)
    buf = io.BytesIO()
    np.save(buf, sig_nan)
    resp = client.post(
        "/analyse",
        files={"file": ("nan.npy", buf.getvalue(), "application/octet-stream")},
        data={"sampling_rate": "12000"},
    )
    assert resp.status_code == 400
    assert "NaN or Inf" in resp.json()["detail"]

    # 4. Signal too short (.npy)
    sig_short = np.random.randn(100)
    buf = io.BytesIO()
    np.save(buf, sig_short)
    resp = client.post(
        "/analyse",
        files={"file": ("short.npy", buf.getvalue(), "application/octet-stream")},
        data={"sampling_rate": "12000"},
    )
    assert resp.status_code == 400
    assert "too short" in resp.json()["detail"].lower()

    # 5. Bad channel in .mat
    normal_path = DATA_RAW_DIR / "97.mat"
    if normal_path.exists():
        with open(normal_path, "rb") as f:
            file_bytes = f.read()
        resp = client.post(
            "/analyse",
            files={"file": ("97.mat", file_bytes, "application/octet-stream")},
            data={"sampling_rate": "48000", "channel": "NONEXISTENT_CHANNEL"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()
