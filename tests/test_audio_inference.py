"""
Unit and integration tests for audio anomaly detection inference (predict_audio).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from audio_model.config_audio import CHECKPOINTS_AUDIO_DIR, RAW_AUDIO_DIR
from audio_model.inference import predict_audio


def get_sample_test_clips():
    """Locate a known normal and anomalous test clip from raw dataset."""
    fan_test_dir = RAW_AUDIO_DIR / "fan" / "test"
    if not fan_test_dir.exists():
        return None, None

    normal_clip = fan_test_dir / "section_00_source_test_normal_0000_m-n_W.wav"
    anomaly_clip = fan_test_dir / "section_00_source_test_anomaly_0003_m-n_W.wav"

    if not normal_clip.exists() or not anomaly_clip.exists():
        return None, None

    return normal_clip, anomaly_clip


def test_predict_audio_end_to_end():
    """Verify normal and anomalous test clips score properly and return all required keys and types."""
    checkpoint_file = CHECKPOINTS_AUDIO_DIR / "fan_ae.pt"
    if not checkpoint_file.exists():
        pytest.skip("Audio checkpoint 'fan_ae.pt' not found. Run training first.")

    normal_clip, anomaly_clip = get_sample_test_clips()
    if normal_clip is None or anomaly_clip is None:
        pytest.skip("Test audio clips not found in data/audio/raw/fan/test.")

    expected_keys = {
        "machine_type",
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "status",
        "frame_scores",
        "sampling_rate_used",
    }

    # 1. Normal Clip
    res_normal = predict_audio(normal_clip, machine_type="fan", threshold_percentile=95)
    assert set(res_normal.keys()) == expected_keys
    assert res_normal["machine_type"] == "fan"
    assert isinstance(res_normal["anomaly_score"], float)
    assert isinstance(res_normal["threshold"], float)
    assert isinstance(res_normal["is_anomaly"], bool)
    assert res_normal["status"] in ("NORMAL", "ANOMALOUS")
    assert isinstance(res_normal["frame_scores"], list)
    assert len(res_normal["frame_scores"]) > 0
    assert all(isinstance(x, float) for x in res_normal["frame_scores"])
    assert res_normal["sampling_rate_used"] == 16000

    assert res_normal["is_anomaly"] is False
    assert res_normal["status"] == "NORMAL"
    assert res_normal["anomaly_score"] <= res_normal["threshold"]

    # 2. Anomalous Clip
    res_anomalous = predict_audio(anomaly_clip, machine_type="fan", threshold_percentile=95)
    assert set(res_anomalous.keys()) == expected_keys
    assert res_anomalous["machine_type"] == "fan"
    assert isinstance(res_anomalous["anomaly_score"], float)
    assert isinstance(res_anomalous["threshold"], float)
    assert isinstance(res_anomalous["is_anomaly"], bool)
    assert res_anomalous["status"] in ("NORMAL", "ANOMALOUS")
    assert isinstance(res_anomalous["frame_scores"], list)
    assert len(res_anomalous["frame_scores"]) > 0
    assert res_anomalous["sampling_rate_used"] == 16000

    assert res_anomalous["is_anomaly"] is True
    assert res_anomalous["status"] == "ANOMALOUS"
    assert res_anomalous["anomaly_score"] > res_anomalous["threshold"]


def test_predict_audio_numpy_inputs():
    """Verify inference accepts raw numpy arrays, pads short audio, windows long audio, and resamples."""
    checkpoint_file = CHECKPOINTS_AUDIO_DIR / "fan_ae.pt"
    if not checkpoint_file.exists():
        pytest.skip("Audio checkpoint 'fan_ae.pt' not found. Run training first.")

    # 1. Short audio (e.g. 3 seconds at 16 kHz) -> padded to 10s
    short_signal = np.random.randn(16_000 * 3).astype(np.float32)
    res_short = predict_audio(short_signal, sampling_rate=16_000, machine_type="fan")
    assert res_short["sampling_rate_used"] == 16_000
    assert len(res_short["frame_scores"]) == 309

    # 2. Long audio (e.g. 15 seconds at 16 kHz) -> windowed & averaged
    long_signal = np.random.randn(16_000 * 15).astype(np.float32)
    res_long = predict_audio(long_signal, sampling_rate=16_000, machine_type="fan")
    assert res_long["sampling_rate_used"] == 16_000
    assert len(res_long["frame_scores"]) > 309

    # 3. Audio with different sampling rate (e.g. 22,050 Hz) -> resampled
    diff_sr_signal = np.random.randn(22_050 * 5).astype(np.float32)
    res_resampled = predict_audio(diff_sr_signal, sampling_rate=22_050, machine_type="fan")
    assert res_resampled["sampling_rate_used"] == 16_000
    assert isinstance(res_resampled["anomaly_score"], float)


def test_invalid_inputs_raise_errors():
    """Verify empty audio, silent audio, nonexistent files, and missing checkpoints raise clear errors."""
    checkpoint_file = CHECKPOINTS_AUDIO_DIR / "fan_ae.pt"
    if not checkpoint_file.exists():
        pytest.skip("Audio checkpoint 'fan_ae.pt' not found.")

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        predict_audio(np.array([], dtype=np.float32), machine_type="fan")

    # Silent array (all zeros)
    with pytest.raises(ValueError, match="silent"):
        predict_audio(np.zeros(16_000, dtype=np.float32), machine_type="fan")

    # Silent array (near-zero amplitude)
    with pytest.raises(ValueError, match="silent"):
        predict_audio(np.ones(16_000, dtype=np.float32) * 1e-7, machine_type="fan")

    # Nonexistent file
    with pytest.raises(FileNotFoundError, match="does not exist"):
        predict_audio("nonexistent_audio_path_12345.wav", machine_type="fan")

    # Machine type with no checkpoint
    dummy_signal = np.random.randn(16_000).astype(np.float32)
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        predict_audio(dummy_signal, machine_type="unsupported_machine_xyz")
