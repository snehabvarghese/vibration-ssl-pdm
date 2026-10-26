"""
Unit tests for the audio preprocessing and exploration module (MIMII DG).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import scipy.io.wavfile

from audio_model.config_audio import AUDIO_CFG
from audio_model.preprocessing.download_mimii_dg import (
    MACHINE_CATALOG,
    is_already_extracted,
)
from audio_model.preprocessing.explore import (
    compute_log_mel_spectrogram,
    compute_mel_filterbank,
    parse_wav_filename,
    read_audio_specs,
    scan_dataset,
)


def test_parse_wav_filename_standard():
    """Verify standard DCASE 2022 Task 2 filename parsing."""
    fname = "section_00_source_train_normal_0000_m-n_W.wav"
    meta = parse_wav_filename(fname)
    assert meta["section"] == "section_00"
    assert meta["domain"] == "source"
    assert meta["split"] == "train"
    assert meta["condition"] == "normal"
    assert meta["index"] == "0000"
    assert meta["attributes"] == "m-n_W"


def test_parse_wav_filename_anomaly_target():
    """Verify parsing target domain anomaly filenames."""
    fname = "section_02_target_test_anomaly_0045_vel_12.wav"
    meta = parse_wav_filename(fname)
    assert meta["section"] == "section_02"
    assert meta["domain"] == "target"
    assert meta["split"] == "test"
    assert meta["condition"] == "anomaly"
    assert meta["index"] == "0045"


def test_mel_filterbank_dimensions():
    """Verify Mel filterbank generation shape and bounds."""
    fb = compute_mel_filterbank(sr=16000, n_fft=1024, n_mels=128, f_min=50.0, f_max=8000.0)
    assert fb.shape == (128, 513)
    assert fb.min() >= 0.0
    assert fb.max() > 0.0


def test_log_mel_spectrogram_output():
    """Verify log-mel spectrogram shape and finiteness on dummy 16 kHz audio."""
    sr = 16000
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # 440 Hz pure tone
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)

    times, freqs, log_mel = compute_log_mel_spectrogram(
        sig,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
    )
    assert log_mel.shape[0] == 64
    assert log_mel.shape[1] > 0
    assert not np.isnan(log_mel).any()
    assert not np.isinf(log_mel).any()


def test_machine_catalog_urls():
    """Verify Zenodo record file catalogue has expected entries."""
    for m in ["fan", "bearing", "gearbox", "slider", "valve"]:
        assert m in MACHINE_CATALOG
        entry = MACHINE_CATALOG[m]
        assert entry["zip_name"].startswith("dev_")
        assert entry["size_bytes"] > 500_000_000
        assert "zenodo.org/api/records/6355122" in entry["url"]


def test_scan_dataset_mock(tmp_path: Path):
    """Test scanner on a synthetic mock dataset folder."""
    raw_dir = tmp_path / "raw"
    fan_dir = raw_dir / "fan"
    train_dir = fan_dir / "train"
    test_dir = fan_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    sr = 16000
    dummy_audio = (np.random.randn(sr) * 1000).astype(np.int16)

    # Save mock train normal and test anomaly clips
    f_tr_src = train_dir / "section_00_source_train_normal_0000.wav"
    f_tr_tgt = train_dir / "section_00_target_train_normal_0001.wav"
    f_te_anom = test_dir / "section_00_source_test_anomaly_0000.wav"
    f_te_norm = test_dir / "section_00_target_test_normal_0000.wav"

    scipy.io.wavfile.write(str(f_tr_src), sr, dummy_audio)
    scipy.io.wavfile.write(str(f_tr_tgt), sr, dummy_audio)
    scipy.io.wavfile.write(str(f_te_anom), sr, dummy_audio)
    scipy.io.wavfile.write(str(f_te_norm), sr, dummy_audio)

    # Test read_audio_specs
    s_sr, s_dur, s_ch = read_audio_specs(f_tr_src)
    assert s_sr == sr
    assert s_dur == 1.0
    assert s_ch == 1

    summary = scan_dataset(raw_dir)
    assert "fan" in summary
    fan_stats = summary["fan"]
    assert fan_stats["total_clips"] == 4
    assert "section_00" in fan_stats["sections"]
    sec0 = fan_stats["sections"]["section_00"]
    assert sec0["train"]["source_normal"] == 1
    assert sec0["train"]["target_normal"] == 1
    assert sec0["test"]["source_anomaly"] == 1
    assert sec0["test"]["target_normal"] == 1
