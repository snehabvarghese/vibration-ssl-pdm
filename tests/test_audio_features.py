"""
Unit tests for audio feature extraction, frame stacking, and normalization.

Tests verify:
  * output shapes (160000 samples -> (128, 313) log-Mel -> (309, 640) stacked frames)
  * numerical integrity (no NaNs, no Infs)
  * deterministic behavior across repeated calls
  * normalizer is fitted strictly on training data without leakage
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
from audio_model.preprocessing.build_dataset import stratified_train_val_split
from audio_model.preprocessing.features import (
    AudioNormalizer,
    load_audio,
    log_mel,
    stack_frames,
)


@pytest.fixture
def dummy_wav(tmp_path: Path) -> Path:
    """Generate a clean 10 s 16 kHz WAV file."""
    sr = 16_000
    duration_s = 10.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wav_path = tmp_path / "test_tone.wav"
    scipy.io.wavfile.write(str(wav_path), sr, sig)
    return wav_path


def test_load_audio_shape_and_dtype(dummy_wav: Path):
    """Verify load_audio returns exact length, float32, mono signal."""
    y = load_audio(dummy_wav, sr=16_000, duration_s=10.0)
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.shape == (160_000,)
    assert not np.isnan(y).any()


def test_load_audio_padding_and_trimming(tmp_path: Path):
    """Verify load_audio properly pads short clips and trims long clips."""
    sr = 16_000
    # 5s short clip
    short_sig = (np.random.randn(int(sr * 5)) * 1000).astype(np.int16)
    p_short = tmp_path / "short.wav"
    scipy.io.wavfile.write(str(p_short), sr, short_sig)

    y_pad = load_audio(p_short, sr=sr, duration_s=10.0)
    assert y_pad.shape == (160_000,)
    # Tail should be zero-padded
    assert np.all(y_pad[80_000:] == 0.0)

    # 12s long clip
    long_sig = (np.random.randn(int(sr * 12)) * 1000).astype(np.int16)
    p_long = tmp_path / "long.wav"
    scipy.io.wavfile.write(str(p_long), sr, long_sig)

    y_trim = load_audio(p_long, sr=sr, duration_s=10.0)
    assert y_trim.shape == (160_000,)


def test_log_mel_output_shape_and_no_nans(dummy_wav: Path):
    """Verify log-Mel spectrogram shape is (128, 313) without NaNs/Infs."""
    y = load_audio(dummy_wav, sr=16_000, duration_s=10.0)
    cfg = AUDIO_CFG.feature
    mel = log_mel(
        y,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        power=cfg.power,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    assert mel.shape == (128, 313)
    assert mel.dtype == np.float32
    assert not np.isnan(mel).any()
    assert not np.isinf(mel).any()


def test_stack_frames_shape_and_no_nans():
    """Verify DCASE-baseline context frame stacking shape: (309, 640)."""
    # 128 mel bins, 313 frames
    dummy_logmel = np.random.randn(128, 313).astype(np.float32)
    stacked = stack_frames(dummy_logmel, context_frames=5, frame_hop=1)
    # Expected: (313 - 5 + 1, 128 * 5) = (309, 640)
    assert stacked.shape == (309, 640)
    assert stacked.dtype == np.float32
    assert not np.isnan(stacked).any()

    # Test 3D batch shape: (4, 128, 313) -> (4, 309, 640)
    batch = np.random.randn(4, 128, 313).astype(np.float32)
    b_stacked = stack_frames(batch, context_frames=5, frame_hop=1)
    assert b_stacked.shape == (4, 309, 640)


def test_deterministic_output(dummy_wav: Path):
    """Verify feature pipeline produces bitwise-identical results across runs."""
    y1 = load_audio(dummy_wav, sr=16_000)
    m1 = log_mel(y1, sr=16_000)
    s1 = stack_frames(m1, context_frames=5)

    y2 = load_audio(dummy_wav, sr=16_000)
    m2 = log_mel(y2, sr=16_000)
    s2 = stack_frames(m2, context_frames=5)

    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


def test_normalizer_fitted_strictly_on_train(tmp_path: Path):
    """Verify normalizer fits ONLY on train normal clips and transforms without leakage."""
    rng = np.random.default_rng(42)

    # Simulated training normal data: mean ~ 10, std ~ 2
    train_data = rng.normal(loc=10.0, scale=2.0, size=(100, 128, 313)).astype(np.float32)

    # Simulated test anomaly data with huge shift: mean ~ 100, std ~ 20
    test_data = rng.normal(loc=100.0, scale=20.0, size=(20, 128, 313)).astype(np.float32)

    normalizer = AudioNormalizer()
    normalizer.fit(train_data)

    # Verify fitted stats reflect train_data, not test_data
    np.testing.assert_allclose(normalizer.mean.mean(), 10.0, atol=0.2)
    np.testing.assert_allclose(normalizer.std.mean(), 2.0, atol=0.2)

    # Normalize train: should have mean ~ 0, std ~ 1
    norm_train = normalizer.transform(train_data)
    np.testing.assert_allclose(norm_train.mean(), 0.0, atol=0.05)
    np.testing.assert_allclose(norm_train.std(), 1.0, atol=0.05)

    # Normalize test using TRAIN stats: test mean should remain ~ (100 - 10) / 2 = 45
    norm_test = normalizer.transform(test_data)
    np.testing.assert_allclose(norm_test.mean(), 45.0, atol=2.0)

    # Verify save & load serialization
    save_path = tmp_path / "scaler.npz"
    normalizer.save(save_path)
    loaded_norm = AudioNormalizer.load(save_path)

    re_norm_test = loaded_norm.transform(test_data)
    np.testing.assert_array_equal(norm_test, re_norm_test)


def test_stratified_train_val_split_no_leakage(tmp_path: Path):
    """Verify train/val split is disjoint with zero data leakage."""
    clip_paths = []
    for s in ["section_00", "section_01", "section_02"]:
        for d in ["source", "target"]:
            for i in range(20):
                p = tmp_path / f"{s}_{d}_train_normal_{i:04d}.wav"
                p.touch()
                clip_paths.append(p)

    train_clips, val_clips = stratified_train_val_split(clip_paths, val_fraction=0.1, seed=42)

    # Assert mutually exclusive
    overlap = set(train_clips).intersection(set(val_clips))
    assert len(overlap) == 0, f"Data leakage detected! Overlapping clips: {overlap}"
    assert len(train_clips) + len(val_clips) == len(clip_paths)
