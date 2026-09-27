"""
Smoke tests for the Phase 1 pipeline.

These are deliberately small and run on synthetic signals, so they need no
downloaded data and finish in seconds. They guard the properties that are easy
to break silently and expensive to notice later:

  * preprocessing shapes and normalization
  * augmented views stay related to the original signal
  * encoder output shape and the projection head being bypassable
  * NT-Xent behaves like a contrastive loss (aligned views -> lower loss)
  * record-level splits never leak a record across splits
  * EDR is ~0 for healthy-vs-healthy and large for shifted distributions

Run:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anomaly_detection.edr import EDRDetector
from config import CFG
from models.encoder import CNN1DEncoder
from models.siamese import NTXentLoss, SiameseContrastiveModel
from preprocessing.augmentation import make_two_views
from preprocessing.filtering import downsample_signal, filter_signal
from preprocessing.segmentation import normalize_signal, preprocess_record, segment_signal
from training.splits import assert_no_leakage, split_by_record


def synthetic_signal(seconds: float = 4.0, rate: int = 12_000, seed: int = 0) -> np.ndarray:
    """A bearing-like signal: two tones plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 60 * t) + 0.4 * np.sin(2 * np.pi * 3_000 * t)
            + 0.1 * rng.standard_normal(t.size))


# --------------------------------------------------------------- preprocessing
def test_filter_keeps_length_and_removes_dc():
    x = synthetic_signal() + 5.0          # large DC offset
    y = filter_signal(x, 12_000)
    assert y.shape == x.shape
    assert abs(y.mean()) < 0.1            # 10 Hz high-pass edge kills the offset


def test_downsample_factor_one_is_a_no_op():
    x = synthetic_signal(0.5)
    y, rate = downsample_signal(x, 12_000, factor=1)
    assert rate == 12_000
    np.testing.assert_allclose(y, x)


def test_segmentation_overlap_and_count():
    x = np.arange(1000, dtype=float)
    seg = segment_signal(x, window_size=100, overlap=0.5)
    assert seg.shape == (19, 100)         # hop 50 -> floor((1000-100)/50)+1
    np.testing.assert_allclose(seg[1], x[50:150])


def test_zscore_normalization_is_per_segment():
    seg = np.stack([synthetic_signal(0.1, seed=i) * (i + 1) for i in range(4)])
    z = normalize_signal(seg, method="zscore")
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-6)


def test_preprocess_record_end_to_end():
    segments, rate = preprocess_record(synthetic_signal(2.0), 12_000)
    expected_len = int(CFG.preprocess.window_seconds * rate)
    assert segments.ndim == 2 and segments.shape[1] == expected_len
    assert np.isfinite(segments).all()


# --------------------------------------------------------------- augmentation
def test_two_views_differ_but_stay_finite():
    rng = np.random.default_rng(0)
    x = normalize_signal(synthetic_signal(0.25))
    a, b = make_two_views(x, rng=rng)
    assert a.shape == b.shape == x.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.allclose(a, b)          # two views must not be identical


# --------------------------------------------------------------- model
def test_encoder_output_shape_and_channel_handling():
    enc = CNN1DEncoder()
    x = torch.randn(8, 3_000)
    assert enc(x).shape == (8, CFG.model.embedding_dim)
    assert enc(x.unsqueeze(1)).shape == (8, CFG.model.embedding_dim)


def test_embed_bypasses_projection_head():
    model = SiameseContrastiveModel()
    x = torch.randn(4, 3_000)
    z = model.embed(x)
    assert z.shape == (4, CFG.model.embedding_dim)
    za, _ = model(x, x)
    assert za.shape == (4, CFG.model.projection_dim)


def test_ntxent_rewards_aligned_views():
    loss_fn = NTXentLoss(temperature=0.1)
    torch.manual_seed(0)
    z = torch.randn(16, 32)
    aligned = float(loss_fn(z, z.clone()))
    misaligned = float(loss_fn(z, torch.randn(16, 32)))
    assert aligned < misaligned


def test_ntxent_rejects_degenerate_batch():
    with pytest.raises(ValueError):
        NTXentLoss()(torch.randn(1, 8), torch.randn(1, 8))


# --------------------------------------------------------------- splits
def test_record_split_never_leaks_a_record():
    record_ids = np.repeat([f"r{i}" for i in range(20)], 10)
    labels = np.repeat(np.arange(20) % 4, 10)
    split = split_by_record(record_ids, labels)
    assert_no_leakage(record_ids, split)          # raises if a record straddles
    assert sum(split.sizes().values()) == len(record_ids)


# --------------------------------------------------------------- EDR
def test_edr_small_for_healthy_and_large_for_shifted():
    rng = np.random.default_rng(0)
    healthy = rng.standard_normal((400, 16))
    det = EDRDetector(hop_seconds=0.125).fit(healthy)

    same = det.divergence(rng.standard_normal((100, 16)))
    shifted = det.divergence(rng.standard_normal((100, 16)) + 6.0)
    assert same >= 0
    assert shifted > 10 * max(same, 1e-6)


def test_edr_stream_flags_only_after_calibration():
    rng = np.random.default_rng(1)
    healthy = rng.standard_normal((600, 16))
    det = EDRDetector(hop_seconds=0.125).fit(healthy)
    det.calibrate(rng.standard_normal((300, 16)))
    faulty = rng.standard_normal((300, 16)) * 3 + 5
    assert det.score_stream(faulty).is_anomalous.mean() > 0.9
