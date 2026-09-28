"""
Unit and smoke tests for the vibration-ssl-pdm pipeline.

Includes tests for:
  - Encoder shape assertion
  - NT-Xent loss computation (with & without record-aware negative mask)
  - Split leak-free assertion
  - EDR detector calculation
  - Signal preprocessing and segmentation
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from anomaly_detection.edr import EDRDetector
from config import CFG
from models.encoder import CNN1DEncoder
from models.siamese import NTXentLoss, SiameseContrastiveModel
from preprocessing.segmentation import segment_signal
from training.splits import assert_no_leakage, make_split


def test_encoder_shape():
    """Verify 1-D CNN encoder maps (B, 1, 3000) -> (B, 128)."""
    enc = CNN1DEncoder(CFG.model)
    x = torch.randn(4, 1, 3000)
    out = enc(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"
    assert not torch.isnan(out).any(), "Encoder output contains NaNs"


def test_siamese_model_forward():
    """Verify SiameseContrastiveModel produces two (B, 64) projected views."""
    model = SiameseContrastiveModel()
    va = torch.randn(4, 1, 3000)
    vb = torch.randn(4, 1, 3000)
    za, zb = model(va, vb)
    assert za.shape == (4, CFG.model.projection_dim)
    assert zb.shape == (4, CFG.model.projection_dim)


def test_ntxent_loss():
    """Verify NT-Xent loss calculation."""
    criterion = NTXentLoss(temperature=0.1)
    za = torch.randn(8, 64)
    zb = torch.randn(8, 64)
    loss = criterion(za, zb)
    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() > 0.0, "Loss must be positive"


def test_ntxent_loss_record_aware():
    """Verify NT-Xent loss with record-aware negative masking."""
    criterion = NTXentLoss(temperature=0.1)
    za = torch.randn(8, 64)
    zb = torch.randn(8, 64)
    rec_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = criterion(za, zb, record_ids=rec_ids)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_split_leakage_assertion():
    """Verify record-level split has zero record leakage between train/val/test."""
    record_ids = np.array([f"rec_{i//10}" for i in range(100)])
    labels = np.random.randint(0, 4, size=100)
    loads = np.zeros(100, dtype=int)
    split = make_split(record_ids, labels, loads)
    assert_no_leakage(record_ids, split)


def test_segmentation():
    """Verify signal segmentation cuts expected window counts."""
    signal = np.random.randn(12000)  # 1s @ 12kHz
    wins = segment_signal(signal, window_size=3000, hop_size=1500)
    assert wins.shape[1] == 3000
    assert len(wins) > 0


def test_edr_detector():
    """Verify EDR detector fitting and score_stream computation."""
    healthy_emb = np.random.randn(100, 128)
    edr = EDRDetector(hop_seconds=0.25)
    edr.fit(healthy_emb)
    tau = edr.calibrate(healthy_emb[50:])
    assert tau > 0.0

    test_emb = np.random.randn(50, 128)
    stream = edr.score_stream(test_emb, stride=4)
    assert len(stream.scores) > 0
    assert not np.isnan(stream.scores).any()


if __name__ == "__main__":
    print("Running pipeline smoke tests...")
    test_encoder_shape()
    print("✓ test_encoder_shape passed")
    test_siamese_model_forward()
    print("✓ test_siamese_model_forward passed")
    test_ntxent_loss()
    print("✓ test_ntxent_loss passed")
    test_ntxent_loss_record_aware()
    print("✓ test_ntxent_loss_record_aware passed")
    test_split_leakage_assertion()
    print("✓ test_split_leakage_assertion passed")
    test_segmentation()
    print("✓ test_segmentation passed")
    test_edr_detector()
    print("✓ test_edr_detector passed")
    print("ALL SMOKE TESTS PASSED!")
