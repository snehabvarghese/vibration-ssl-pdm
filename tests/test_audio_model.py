"""
Unit tests for the Acoustic Dense Autoencoder architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest
import torch

from audio_model.models.autoencoder import DenseAutoencoder


def test_autoencoder_shapes():
    """Verify DenseAutoencoder forward pass and bottleneck dimensions."""
    model = DenseAutoencoder(input_dim=640, hidden_dim=128, bottleneck_dim=8)
    x = torch.randn(16, 640)

    # Encode
    z = model.encode(x)
    assert z.shape == (16, 8), f"Expected bottleneck (16, 8), got {z.shape}"

    # Decode
    x_rec = model.decode(z)
    assert x_rec.shape == (16, 640), f"Expected reconstruction (16, 640), got {x_rec.shape}"

    # Forward
    out = model(x)
    assert out.shape == (16, 640)
    assert not torch.isnan(out).any()


def test_reconstruction_error_computation():
    """Verify element-wise frame reconstruction MSE computation."""
    model = DenseAutoencoder(input_dim=640, hidden_dim=128, bottleneck_dim=8)
    model.eval()
    x = torch.randn(32, 640)

    errors = model.get_reconstruction_errors(x)
    assert errors.shape == (32,)
    assert (errors >= 0.0).all()
    assert not torch.isnan(errors).any()


def test_autoencoder_overfits_dummy_batch():
    """Verify autoencoder gradient optimization reduces MSE on a tiny batch."""
    torch.manual_seed(42)
    model = DenseAutoencoder(input_dim=640, hidden_dim=128, bottleneck_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    x = torch.randn(8, 640)
    initial_loss = criterion(model(x), x).item()

    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion(model(x), x)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), x).item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_auc_and_pauc_mcclish():
    """Verify AUC and McClish-standardised partial AUC (pAUC with max_fpr=0.1)."""
    from audio_model.evaluation.evaluate_audio import compute_auc_and_pauc

    # Perfect separation: anomalies have strictly higher scores than normals
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    auc, pauc = compute_auc_and_pauc(y_true, scores_perfect, max_fpr=0.1)

    assert auc == 1.0, f"Expected perfect AUC 1.0, got {auc}"
    # Under McClish standardization, perfect detection yields 1.0
    assert pauc == 1.0, f"Expected perfect McClish pAUC 1.0, got {pauc}"

    # Inverted ranking
    scores_inverted = -scores_perfect
    auc_inv, _ = compute_auc_and_pauc(y_true, scores_inverted, max_fpr=0.1)
    assert auc_inv == 0.0, f"Expected inverted AUC 0.0, got {auc_inv}"


def test_binary_metrics_computation():
    """Verify confusion matrix and derived rates under a fixed threshold."""
    from audio_model.evaluation.evaluate_audio import compute_binary_metrics

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9])
    threshold = 0.5

    # Predictions: [0, 0, 1, 1, 0, 0, 1, 1]
    # y_true:      [0, 0, 0, 0, 1, 1, 1, 1]
    # TN: 2 (0.1, 0.2), FP: 2 (0.6, 0.7), FN: 2 (0.3, 0.4), TP: 2 (0.8, 0.9)
    res = compute_binary_metrics(y_true, scores, threshold)
    cm = res["confusion_matrix"]

    assert cm["tn"] == 2
    assert cm["fp"] == 2
    assert cm["fn"] == 2
    assert cm["tp"] == 2
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["false_alarm_rate"] == 0.5

