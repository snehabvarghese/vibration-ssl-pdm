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
