"""
Dense Autoencoder for Sound Anomaly Detection on stacked log-Mel frames.

Architecture:
  * Input: n_mels * context_frames (e.g. 128 * 5 = 640)
  * Encoder: 640 -> 128 -> 128 -> 128 -> 128 -> 8 (bottleneck)
  * Decoder: 8 -> 128 -> 128 -> 128 -> 128 -> 640 (symmetric)
  * Activations: BatchNorm1d + ReLU on hidden layers; linear reconstruction output.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    """Symmetric Dense Autoencoder with 8-dimensional bottleneck for anomaly detection."""

    def __init__(
        self,
        input_dim: int = 640,
        hidden_dim: int = 128,
        bottleneck_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder: 640 -> 128 -> 128 -> 128 -> 128 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, bottleneck_dim),
        )

        # Decoder: 8 -> 128 -> 128 -> 128 -> 128 -> 640
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input frames to latent bottleneck vectors."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent bottleneck vectors back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward reconstruction pass: x -> encode -> decode -> x_rec."""
        z = self.encode(x)
        return self.decode(z)

    def get_reconstruction_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate element-wise MSE reconstruction error per frame vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            1-D tensor of shape (N,) with the mean squared reconstruction error per vector.
        """
        x_rec = self.forward(x)
        return torch.mean((x - x_rec) ** 2, dim=-1)
