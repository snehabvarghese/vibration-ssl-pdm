"""
1-D CNN encoder -- the component that is actually transferred between stages.

WHAT IT DOES
------------
Maps a raw (z-scored) vibration window to a fixed-length embedding:

    (batch, 1, window)  --conv blocks-->  (batch, C, T')  --GAP-->  (batch, embedding_dim)

WHY THIS DESIGN
---------------
* Wide first kernel (15 samples). Bearing faults appear as short impulses;
  a wide first receptive field captures a whole impulse and its decay rather
  than a couple of samples of it. This mirrors the "wide-first-kernel" design
  that is standard for 1-D vibration CNNs.
* Progressively narrower kernels with stride-2 convolutions + pooling
  aggregate those local impulse shapes into longer-range structure while
  keeping the parameter count small enough to train on a CPU.
* BatchNorm + ReLU for stable optimisation; dropout for regularisation.
* Global average pooling at the end makes the encoder **length-agnostic**: the
  same weights accept a longer or shorter window, which matters for Phase 2
  where the live stream's chunk size may differ.

INPUT  : float tensor (batch, in_channels, window)
OUTPUT : float tensor (batch, embedding_dim)

The encoder is shared (same weights) by both branches of the Siamese model,
and is reused untouched by the downstream classifier and the baseline.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from config import CFG, ModelConfig


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size) if pool_size > 1 else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DEncoder(nn.Module):
    """Stack of `ConvBlock`s + global average pooling + linear projection."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = cfg or CFG.model
        self.cfg = cfg

        channels: List[int] = [cfg.in_channels] + list(cfg.conv_channels)
        blocks = []
        for i in range(len(cfg.conv_channels)):
            blocks.append(
                ConvBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=cfg.kernel_sizes[i],
                    stride=cfg.strides[i],
                    pool_size=cfg.pool_size,
                    dropout=cfg.dropout,
                )
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.conv_channels[-1], cfg.embedding_dim)
        self.embedding_dim = cfg.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, L) -> (B, embedding_dim). Accepts (B, L) for convenience."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.features(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
