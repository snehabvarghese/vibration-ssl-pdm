"""
Supervised 1-D CNN baseline.

WHY IT EXISTS
-------------
It is the control experiment for the whole project. It uses the *same*
encoder architecture and the *same* labelled subset as the self-supervised
pipeline, but is trained end-to-end from random initialisation with
cross-entropy -- i.e. no unlabelled data is exploited.

Keeping the architecture identical is deliberate: any difference in accuracy
can then be attributed to the self-supervised pretraining rather than to one
model simply being bigger.

INPUT  : (batch, 1, window) raw z-scored segments
OUTPUT : (batch, n_classes) logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config import CFG, ModelConfig
from models.encoder import CNN1DEncoder


class BaselineCNN(nn.Module):
    """Same encoder backbone + a linear classification head."""

    def __init__(self, n_classes: int, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = cfg or CFG.model
        self.encoder = CNN1DEncoder(cfg)
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
