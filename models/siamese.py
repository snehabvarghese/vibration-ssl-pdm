"""
Siamese contrastive model + NT-Xent loss (the core of Phase 1).

ARCHITECTURE
------------
        view A ──┐                    ┌── z_A = g(f(view A))
                 ├─ shared encoder f ─┤
        view B ──┘   + proj. head g   └── z_B = g(f(view B))

"Siamese" = both branches are literally the same module with the same weights,
so the two views are mapped by the same function. Only the *inputs* differ.

WHY A PROJECTION HEAD
---------------------
The contrastive loss is applied to g(f(x)), not to f(x). The head absorbs the
information that is only useful for solving the augmentation-invariance task
(e.g. "how much noise was added"), leaving f(x) a more general representation.
After pretraining the head is thrown away and f(x) is used everywhere else --
this is the standard SimCLR finding and is why the embedding we export is the
encoder output, not the projection output.

NT-Xent (normalized temperature-scaled cross entropy, a.k.a. InfoNCE)
---------------------------------------------------------------------
For a batch of N segments we build 2N views. For an anchor i with positive
j(i) (the other view of the same segment):

        l_i = -log [ exp(sim(z_i, z_j) / tau) /
                     sum_{k != i} exp(sim(z_i, z_k) / tau) ]

        sim(u, v) = u . v / (||u|| ||v||)     (cosine similarity)

The loss is the mean of l_i over all 2N views. tau (temperature) controls how
sharply the softmax penalises hard negatives; smaller tau = harder penalty.

NOTE: no labels appear anywhere in this file. Positives are defined purely by
"came from the same segment", which is what makes this self-supervised.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG, ModelConfig
from models.encoder import CNN1DEncoder


class ProjectionHead(nn.Module):
    """2-layer MLP used only during contrastive pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SiameseContrastiveModel(nn.Module):
    """Shared 1-D CNN encoder + projection head."""

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = cfg or CFG.model
        self.encoder = CNN1DEncoder(cfg)
        self.projector = ProjectionHead(
            cfg.embedding_dim, cfg.projection_hidden_dim, cfg.projection_dim
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the projected embeddings of both views."""
        return self.projector(self.encoder(x_a)), self.projector(self.encoder(x_b))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference path used after pretraining: encoder only, head bypassed."""
        self.eval()
        return self.encoder(x)


class NTXentLoss(nn.Module):
    """NT-Xent / InfoNCE loss over 2N views built from N segments.
    Optionally supports record-aware negative masking to eliminate false negatives
    coming from overlapping segments of the same record/class.
    """

    def __init__(self, temperature: Optional[float] = None) -> None:
        super().__init__()
        self.temperature = temperature if temperature is not None else CFG.ssl.temperature

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        record_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = z_a.shape[0]
        if n < 2:
            raise ValueError("NT-Xent needs a batch size of at least 2 to have negatives")

        z = torch.cat([z_a, z_b], dim=0)              # (2N, D)
        z = F.normalize(z, dim=1)                     # cosine similarity via dot product
        sim = z @ z.t() / self.temperature            # (2N, 2N)

        # Mask out self-similarity so an anchor cannot match itself.
        eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))

        if record_ids is not None:
            # Mask out false negatives coming from the same recording/file
            rec_2n = torch.cat([record_ids, record_ids], dim=0)
            same_rec_mask = (rec_2n.unsqueeze(0) == rec_2n.unsqueeze(1))
            # Keep true positive pairs (i <-> i+n)
            pos_mask = torch.zeros((2 * n, 2 * n), dtype=torch.bool, device=z.device)
            pos_idx = torch.arange(n, device=z.device)
            pos_mask[pos_idx, pos_idx + n] = True
            pos_mask[pos_idx + n, pos_idx] = True

            # False negatives: same record, but not self, and not true positive pair
            false_neg_mask = same_rec_mask & (~pos_mask) & (~eye)
            sim = sim.masked_fill(false_neg_mask, float("-inf"))

        # Positive of index i is i+N (and vice versa).
        targets = torch.cat(
            [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
        )
        return F.cross_entropy(sim, targets)
