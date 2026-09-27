"""
PyTorch dataset wrappers around the processed segment file.

Two views of the same data:

  ContrastiveSegmentDataset -> returns (view_A, view_B); NO labels. Used for
                               self-supervised pretraining.
  SupervisedSegmentDataset  -> returns (segment, label). Used for the
                               supervised baseline and for embedding export.

Both read the single `data/processed/segments.npz` produced by
`preprocessing.build_dataset`, so every experiment sees identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG, DATA_PROCESSED_DIR, AugmentConfig
from preprocessing.augmentation import make_two_views

DEFAULT_PROCESSED = DATA_PROCESSED_DIR / "segments.npz"


@dataclass
class ProcessedData:
    """In-memory view of the processed dataset."""

    segments: np.ndarray      # (N, L) float32
    labels: np.ndarray        # (N,) int64
    classes: List[str]
    record_ids: np.ndarray    # (N,) str
    loads: np.ndarray         # (N,) int64
    fault_types: np.ndarray   # (N,) str
    sampling_rate: int

    def __len__(self) -> int:
        return self.segments.shape[0]

    def subset(self, idx: Sequence[int]) -> "ProcessedData":
        idx = np.asarray(idx)
        return ProcessedData(
            segments=self.segments[idx],
            labels=self.labels[idx],
            classes=self.classes,
            record_ids=self.record_ids[idx],
            loads=self.loads[idx],
            fault_types=self.fault_types[idx],
            sampling_rate=self.sampling_rate,
        )


def load_processed(path: Path = DEFAULT_PROCESSED) -> ProcessedData:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m preprocessing.build_dataset` first"
        )
    d = np.load(path, allow_pickle=False)
    return ProcessedData(
        segments=d["segments"],
        labels=d["labels"],
        classes=[str(c) for c in d["classes"]],
        record_ids=d["record_ids"].astype(str),
        loads=d["loads"],
        fault_types=d["fault_types"].astype(str),
        sampling_rate=int(d["sampling_rate"]),
    )


class ContrastiveSegmentDataset(Dataset):
    """Yields two augmented views of one segment. Labels are never touched."""

    def __init__(
        self,
        segments: np.ndarray,
        aug_cfg: Optional[AugmentConfig] = None,
        seed: int = CFG.seed,
    ) -> None:
        self.segments = segments
        self.aug_cfg = aug_cfg or CFG.augment
        self._seed = seed

    def __len__(self) -> int:
        return self.segments.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # A per-item RNG keeps augmentation reproducible and worker-safe while
        # still differing every epoch (torch reshuffles the order, and the
        # epoch counter is folded in by `set_epoch`).
        rng = np.random.default_rng((self._seed, i, getattr(self, "_epoch", 0)))
        a, b = make_two_views(self.segments[i], self.aug_cfg, rng)
        return (
            torch.from_numpy(a).float().unsqueeze(0),
            torch.from_numpy(b).float().unsqueeze(0),
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


class SupervisedSegmentDataset(Dataset):
    """Yields (segment, label) with no augmentation."""

    def __init__(self, segments: np.ndarray, labels: np.ndarray) -> None:
        self.segments = segments
        self.labels = labels

    def __len__(self) -> int:
        return self.segments.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.segments[i]).float().unsqueeze(0),
            torch.tensor(int(self.labels[i]), dtype=torch.long),
        )
