"""
Leak-free dataset splitting.

THE PROBLEM
-----------
Segments are cut with 50 % overlap, so segment k and segment k+1 share half
their samples. If segments were assigned to train/test at random, the test set
would contain near-duplicates of training segments and the reported accuracy
would measure memorisation, not generalisation. Published CWRU results are
frequently inflated for exactly this reason.

THE FIX
-------
Split at **record level**: every segment from a given `.mat` recording goes
entirely into one split. `assert_no_leakage` enforces this and is called by
every split function.

STRATEGIES
----------
record          stratified by class, records shuffled -- the default
load            hold out whole operating conditions (motor loads); the hardest
                and most realistic setting: train on some loads, test on unseen
random_segment  deliberately unsafe; provided only so the report can *show*
                the size of the leakage effect as an ablation

OUTPUT
------
`Split(train_idx, val_idx, test_idx)` -- integer index arrays into the
segment array.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import CFG, SplitConfig


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> Dict[str, int]:
        return {
            "train": int(self.train_idx.size),
            "val": int(self.val_idx.size),
            "test": int(self.test_idx.size),
        }


def assert_no_leakage(record_ids: Sequence[str], split: Split) -> None:
    """Fail loudly if any recording appears in more than one split."""
    rid = np.asarray(record_ids)
    sets = {
        name: set(rid[idx].tolist())
        for name, idx in (
            ("train", split.train_idx),
            ("val", split.val_idx),
            ("test", split.test_idx),
        )
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise AssertionError(f"record leakage between {a} and {b}: {sorted(overlap)}")


def split_by_record(
    record_ids: Sequence[str],
    labels: Sequence[int],
    cfg: Optional[SplitConfig] = None,
    seed: int = CFG.seed,
) -> Split:
    """Stratified record-level split.

    Records are grouped by class so that every split contains every class,
    then shuffled within class and dealt out according to the configured
    fractions. With 4 records per class (CWRU: one per motor load) this gives
    roughly 2 train / 1 val / 1 test records per class.
    """
    cfg = cfg or CFG.split
    rid = np.asarray(record_ids)
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)

    # class -> its records (a record has exactly one class in CWRU)
    records_by_class: Dict[int, List[str]] = defaultdict(list)
    for r in np.unique(rid):
        cls = int(y[rid == r][0])
        records_by_class[cls].append(str(r))

    train_records: List[str] = []
    val_records: List[str] = []
    test_records: List[str] = []
    for _cls, recs in sorted(records_by_class.items()):
        recs = list(recs)
        rng.shuffle(recs)
        n = len(recs)
        n_train = max(1, int(round(cfg.train_fraction * n)))
        n_val = max(1, int(round(cfg.val_fraction * n))) if n - n_train >= 2 else 0
        n_train = min(n_train, n - 1 - n_val) if n - n_val > 1 else n_train
        train_records += recs[:n_train]
        val_records += recs[n_train : n_train + n_val]
        test_records += recs[n_train + n_val :]

    def idx_of(records: List[str]) -> np.ndarray:
        return np.where(np.isin(rid, records))[0]

    split = Split(idx_of(train_records), idx_of(val_records), idx_of(test_records))
    assert_no_leakage(rid, split)
    return split


def split_by_load(
    record_ids: Sequence[str],
    loads: Sequence[int],
    test_loads: Optional[List[int]] = None,
    val_loads: Optional[List[int]] = None,
    seed: int = CFG.seed,
) -> Split:
    """Hold out whole operating conditions (domain-shift evaluation)."""
    rid = np.asarray(record_ids)
    ld = np.asarray(loads)
    all_loads = sorted(set(int(v) for v in ld))
    test_loads = test_loads or [all_loads[-1]]
    val_loads = val_loads or [all_loads[-2]] if len(all_loads) > 2 else []

    test_idx = np.where(np.isin(ld, test_loads))[0]
    val_idx = np.where(np.isin(ld, val_loads))[0]
    train_idx = np.where(~np.isin(ld, list(test_loads) + list(val_loads)))[0]

    split = Split(train_idx, val_idx, test_idx)
    assert_no_leakage(rid, split)
    return split


def split_random_segments(
    n_segments: int,
    cfg: Optional[SplitConfig] = None,
    seed: int = CFG.seed,
) -> Split:
    """UNSAFE random segment split -- for the leakage ablation only."""
    cfg = cfg or CFG.split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_segments)
    n_train = int(cfg.train_fraction * n_segments)
    n_val = int(cfg.val_fraction * n_segments)
    return Split(idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :])


def make_split(
    record_ids: Sequence[str],
    labels: Sequence[int],
    loads: Sequence[int],
    cfg: Optional[SplitConfig] = None,
    seed: int = CFG.seed,
) -> Split:
    """Dispatch on `cfg.strategy`."""
    cfg = cfg or CFG.split
    if cfg.strategy == "record":
        return split_by_record(record_ids, labels, cfg, seed)
    if cfg.strategy == "load":
        return split_by_load(record_ids, loads, cfg.test_loads, seed=seed)
    if cfg.strategy == "random_segment":
        return split_random_segments(len(record_ids), cfg, seed)
    raise ValueError(f"unknown split strategy {cfg.strategy!r}")
