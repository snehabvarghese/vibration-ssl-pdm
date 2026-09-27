"""
Central configuration for the Phase 1 pipeline.

WHY THIS FILE EXISTS
--------------------
Every tunable number in the project lives here, in one place, so that:
  * experiments are reproducible (one seed, one config),
  * nothing dataset-specific is hard-coded deep inside the model code,
  * a Phase 2 live sensor stream only needs a different `SAMPLING_RATE` /
    data source, not a rewrite of the pipeline.

Values can be overridden from the command line by the training scripts, or by
editing this file.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("VIB_DATA_DIR", PROJECT_ROOT / "data"))
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

for _d in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
    CHECKPOINTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python / numpy / torch so that runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional for the pure-preprocessing steps
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'mps' (Mac), else 'cpu'."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# Dataset selection
# --------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Which slice of the public dataset we use.

    CWRU is organised as: sampling rate (12k / 48k) x sensor position
    (drive end / fan end) x fault type x fault diameter x motor load.
    We default to the most widely used benchmark slice: 12 kHz drive-end
    faults plus the normal baseline, all four motor loads.
    """

    name: str = "cwru"
    sampling_rate: int = 12_000           # Hz, native rate of the selected slice
    subsets: List[str] = field(default_factory=lambda: ["normal", "12k_drive_end"])
    loads: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # motor load in HP
    # Which channel of the CWRU .mat file to use. DE = drive end accelerometer.
    channel: str = "DE"
    # Label granularity:
    #   "fault_type"      -> Normal / IR / B / OR                 (4 classes)
    #   "fault_type_size" -> Normal / IR007 / B007 / OR007@6 ...  (10 classes, classic benchmark)
    label_mode: str = "fault_type_size"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """Filtering -> downsampling -> segmentation -> normalization.

    DEVIATIONS FROM THE PAPER, AND WHY
    ----------------------------------
    * Window length. The paper uses 1-3 s windows on continuously monitored
      compressor data. CWRU recordings are only ~10 s long, so 1 s windows
      with 50 % overlap yield just ~19 segments per recording (~1.2 k in
      total) -- far too few for contrastive pretraining. We use 0.25 s
      instead, which still spans ~25-40 fault-impulse periods (BPFO ~107 Hz,
      BPFI ~162 Hz at 1797 rpm), giving ~5 k segments.
    * No downsampling by default. CWRU bearing signatures are impulse
      responses that excite structural resonances in the 2-4 kHz band.
      Decimating to 3 kHz (Nyquist 1.5 kHz) would remove exactly that band,
      so `downsample_factor` defaults to 1. It is kept configurable because a
      Phase 2 edge device may need the cheaper input.
    """

    # Band-pass filter applied to the raw signal (Hz). Set `filter_enabled`
    # False to bypass. high must stay below Nyquist of the *native* rate.
    filter_enabled: bool = True
    filter_type: str = "bandpass"      # "bandpass" | "highpass" | "lowpass"
    filter_low_hz: float = 10.0        # removes DC / mounting drift
    filter_high_hz: float = 5_000.0    # below Nyquist (6 kHz) for 12 kHz data
    filter_order: int = 4

    downsample_factor: int = 1         # 1 = keep the native rate (see docstring)

    window_seconds: float = 0.25       # 1-3 s in the paper; see docstring
    overlap: float = 0.5               # 50 % overlap

    normalization: str = "zscore"      # "zscore" | "minmax" | "none"

    # Segments whose std is below this are treated as dead/corrupt and dropped.
    min_segment_std: float = 1e-8

    def window_size(self, sampling_rate: int) -> int:
        """Window length in samples *after* downsampling."""
        return int(round(self.window_seconds * sampling_rate / self.downsample_factor))

    def hop_size(self, sampling_rate: int) -> int:
        return max(1, int(round(self.window_size(sampling_rate) * (1.0 - self.overlap))))


# --------------------------------------------------------------------------
# Augmentation (used only by the self-supervised stage)
# --------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    jitter_prob: float = 0.8
    jitter_sigma: float = 0.05         # relative to the (unit-variance) signal

    scaling_prob: float = 0.8
    scaling_sigma: float = 0.2

    time_mask_prob: float = 0.5
    time_mask_max_frac: float = 0.15   # fraction of the window that can be zeroed

    permutation_prob: float = 0.3
    permutation_segments: int = 5

    time_shift_prob: float = 0.5
    time_shift_max_frac: float = 0.25  # circular shift up to 25 % of the window


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """1-D CNN encoder + projection head."""

    in_channels: int = 1
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 9, 7, 5])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    pool_size: int = 2
    dropout: float = 0.1

    embedding_dim: int = 128           # paper's practical trade-off
    projection_hidden_dim: int = 256
    projection_dim: int = 64           # dim of the space where NT-Xent is computed


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class SSLTrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 0.1           # NT-Xent temperature
    num_workers: int = 0
    val_fraction: float = 0.1          # of the *pretraining* pool, for loss monitoring
    checkpoint_name: str = "ssl_encoder.pt"


@dataclass
class DownstreamConfig:
    label_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.20])
    classifiers: List[str] = field(default_factory=lambda: ["svm_rbf", "mlp"])
    mlp_hidden: int = 128
    mlp_epochs: int = 100
    mlp_lr: float = 1e-3
    n_repeats: int = 3                 # repeat label sampling to report mean +/- std


@dataclass
class BaselineConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    checkpoint_name: str = "baseline_cnn.pt"


# --------------------------------------------------------------------------
# Anomaly detection (EDR)
# --------------------------------------------------------------------------
@dataclass
class EDRConfig:
    """Entropy Divergence Rate -- see anomaly_detection/edr.py.

    NOTE: this is an approximation inspired by the reference paper, whose full
    mathematical formulation was not available to us.
    """

    estimator: str = "gaussian"        # "gaussian" | "histogram" | "knn"
    window_segments: int = 32          # how many consecutive segments form a "current" distribution
    window_stride: int = 16
    histogram_bins: int = 20
    pca_components: int = 8            # reduce embeddings before density estimation
    knn_k: int = 5
    regularization: float = 1e-6       # added to covariance diagonals
    # Threshold = mean + n_sigma * std of EDR measured on healthy validation data
    threshold_n_sigma: float = 3.0


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """Record-level split to avoid leakage between overlapping windows."""

    strategy: str = "record"           # "record" | "load" | "random_segment" (unsafe, for ablation)
    train_fraction: float = 0.6
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    # If strategy == "load": which motor loads go to test (domain-shift setting)
    test_loads: Optional[List[int]] = None


# --------------------------------------------------------------------------
# Top-level bundle
# --------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = SEED
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssl: SSLTrainConfig = field(default_factory=SSLTrainConfig)
    downstream: DownstreamConfig = field(default_factory=DownstreamConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    edr: EDRConfig = field(default_factory=EDRConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()
