"""
Configuration for the Sound-Based Anomaly Detection Pipeline (MIMII DG / DCASE 2022 Task 2).

Conventions align with config.py:
  * Centralized reproducibility (set_seed(42))
  * Hardware-aware device selection ('cuda' vs 'cpu')
  * Windows-safe multiprocessing defaults (num_workers=0)
  * Strong typing via dataclasses
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_AUDIO_DIR = Path(os.environ.get("AUDIO_DATA_DIR", PROJECT_ROOT / "data" / "audio"))
RAW_AUDIO_DIR = DATA_AUDIO_DIR / "raw"
PROCESSED_AUDIO_DIR = DATA_AUDIO_DIR / "processed"

RESULTS_AUDIO_DIR = PROJECT_ROOT / "results" / "audio"
FIGURES_AUDIO_DIR = RESULTS_AUDIO_DIR / "figures"
METRICS_AUDIO_DIR = RESULTS_AUDIO_DIR / "metrics"
EMBEDDINGS_AUDIO_DIR = RESULTS_AUDIO_DIR / "embeddings"
CHECKPOINTS_AUDIO_DIR = PROJECT_ROOT / "checkpoints" / "audio"

for _dir in (
    DATA_AUDIO_DIR,
    RAW_AUDIO_DIR,
    PROCESSED_AUDIO_DIR,
    RESULTS_AUDIO_DIR,
    FIGURES_AUDIO_DIR,
    METRICS_AUDIO_DIR,
    EMBEDDINGS_AUDIO_DIR,
    CHECKPOINTS_AUDIO_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy, and torch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------
# Dataset Configuration
# --------------------------------------------------------------------------
@dataclass
class AudioDatasetConfig:
    """MIMII DG (DCASE 2022 Challenge Task 2 development dataset)."""

    name: str = "mimii_dg"
    zenodo_record: int = 6355122
    zenodo_base_url: str = "https://zenodo.org/api/records/6355122/files"
    sampling_rate: int = 16_000         # Native sampling rate of MIMII DG clips
    clip_duration_s: float = 10.0       # Standard duration of each recording
    num_channels: int = 1               # Single-channel mono audio
    # Supported machines: fan, gearbox, bearing, slider, valve, ToyTrain, ToyCar
    machines: List[str] = field(default_factory=lambda: ["fan", "bearing"])
    target_sections: List[str] = field(default_factory=lambda: ["section_00", "section_01", "section_02"])


# --------------------------------------------------------------------------
# Feature Extraction (Log-Mel Spectrogram)
# --------------------------------------------------------------------------
@dataclass
class AudioFeatureConfig:
    """Log-Mel Spectrogram and context frame stacking parameters."""

    sample_rate: int = 16_000          # Audio sampling rate in Hz
    n_fft: int = 1024                  # FFT window size
    hop_length: int = 512              # Hop length between STFT frames
    win_length: int = 1024             # Window length
    n_mels: int = 128                  # Number of Mel frequency bands
    power: float = 2.0                 # Power for magnitude spectrogram (2.0 = power)
    context_frames: int = 5            # Number of adjacent frames to stack (DCASE baseline)
    frame_hop: int = 1                 # Hop size when sliding context window
    f_min: float = 50.0                # Minimum frequency
    f_max: float = 8000.0              # Maximum frequency (Nyquist for 16 kHz)
    clip_duration_s: float = 10.0      # Expected fixed clip duration in seconds


# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------
@dataclass
class AudioModelConfig:
    """Acoustic anomaly detection model config."""

    in_channels: int = 1
    embedding_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dropout: float = 0.1


# --------------------------------------------------------------------------
# Training & DataLoader Configuration
# --------------------------------------------------------------------------
@dataclass
class AudioTrainConfig:
    """Training configuration. Note: Windows requires num_workers=0."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0               # Must be 0 on Windows PowerShell
    val_fraction: float = 0.15         # Fraction of normal training set used for validation monitoring
    checkpoint_name: str = "audio_anomaly_detector.pt"


# --------------------------------------------------------------------------
# Anomaly Detection & Scoring
# --------------------------------------------------------------------------
@dataclass
class AudioAnomalyConfig:
    """Anomaly scoring config based on normal clips only."""

    method: str = "mahalanobis"        # 'mahalanobis' | 'autoencoder_recon' | 'knn'
    threshold_n_sigma: float = 3.0     # Mean + 3*std of normal validation scores
    pca_components: int = 16           # Optional PCA dimensionality reduction before density estimation


# --------------------------------------------------------------------------
# Top-level Audio Configuration Bundle
# --------------------------------------------------------------------------
@dataclass
class AudioConfig:
    seed: int = SEED
    dataset: AudioDatasetConfig = field(default_factory=AudioDatasetConfig)
    feature: AudioFeatureConfig = field(default_factory=AudioFeatureConfig)
    model: AudioModelConfig = field(default_factory=AudioModelConfig)
    train: AudioTrainConfig = field(default_factory=AudioTrainConfig)
    anomaly: AudioAnomalyConfig = field(default_factory=AudioAnomalyConfig)

    def to_dict(self) -> dict:
        return asdict(self)


AUDIO_CFG = AudioConfig()
