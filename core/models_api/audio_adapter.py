"""
Adapter for the Acoustic Anomaly Detection Subsystem (MIMII DG / DCASE 2022 Task 2).

Wraps audio_model/inference.py predict_audio without altering the underlying code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from audio_model.config_audio import CHECKPOINTS_AUDIO_DIR
from audio_model.inference import predict_audio
from core.models_api.interface import BaseModelAdapter, ModelResult


class AudioAdapter(BaseModelAdapter):
    """Acoustic anomaly detection adapter wrapping the Dense Autoencoder baseline."""

    def __init__(
        self,
        name: str = "audio_mimii_dg_ae",
        machine_type: str = "fan",
        checkpoints_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(name=name, modality="audio")
        self.machine_type = machine_type.strip().lower()
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_AUDIO_DIR
        self.training_dataset = f"MIMII DG (DCASE 2022 Task 2) - {self.machine_type.capitalize()}"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoints_dir / f"{self.machine_type}_ae.pt"

    def load(self) -> None:
        """Preload model weights into cache."""
        from audio_model.inference import get_cached_model
        get_cached_model(self.machine_type, checkpoints_dir=self.checkpoints_dir)

    def is_available(self) -> bool:
        """Check whether the machine autoencoder checkpoint exists on disk."""
        return self.checkpoint_path.exists()

    def info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the audio model."""
        threshold_val = 0.0
        if self.is_available():
            try:
                import torch
                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                threshold_val = float(ckpt.get("thresholds", {}).get("p95", 0.0))
            except Exception:
                pass

        return {
            "model_name": self.name,
            "modality": self.modality,
            "machine_type": self.machine_type,
            "is_available": self.is_available(),
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "threshold": threshold_val,
            "threshold_metric": "Reconstruction MSE (95th percentile on normal training clips)",
            "training_dataset": self.training_dataset,
        }

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],
        sampling_rate: Optional[int] = None,
        machine_type: Optional[str] = None,
        threshold_percentile: int = 95,
        source_name: Optional[str] = None,
        is_replay: bool = False,
        **kwargs: Any,
    ) -> ModelResult:
        """Score audio clip and return standardized ModelResult.

        Parameters
        ----------
        input_data : Union[str, Path, np.ndarray]
            Path to .wav file or 1-D numpy array.
        sampling_rate : Optional[int]
            Sampling rate of array in Hz (None detects from file).
        machine_type : Optional[str]
            Overrides default machine category (e.g. "fan", "bearing").
        threshold_percentile : int
            Training normal percentile threshold (default: 95).
        source_name : Optional[str]
            Label identifying input source.
        is_replay : bool
            True if clip was replayed from dataset.
        """
        mt = (machine_type or self.machine_type).strip().lower()

        # Check availability
        target_ckpt = self.checkpoints_dir / f"{mt}_ae.pt"
        if not target_ckpt.exists():
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                is_replay=is_replay,
                input_info={"source_name": source_name or "unknown", "machine_type": mt},
                error=f"Acoustic checkpoint missing for machine '{mt}': {target_ckpt}",
            )

        # Resolve source name
        if source_name is None:
            if isinstance(input_data, (str, Path)):
                source_name = Path(input_data).name
            else:
                source_name = "audio_numpy_array"

        try:
            res = predict_audio(
                audio=input_data,
                sampling_rate=sampling_rate,
                machine_type=mt,
                threshold_percentile=threshold_percentile,
            )
        except Exception as exc:
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                is_replay=is_replay,
                input_info={"source_name": source_name, "machine_type": mt},
                error=f"Acoustic inference error: {exc}",
            )

        anomaly_score = float(res["anomaly_score"])
        threshold = float(res["threshold"])
        score_ratio = (anomaly_score / threshold) if threshold > 0 else (1.0 if anomaly_score > 0 else 0.0)

        is_anom = bool(res.get("is_anomaly", False)) or (score_ratio >= 1.0)
        if is_anom and score_ratio < 1.0:
            score_ratio = 1.0
        status = "ANOMALOUS" if is_anom else "NORMAL"
        label = "Anomaly" if is_anom else "Normal"

        # Confidence: distance from decision threshold
        if status == "NORMAL":
            confidence = max(0.5, 1.0 - (score_ratio * 0.5))
        else:
            confidence = min(1.0, 0.5 + min(0.5, (score_ratio - 1.0) * 0.5))

        n_frames = len(res.get("frame_scores", []))
        duration_s = round(n_frames * 512 / 16000, 2)  # approximate duration from hop frames

        details = {
            "machine_type": res["machine_type"],
            "threshold_percentile": threshold_percentile,
            "evaluated_frames": n_frames,
            "frame_scores": res.get("frame_scores", []),
        }

        return ModelResult(
            model_name=self.name,
            modality=self.modality,
            status=status,
            anomaly_score=round(anomaly_score, 5),
            threshold=round(threshold, 5),
            score_ratio=round(score_ratio, 4),
            prediction_label=label,
            confidence=round(confidence, 4),
            details=details,
            input_info={
                "source_name": source_name,
                "sampling_rate": res.get("sampling_rate_used", 16000),
                "duration_seconds": duration_s,
                "frames": n_frames,
            },
            is_replay=is_replay,
            error=None,
        )
