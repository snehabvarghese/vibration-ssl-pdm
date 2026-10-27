"""
Adapters for Motor Current and Temperature Predictive Maintenance Models.

Wraps teammate's models in core/motor_temperature.py without modifying teammate files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from core.models_api.interface import BaseModelAdapter, ModelResult
from core.motor_temperature import CHECKPOINTS_DIR, analyse_motor_signal, get_motor_model


class BaseMotorAdapter(BaseModelAdapter):
    """Base adapter class for Paderborn motor models."""

    def __init__(
        self,
        name: str,
        model_mode: str,  # "current_only", "temperature_only", "combined"
        sub_type: str,    # "current", "temperature", "combined"
    ) -> None:
        super().__init__(name=name, modality="motor")
        self.model_mode = model_mode
        self.sub_type = sub_type
        self.training_dataset = "Paderborn Bearing Dataset (Drive System Current & Temperature)"

    @property
    def checkpoint_path(self) -> Path:
        return CHECKPOINTS_DIR / f"{self.model_mode}_model.joblib"

    def load(self) -> None:
        """Preload the motor model into cache."""
        get_motor_model(model_mode=self.model_mode)

    def is_available(self) -> bool:
        """Check whether the model checkpoint exists."""
        return self.checkpoint_path.exists()

    def info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the motor model."""
        return {
            "model_name": self.name,
            "modality": self.modality,
            "sub_type": self.sub_type,
            "is_available": self.is_available(),
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "threshold": 0.5,
            "threshold_metric": "Calibrated Isolation Forest sigmoid score in [0.0, 1.0]",
            "training_dataset": self.training_dataset,
        }

    def predict(
        self,
        input_data: Union[np.ndarray, Dict[str, Any]],
        current_phase2: Optional[np.ndarray] = None,
        temperature: Optional[Union[float, np.ndarray]] = None,
        sampling_rate: int = 64_000,
        source_name: Optional[str] = None,
        is_replay: bool = False,
        **kwargs: Any,
    ) -> ModelResult:
        """Execute motor signal inference and return standardized ModelResult.

        Parameters
        ----------
        input_data : Union[np.ndarray, Dict[str, Any]]
            1-D phase 1 current array, or dictionary with current/temperature inputs.
        current_phase2 : Optional[np.ndarray]
            Phase 2 current array.
        temperature : Optional[float or np.ndarray]
            Bearing housing temperature in Celsius (defaults to 45.0 if None).
        sampling_rate : int
            Sampling frequency in Hz (default: 64,000 Hz).
        source_name : Optional[str]
            Source identifier.
        is_replay : bool
            True if signal is replayed from benchmark dataset.
        """
        # 1. Check availability
        if not self.is_available():
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.5,
                score_ratio=0.0,
                details={"sub_type": self.sub_type},
                input_info={"source_name": source_name or "unknown", "sampling_rate": sampling_rate},
                is_replay=is_replay,
                error=f"Checkpoint not found for motor model mode '{self.model_mode}': {self.checkpoint_path}",
            )

        # 2. Extract inputs
        p1 = None
        p2 = current_phase2
        temp_val = temperature

        if isinstance(input_data, dict):
            p1 = input_data.get("current_phase1", input_data.get("current", None))
            p2 = input_data.get("current_phase2", p2)
            temp_val = input_data.get("temperature", temp_val)
            sampling_rate = input_data.get("sampling_rate", sampling_rate)
            source_name = source_name or input_data.get("source_name", "motor_data_dict")
        elif isinstance(input_data, (int, float, np.floating, np.integer)):
            temp_val = float(input_data)
            source_name = source_name or "temperature_scalar"
        elif isinstance(input_data, np.ndarray):
            p1 = input_data
            source_name = source_name or "current_phase1_array"

        if p1 is None or len(p1) == 0:
            if self.model_mode == "temperature_only" and temp_val is not None:
                p1 = np.zeros(1000, dtype=np.float64)
            else:
                return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.5,
                score_ratio=0.0,
                details={"sub_type": self.sub_type},
                input_info={"source_name": source_name or "unknown", "sampling_rate": sampling_rate},
                is_replay=is_replay,
                error="Empty or missing motor current array.",
            )

        n_samples = len(p1)
        duration_s = round(n_samples / max(1, sampling_rate), 4)

        # 3. Call teammate core inference
        try:
            res = analyse_motor_signal(
                current_phase1=p1,
                current_phase2=p2,
                temperature=temp_val,
                sampling_rate=sampling_rate,
                model_mode=self.model_mode,
            )
        except Exception as exc:
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.5,
                score_ratio=0.0,
                details={"sub_type": self.sub_type},
                input_info={
                    "source_name": source_name,
                    "sampling_rate": sampling_rate,
                    "duration_seconds": duration_s,
                    "samples": n_samples,
                },
                is_replay=is_replay,
                error=f"Motor analysis error: {exc}",
            )

        # 4. Standardize scores
        anomaly_score = float(res.get("anomaly_score", 0.0))
        threshold = 0.5
        score_ratio = round(anomaly_score / threshold, 4)

        pred_condition = str(res.get("predicted_condition", "Normal"))
        is_anom = (res.get("status") == "ANOMALOUS") or (score_ratio >= 1.0) or (pred_condition != "Normal")

        # Ensure score_ratio >= 1.0 when anomalous
        if is_anom and score_ratio < 1.0:
            score_ratio = 1.0

        status = "ANOMALOUS" if is_anom else "NORMAL"
        confidence = float(res.get("confidence", 0.0))

        details = {
            "sub_type": self.sub_type,
            "model_mode": self.model_mode,
            "features": res.get("features", {}),
            "class_probabilities": res.get("class_probabilities", {}),
            "input_diagnostics": res.get("input_diagnostics", {}),
        }

        return ModelResult(
            model_name=self.name,
            modality=self.modality,
            status=status,
            anomaly_score=round(anomaly_score, 4),
            threshold=threshold,
            score_ratio=score_ratio,
            prediction_label=pred_condition,
            confidence=round(confidence, 4),
            details=details,
            input_info={
                "source_name": source_name,
                "sampling_rate": sampling_rate,
                "duration_seconds": duration_s,
                "samples": n_samples,
                "temperature_deg_c": temp_val if temp_val is not None else 45.0,
            },
            is_replay=is_replay,
            error=None,
        )


class MotorCurrentAdapter(BaseMotorAdapter):
    """Adapter for Current-Only MCSA model."""

    def __init__(self, name: str = "motor_current_only") -> None:
        super().__init__(name=name, model_mode="current_only", sub_type="current")


class MotorTemperatureAdapter(BaseMotorAdapter):
    """Adapter for Temperature-Only thermal model."""

    def __init__(self, name: str = "motor_temperature_only") -> None:
        super().__init__(name=name, model_mode="temperature_only", sub_type="temperature")


class MotorCombinedAdapter(BaseMotorAdapter):
    """Adapter for Combined Current-and-Temperature model."""

    def __init__(self, name: str = "motor_current_temperature_combined") -> None:
        super().__init__(name=name, model_mode="combined", sub_type="combined")
