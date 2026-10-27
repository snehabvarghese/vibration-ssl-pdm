"""
Common model interface and abstract adapter specifications for PdM diagnostic models.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelResult:
    """Standardized result schema across all predictive maintenance models.

    Attributes
    ----------
    model_name : str
        Unique name of the model producing the diagnosis.
    modality : {"vibration", "audio", "motor"}
        Sensor modality.
    status : {"NORMAL", "ANOMALOUS", "UNAVAILABLE"}
        Diagnostic verdict.
    anomaly_score : float
        Scalar anomaly score computed by the model.
    threshold : float
        Decision threshold derived from training/validation commissioning data.
    score_ratio : float
        Ratio of anomaly_score to threshold. A value >= 1.0 indicates an anomaly.
    prediction_label : Optional[str]
        Categorical fault classification (e.g. "Normal", "IR007", "Outer_Race").
    confidence : Optional[float]
        Prediction confidence score in [0.0, 1.0].
    details : Dict[str, Any]
        Model-specific extras (e.g. per-frame scores, probabilities, MCSA metrics).
    input_info : Dict[str, Any]
        Input signal metadata (source_name, sampling_rate, duration_seconds, samples).
    is_replay : bool
        Flag indicating if input was replayed from recorded benchmark data.
    timestamp : str
        ISO 8601 acquisition/processing timestamp.
    error : Optional[str]
        Error description if status is UNAVAILABLE.
    """

    model_name: str
    modality: Literal["vibration", "audio", "motor"]
    status: Literal["NORMAL", "ANOMALOUS", "UNAVAILABLE"]
    anomaly_score: float
    threshold: float
    score_ratio: float
    prediction_label: Optional[str] = None
    confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    input_info: Dict[str, Any] = field(default_factory=dict)
    is_replay: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        d = asdict(self)
        # Helper to convert nested numpy numbers or arrays
        return _make_serializable(d)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        import math
        f_val = float(obj)
        return None if (math.isnan(f_val) or math.isinf(f_val)) else f_val
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


class BaseModelAdapter(abc.ABC):
    """Abstract Base Class for all modality model adapters."""

    def __init__(self, name: str, modality: Literal["vibration", "audio", "motor"]) -> None:
        self.name = name
        self.modality = modality

    @abc.abstractmethod
    def load(self) -> None:
        """Load underlying model weights and parameters into memory."""
        pass

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check whether the model checkpoint and required runtime components exist."""
        pass

    @abc.abstractmethod
    def predict(self, input_data: Any, **kwargs: Any) -> ModelResult:
        """Execute inference and return standardized ModelResult.

        Must handle errors gracefully and never crash if the model is unavailable.
        """
        pass

    @abc.abstractmethod
    def info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the adapter.

        Returns
        -------
        dict
            Must contain at least 'checkpoint_path', 'threshold', and 'training_dataset'.
        """
        pass
