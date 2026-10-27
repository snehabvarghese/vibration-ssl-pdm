"""
Adapter for the Vibration SSL Anomaly Detection and Fault Classification Subsystem.

Wraps core/analysis.py analyse_signal without altering the underlying code.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from config import CFG, CHECKPOINTS_DIR
from core.analysis import analyse_signal, get_monitor
from core.models_api.interface import BaseModelAdapter, ModelResult


class VibrationAdapter(BaseModelAdapter):
    """Vibration diagnostic adapter wrapping the CWRU SSL encoder and HealthMonitor."""

    def __init__(
        self,
        name: str = "vibration_cwru_ssl",
        encoder_path: Optional[Path] = None,
        monitor_path: Optional[Path] = None,
    ) -> None:
        super().__init__(name=name, modality="vibration")
        self.encoder_path = Path(encoder_path) if encoder_path is not None else (CHECKPOINTS_DIR / "ssl_encoder.pt")
        self.monitor_path = Path(monitor_path) if monitor_path is not None else (CHECKPOINTS_DIR / "health_monitor.joblib")
        self.training_dataset = "CWRU Bearing Dataset (12 kHz Drive End)"

    def load(self) -> None:
        """Load encoder and monitor into cached memory."""
        from core.analysis import get_encoder, get_monitor
        get_encoder()
        get_monitor()

    def is_available(self) -> bool:
        """Check if both the encoder and health monitor checkpoints exist."""
        return self.encoder_path.exists() and self.monitor_path.exists()

    def info(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the vibration model."""
        threshold_val = 0.0
        if self.monitor_path.exists():
            try:
                mon = get_monitor()
                if mon is not None and hasattr(mon, "edr") and hasattr(mon.edr, "threshold"):
                    threshold_val = float(mon.edr.threshold)
            except Exception:
                pass

        return {
            "model_name": self.name,
            "modality": self.modality,
            "is_available": self.is_available(),
            "checkpoint_path": str(self.encoder_path.resolve()),
            "monitor_path": str(self.monitor_path.resolve()),
            "threshold": threshold_val,
            "threshold_metric": "EDR divergence (nats/s, healthy mean + 3*sigma)",
            "training_dataset": self.training_dataset,
        }

    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],
        sampling_rate: int = 12_000,
        clf_name: str = "svm_rbf",
        source_name: Optional[str] = None,
        is_replay: bool = False,
        **kwargs: Any,
    ) -> ModelResult:
        """Run vibration signal inference and return standardized ModelResult.

        Parameters
        ----------
        input_data : Union[str, Path, np.ndarray]
            Raw vibration acceleration array or file path (.mat, .npy, .csv).
        sampling_rate : int
            Sampling frequency in Hz (default: 12,000 Hz).
        clf_name : str
            Downstream classifier (e.g. "svm_rbf").
        source_name : Optional[str]
            Identifier of signal origin.
        is_replay : bool
            True if input is from a pre-recorded dataset replay.
        """
        # 1. Check availability
        if not self.is_available():
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                is_replay=is_replay,
                input_info={"source_name": source_name or "unknown", "sampling_rate": sampling_rate},
                error=f"Vibration checkpoints missing: {self.encoder_path} or {self.monitor_path}",
            )

        # 2. Extract signal array
        try:
            sig, resolved_name = self._parse_input(input_data)
        except Exception as exc:
            return ModelResult(
                model_name=self.name,
                modality=self.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                is_replay=is_replay,
                input_info={"source_name": source_name or "unknown", "sampling_rate": sampling_rate},
                error=f"Failed to read input signal: {exc}",
            )

        src_name = source_name or resolved_name
        n_samples = len(sig)
        duration_s = round(n_samples / max(1, sampling_rate), 4)

        # 3. Execute analysis
        try:
            res = analyse_signal(sig, sampling_rate=sampling_rate, clf_name=clf_name)
            if res is None:
                return ModelResult(
                    model_name=self.name,
                    modality=self.modality,
                    status="UNAVAILABLE",
                    anomaly_score=0.0,
                    threshold=0.0,
                    score_ratio=0.0,
                    is_replay=is_replay,
                    input_info={
                        "source_name": src_name,
                        "sampling_rate": sampling_rate,
                        "duration_seconds": duration_s,
                        "samples": n_samples,
                    },
                    error=f"Signal too short for windowing (requires >= {CFG.preprocess.window_seconds}s).",
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
                input_info={
                    "source_name": src_name,
                    "sampling_rate": sampling_rate,
                    "duration_seconds": duration_s,
                    "samples": n_samples,
                },
                error=f"Inference execution failed: {exc}",
            )

        # 4. Standardize scores
        anomaly_score = float(res.get("edr_mean", 0.0))
        threshold = float(res.get("edr_threshold", 1.0))
        score_ratio = (anomaly_score / threshold) if threshold > 0 else (1.0 if anomaly_score > 0 else 0.0)

        is_anom = bool(res.get("anomalous", False)) or (score_ratio >= 1.0)
        if is_anom and score_ratio < 1.0:
            score_ratio = 1.0
        status = "ANOMALOUS" if is_anom else "NORMAL"

        prediction_label = str(res.get("predicted_class", "Normal"))
        confidence = float(res.get("confidence", 0.0))

        details = {
            "n_segments": res.get("n_segments", 0),
            "class_probabilities": res.get("class_probabilities", {}),
            "edr_mean": anomaly_score,
            "edr_threshold": threshold,
            "segment_threshold": float(res.get("segment_threshold", 0.0)),
            "fraction_segments_flagged": float(res.get("fraction_segments_flagged", 0.0)),
        }

        return ModelResult(
            model_name=self.name,
            modality=self.modality,
            status=status,
            anomaly_score=round(anomaly_score, 5),
            threshold=round(threshold, 5),
            score_ratio=round(score_ratio, 4),
            prediction_label=prediction_label,
            confidence=round(confidence, 4),
            details=details,
            input_info={
                "source_name": src_name,
                "sampling_rate": sampling_rate,
                "duration_seconds": duration_s,
                "samples": n_samples,
            },
            is_replay=is_replay,
            error=None,
        )

    def _parse_input(self, input_data: Union[str, Path, np.ndarray]) -> tuple[np.ndarray, str]:
        """Convert input data to a 1-D numpy array and source label."""
        if isinstance(input_data, (str, Path)):
            p = Path(input_data)
            if not p.is_file():
                raise FileNotFoundError(f"File not found: {p}")
            ext = p.suffix.lower()
            if ext == ".npy":
                arr = np.load(p)
                return np.asarray(arr, dtype=float).ravel(), p.name
            elif ext == ".mat":
                import scipy.io as sio
                mat = sio.loadmat(p)
                de_keys = [k for k in mat if isinstance(k, str) and k.endswith("_DE_time")]
                key = de_keys[0] if de_keys else [k for k in mat if not k.startswith("__")][0]
                return np.asarray(mat[key], dtype=float).ravel(), p.name
            elif ext == ".csv":
                import pandas as pd
                df = pd.read_csv(p)
                col = df.iloc[:, 0]
                return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float).ravel(), p.name
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        elif isinstance(input_data, np.ndarray):
            return np.asarray(input_data, dtype=float).ravel(), "numpy_array"
        else:
            raise TypeError(f"Expected np.ndarray or file path, got {type(input_data)}")
