"""
Central Model Registry for predictive maintenance model adapters.

Provides lazy instantiation, centralized lookup, diagnostic inspection,
and resilient inference routing across Vibration, Audio, and Motor models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.models_api.audio_adapter import AudioAdapter
from core.models_api.interface import BaseModelAdapter, ModelResult
from core.models_api.motor_adapter import (
    BaseMotorAdapter,
    MotorCombinedAdapter,
    MotorCurrentAdapter,
    MotorTemperatureAdapter,
)
from core.models_api.vibration_adapter import VibrationAdapter


class ModelRegistry:
    """Registry managing all diagnostic model adapters."""

    def __init__(self, auto_register: bool = True) -> None:
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._aliases: Dict[str, str] = {}
        if auto_register:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default model adapters lazily (without loading heavy weights)."""
        # Vibration
        vibration = VibrationAdapter()
        self.register(vibration, aliases=["vibration", "vibration_cwru", "cwru"])

        # Audio (default fan, plus explicit fan & bearing)
        audio_default = AudioAdapter(name="audio_mimii_dg_ae", machine_type="fan")
        self.register(audio_default, aliases=["audio", "audio_fan", "mimii_fan"])

        audio_bearing = AudioAdapter(name="audio_bearing_ae", machine_type="bearing")
        self.register(audio_bearing, aliases=["audio_bearing", "mimii_bearing"])

        # Motor (Combined, Current-only, Temperature-only)
        motor_comb = MotorCombinedAdapter()
        self.register(motor_comb, aliases=["motor", "motor_combined", "paderborn"])

        motor_curr = MotorCurrentAdapter()
        self.register(motor_curr, aliases=["motor_current", "motor_current_only"])

        motor_temp = MotorTemperatureAdapter()
        self.register(motor_temp, aliases=["motor_temperature", "motor_temperature_only"])

    def register(self, adapter: BaseModelAdapter, aliases: Optional[List[str]] = None) -> None:
        """Register a model adapter instance with optional aliases."""
        self._adapters[adapter.name] = adapter
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = adapter.name

    def get_adapter(self, name_or_alias: str) -> Optional[BaseModelAdapter]:
        """Look up an adapter by name or alias."""
        key = name_or_alias.strip()
        if key in self._adapters:
            return self._adapters[key]
        canonical = self._aliases.get(key.lower())
        if canonical and canonical in self._adapters:
            return self._adapters[canonical]
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List metadata for all registered models."""
        return [adapter.info() for adapter in self._adapters.values()]

    def is_available(self, name_or_alias: str) -> bool:
        """Check if a registered model's checkpoint exists."""
        adapter = self.get_adapter(name_or_alias)
        return adapter.is_available() if adapter is not None else False

    def predict(
        self,
        model_name: str,
        input_data: Any,
        **kwargs: Any,
    ) -> ModelResult:
        """Execute prediction using specified model name or alias.

        Returns UNAVAILABLE ModelResult if model is unknown or checkpoint is missing,
        never raising unhandled exceptions.
        """
        adapter = self.get_adapter(model_name)
        if adapter is None:
            return ModelResult(
                model_name=model_name,
                modality="vibration",  # fallback modality
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                error=f"Model '{model_name}' is not registered in ModelRegistry. Available: {list(self._adapters.keys())}",
            )

        try:
            return adapter.predict(input_data, **kwargs)
        except Exception as exc:
            return ModelResult(
                model_name=adapter.name,
                modality=adapter.modality,
                status="UNAVAILABLE",
                anomaly_score=0.0,
                threshold=0.0,
                score_ratio=0.0,
                error=f"Unhandled error during prediction with {adapter.name}: {exc}",
            )


# Global singleton instance (lazily initialized)
_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or initialize global ModelRegistry instance."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ModelRegistry()
    return _GLOBAL_REGISTRY
