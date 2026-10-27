"""
Common Model API and Adapters for Vibration, Audio, and Motor PdM Subsystems.
"""

from core.models_api.audio_adapter import AudioAdapter
from core.models_api.interface import BaseModelAdapter, ModelResult
from core.models_api.motor_adapter import (
    BaseMotorAdapter,
    MotorCombinedAdapter,
    MotorCurrentAdapter,
    MotorTemperatureAdapter,
)
from core.models_api.registry import ModelRegistry, get_registry
from core.models_api.vibration_adapter import VibrationAdapter

__all__ = [
    "BaseModelAdapter",
    "ModelResult",
    "VibrationAdapter",
    "AudioAdapter",
    "BaseMotorAdapter",
    "MotorCombinedAdapter",
    "MotorCurrentAdapter",
    "MotorTemperatureAdapter",
    "ModelRegistry",
    "get_registry",
]
