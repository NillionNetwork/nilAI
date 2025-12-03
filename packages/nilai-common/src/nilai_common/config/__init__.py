"""Configuration module for nilai-common."""

from .host import HostSettings, SETTINGS, to_bool
from .model import ModelSettings, ModelCapabilities, MODEL_SETTINGS, MODEL_CAPABILITIES

__all__ = [
    "HostSettings",
    "ModelSettings",
    "ModelCapabilities",
    "SETTINGS",
    "MODEL_SETTINGS",
    "MODEL_CAPABILITIES",
    "to_bool",
]
