"""Настройки приложения."""
from dataclasses import dataclass


@dataclass
class CameraSettings:
    width: int = 320
    height: int = 180
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    hue: float | None = None
    gain: float | None = None
