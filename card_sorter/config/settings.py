"""Настройки приложения."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraSettings:
    width: int = 320
    height: int = 180
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    hue: float | None = None
    gain: float | None = None


@dataclass
class ServoSettings:
    """Параметры подключения сервопривода."""

    pwm_pin: int | None = None
    serial_port: str | None = None


@dataclass
class DatasetSettings:
    """Расположение данных и файлов модели."""

    root: Path = Path("datasets")
    embeddings_file: Path = Path("embeddings.pkl")


@dataclass
class AppConfig:
    """Базовая конфигурация приложения."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    servo: ServoSettings = field(default_factory=ServoSettings)
    data: DatasetSettings = field(default_factory=DatasetSettings)
