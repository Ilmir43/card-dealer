"""Базовые интерфейсы для расширения системы."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Any


class CardRecognizer(Protocol):
    """Определяет контракт распознавания одной карты."""

    def recognize(self, image: Any) -> str:
        """Вернуть название карты по изображению."""


class DeviceController(Protocol):
    """Интерфейс управления устройством выдачи карт."""

    def dispense_card(self, angle: float = 90) -> None:
        """Выдать карту под указанным углом."""

    def cleanup(self) -> None:
        """Освободить ресурсы устройства."""


class CardSorter(Protocol):
    """Интерфейс сортировки/классификации изображений карт."""

    def sort_cards(self, images: list[Path]) -> dict[str, list[Path]]:
        """Разбить изображения по группам/играм."""
