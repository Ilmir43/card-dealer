"""Простейший классификатор типа карты."""
from __future__ import annotations

from pathlib import Path


class SimpleTypeClassifier:
    """Заглушка классификатора, возвращающая фиксированное значение."""

    def predict(self, image: Path) -> str:
        return "unknown"
