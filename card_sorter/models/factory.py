"""Фабрика моделей и распознавания."""
from __future__ import annotations

from typing import Literal

from .template_matcher import TemplateMatcher
from .type_classifier import SimpleTypeClassifier


def create_recognizer(kind: Literal["template", "classifier"] = "template"):
    """Вернуть реализацию распознавания по типу."""
    if kind == "classifier":
        return SimpleTypeClassifier()
    return TemplateMatcher()
