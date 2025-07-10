"""Вспомогательные функции."""
from __future__ import annotations

from typing import Iterable


def deduplicate(items: Iterable[str]) -> list[str]:
    """Вернуть список без дубликатов с сохранением порядка."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
