"""Утилиты для работы с колодой карт."""
from __future__ import annotations

from typing import Iterable, Sequence

from ..data.cards import CardClasses


def deduplicate(items: Iterable[str]) -> list[str]:
    """Вернуть список без дубликатов с сохранением порядка."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def exclude_cards(deck: Iterable[str], to_remove: Iterable[str]) -> tuple[list[str], list[str]]:
    """Разделить колоду на исключённые и оставшиеся карты."""
    remove_set = set(to_remove)
    excluded: list[str] = []
    remaining: list[str] = []
    for card in deck:
        if card in remove_set:
            excluded.append(card)
        else:
            remaining.append(card)
    return excluded, remaining


def count_cards(deck: Iterable[str]) -> int:
    """Вернуть количество карт в колоде."""
    return len(list(deck))


def find_missing_cards(
    deck: Iterable[str], *, full_deck: Sequence[str] | None = None
) -> list[str]:
    """Найти недостающие карты относительно полной колоды."""
    full = set(full_deck or CardClasses.LABELS)
    return sorted(full - set(deck))
