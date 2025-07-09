"""Классы игральных карт и сопутствующие утилиты."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CardClasses:
    """Сведения о картах и вспомогательные методы."""

    # Список названий карт в порядке индексов обучающего датасета
    LABELS = [
        "Ace of Clubs",
        "Ace of Diamonds",
        "Ace of Hearts",
        "Ace of Spades",
        "Eight of Clubs",
        "Eight of Diamonds",
        "Eight of Hearts",
        "Eight of Spades",
        "Five of Clubs",
        "Five of Diamonds",
        "Five of Hearts",
        "Five of Spades",
        "Four of Clubs",
        "Four of Diamonds",
        "Four of Hearts",
        "Four of Spades",
        "Jack of Clubs",
        "Jack of Diamonds",
        "Jack of Hearts",
        "Jack of Spades",
        "Joker",
        "King of Clubs",
        "King of Diamonds",
        "King of Hearts",
        "King of Spades",
        "Nine of Clubs",
        "Nine of Diamonds",
        "Nine of Hearts",
        "Nine of Spades",
        "Queen of Clubs",
        "Queen of Diamonds",
        "Queen of Hearts",
        "Queen of Spades",
        "Seven of Clubs",
        "Seven of Diamonds",
        "Seven of Hearts",
        "Seven of Spades",
        "Six of Clubs",
        "Six of Diamonds",
        "Six of Hearts",
        "Six of Spades",
        "Ten of Clubs",
        "Ten of Diamonds",
        "Ten of Hearts",
        "Ten of Spades",
        "Three of Clubs",
        "Three of Diamonds",
        "Three of Hearts",
        "Three of Spades",
        "Two of Clubs",
        "Two of Diamonds",
        "Two of Hearts",
        "Two of Spades",
    ]

    # Соответствие названия индексу
    LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
    INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

    # Иконки для мастей и сокращения достоинств
    SUIT_ICONS = {
        "Hearts": "♥️",
        "Diamonds": "♦️",
        "Clubs": "♣️",
        "Spades": "♠️",
    }
    RANK_SHORT = {
        "Ace": "A",
        "King": "K",
        "Queen": "Q",
        "Jack": "J",
    }

    @classmethod
    def label_to_icon(cls, label: str) -> str:
        """Преобразовать название карты в вид «A♥️»."""
        parts = label.replace("_", " ").split()
        if "of" in parts:
            rank = parts[0].capitalize()
            suit = parts[-1].capitalize()
            rank_char = cls.RANK_SHORT.get(rank, rank)
            suit_icon = cls.SUIT_ICONS.get(suit, "")
            return f"{rank_char}{suit_icon}"
        return label
