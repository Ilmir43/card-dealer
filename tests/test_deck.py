from card_dealer.deck import (
    deduplicate,
    exclude_cards,
    count_cards,
    find_missing_cards,
)
from card_dealer.cards import CardClasses


def test_deduplicate_preserves_order():
    seq = ["A", "B", "A", "C", "B"]
    assert deduplicate(seq) == ["A", "B", "C"]


def test_exclude_cards():
    deck = ["Ace", "Two", "Three"]
    excl, rest = exclude_cards(deck, ["Two"])
    assert excl == ["Two"]
    assert rest == ["Ace", "Three"]


def test_count_cards():
    deck = ["a", "b", "c"]
    assert count_cards(deck) == 3


def test_find_missing_cards():
    deck = CardClasses.LABELS[:-1]
    missing = find_missing_cards(deck)
    assert missing == [CardClasses.LABELS[-1]]
