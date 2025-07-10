from pathlib import Path

import pandas as pd
import streamlit as st

from card_dealer.cards import CardClasses


MENU_ITEMS = [
    "Настроить количество игроков",
    "Настроить количество карт на игрока",
    "Начать раздачу",
    "Сортировка",
]

SORT_OPTIONS = [
    "Без сортировки",
    "По лицу/рубашке",
    "По игре/дополнению",
]


def load_cards(csv_path: Path = Path("cards.csv")) -> list[str]:
    """Load card labels from ``cards.csv`` or use default deck."""
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "labels" in df.columns:
            labels = df["labels"].dropna().astype(str).tolist()
        else:
            labels = df.iloc[:, 0].dropna().astype(str).tolist()
        if labels:
            return labels
    return list(CardClasses.LABELS)


def init_state() -> None:
    state = st.session_state
    state.setdefault("screen", "main_menu")
    state.setdefault("menu_index", 0)
    state.setdefault("players", 2)
    state.setdefault("cards_per_player", 1)
    state.setdefault("distributed_cards", [])
    state.setdefault("sort_mode", 0)
    state.setdefault("deck", load_cards())


def deal_cards() -> None:
    """Раздать карты игрокам в указанном порядке."""
    state = st.session_state
    deck = state.deck.copy()
    total = state.players * state.cards_per_player
    if total > len(deck):
        total = len(deck)
    hands = []
    idx = 0
    for _ in range(state.players):
        hand = deck[idx : idx + state.cards_per_player]
        hands.append(hand)
        idx += state.cards_per_player
    state.distributed_cards = hands
    state.screen = "deal"


def sorted_hands() -> list[list[str]]:
    hands = st.session_state.distributed_cards
    mode = st.session_state.sort_mode
    if mode == 0:
        return hands
    if mode in (1, 2):
        return [sorted(h) for h in hands]
    return hands


# ----------- Screen renderers -----------


def render_main_menu() -> None:
    st.write("### Главное меню")
    for idx, item in enumerate(MENU_ITEMS):
        prefix = "👉 " if idx == st.session_state.menu_index else "  "
        st.write(f"{prefix}{item}")


def render_set_players() -> None:
    st.write(f"Количество игроков: {st.session_state.players}")


def render_set_cards() -> None:
    st.write(f"Карт на игрока: {st.session_state.cards_per_player}")


def render_deck_config() -> None:
    """Overlay for arranging the deck order with card icons."""
    with st.expander("Порядок колоды", expanded=False):
        deck = st.multiselect(
            "Выберите карты по порядку",
            CardClasses.LABELS,
            default=st.session_state.deck,
            format_func=CardClasses.label_to_icon,
            key="deck_order",
        )
        if deck:
            st.session_state.deck = list(deck)
            icons = [CardClasses.label_to_icon(c) for c in deck]
            st.write("Колода:", " ".join(icons))



def render_deal() -> None:
    for i, hand in enumerate(st.session_state.distributed_cards, 1):
        icons = [CardClasses.label_to_icon(c) for c in hand]
        st.write(f"Игрок {i}: {' '.join(icons)}")


def render_sort_menu() -> None:
    st.write("### Режим сортировки")
    for idx, item in enumerate(SORT_OPTIONS):
        prefix = "👉 " if idx == st.session_state.menu_index else "  "
        st.write(f"{prefix}{item}")


def render_sorted() -> None:
    st.write(f"Сортировка: {SORT_OPTIONS[st.session_state.sort_mode]}")
    for i, hand in enumerate(sorted_hands(), 1):
        icons = [CardClasses.label_to_icon(c) for c in hand]
        st.write(f"Игрок {i}: {' '.join(icons)}")


def render_screen() -> None:
    screen = st.session_state.screen
    if screen == "main_menu":
        render_main_menu()
    elif screen == "set_players":
        render_set_players()
    elif screen == "set_cards":
        render_set_cards()
    elif screen == "deal":
        render_deal()
    elif screen == "sort":
        render_sort_menu()
    elif screen == "sorted":
        render_sorted()


# ----------- Button handlers -----------

def handle_buttons(up: bool, down: bool, left: bool, right: bool, ok: bool) -> None:
    state = st.session_state

    if state.screen in {"main_menu", "sort"}:
        options = MENU_ITEMS if state.screen == "main_menu" else SORT_OPTIONS
        if up:
            state.menu_index = (state.menu_index - 1) % len(options)
        if down:
            state.menu_index = (state.menu_index + 1) % len(options)

    if state.screen == "set_players":
        if left and state.players > 1:
            state.players -= 1
        if right:
            state.players += 1
    elif state.screen == "set_cards":
        if left and state.cards_per_player > 1:
            state.cards_per_player -= 1
        if right:
            state.cards_per_player += 1

    if state.screen == "main_menu" and ok:
        idx = state.menu_index
        state.menu_index = 0
        if idx == 0:
            state.screen = "set_players"
        elif idx == 1:
            state.screen = "set_cards"
        elif idx == 2:
            deal_cards()
        elif idx == 3:
            state.menu_index = state.sort_mode
            state.screen = "sort"
    elif state.screen == "set_players" and ok:
        state.screen = "main_menu"
    elif state.screen == "set_cards" and ok:
        state.screen = "main_menu"
    elif state.screen == "deal" and ok:
        state.screen = "main_menu"
    elif state.screen == "sort" and ok:
        state.sort_mode = state.menu_index
        state.screen = "sorted"
    elif state.screen == "sorted" and ok:
        state.screen = "main_menu"


# ----------- Main app -----------

def main() -> None:
    st.set_page_config(page_title="Card Device Emulator")
    init_state()

    render_deck_config()

    row_up = st.columns(3)
    up = row_up[1].button("🔼 Вверх", key="btn_up")

    row_mid = st.columns(3)
    left = row_mid[0].button("◀️ Влево", key="btn_left")
    ok = row_mid[1].button("🆗 ОК", key="btn_ok")
    right = row_mid[2].button("▶️ Вправо", key="btn_right")

    row_down = st.columns(3)
    down = row_down[1].button("🔽 Вниз", key="btn_down")

    handle_buttons(up, down, left, right, ok)

    st.markdown("---")
    render_screen()


if __name__ == "__main__":
    main()
