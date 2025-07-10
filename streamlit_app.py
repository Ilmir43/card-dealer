from pathlib import Path
import json
import streamlit as st

from card_dealer.cards import CardClasses


MENU_ITEMS = [
    "Настроить количество игроков",
    "Настроить количество карт на игрока",
    "Выбрать режим раздачи",
    "Начать раздачу",
    "Сортировка",
    "Подсчёт карт",
    "Поиск отсутствующих карт",
]

DEAL_MODES = [
    "Все карты сразу",
    "По одной",
]

SORT_OPTIONS = [
    "Без сортировки",
    "По лицу/рубашке",
    "По игре/дополнению",
]


def _normalize_label(label: str) -> str | None:
    """Вернуть корректную запись названия карты или ``None``."""
    cleaned = label.strip()
    for known in CardClasses.LABELS:
        if cleaned.lower() == known.lower():
            return known
    return None


def load_cards(json_path: Path = Path("model.json")) -> list[str]:
    """Загрузить список карт из ``model.json`` или вернуть колоду по умолчанию."""
    if json_path.exists():
        with json_path.open("r") as f:
            mapping = json.load(f)
        if isinstance(mapping, dict):
            items = sorted(mapping.items(), key=lambda x: x[1])
            labels = []
            for name, _ in items:
                norm = _normalize_label(str(name))
                if norm:
                    labels.append(norm)
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
    state.setdefault("leftover_deck", [])
    state.setdefault("sort_mode", 0)
    state.setdefault("deal_mode", 0)
    state.setdefault("unknown_deck_msg", "")
    state.setdefault("deck", load_cards())
    state.setdefault("exclude_select", [])
    state.setdefault("excluded_cards", [])


def deal_cards() -> None:
    """Раздать карты игрокам в указанном порядке."""
    state = st.session_state
    deck = list(state.deck)
    hands = [[] for _ in range(state.players)]
    state.unknown_deck_msg = "Количество карт в колоде заранее неизвестно"

    if state.deal_mode == 0:  # все карты сразу
        for i in range(state.players):
            for _ in range(state.cards_per_player):
                if deck:
                    hands[i].append(deck.pop(0))
    else:  # по одной
        while any(len(h) < state.cards_per_player for h in hands) and deck:
            for h in hands:
                if len(h) < state.cards_per_player and deck:
                    h.append(deck.pop(0))

    state.leftover_deck = deck
    state.deck = deck
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


def render_set_mode() -> None:
    st.write(f"Режим раздачи: {DEAL_MODES[st.session_state.deal_mode]}")


def render_deck_config() -> None:
    """Overlay for arranging the deck order with card icons."""
    with st.expander("Порядок колоды", expanded=False):
        cols = st.columns(2)
        if cols[0].button("Рандом", key="deck_shuffle"):
            import random
            random.shuffle(st.session_state.deck)
        if cols[1].button("Все карты", key="deck_all"):
            current = set(st.session_state.deck)
            for lbl in CardClasses.LABELS:
                if lbl not in current:
                    st.session_state.deck.append(lbl)
                    current.add(lbl)

        deck = st.multiselect(
            "Выберите карты по порядку",
            CardClasses.LABELS,
            default=st.session_state.deck,
            format_func=CardClasses.label_to_icon,
            key="deck_order",
        )
        if deck:
            from card_dealer.deck import deduplicate
            st.session_state.deck = deduplicate(deck)
            icons = [CardClasses.label_to_icon(c) for c in deck]
            st.write("Колода:", " ".join(icons))



def render_deal() -> None:
    st.write(st.session_state.unknown_deck_msg)
    for i, hand in enumerate(st.session_state.distributed_cards, 1):
        icons = [CardClasses.label_to_icon(c) for c in hand]
        st.write(f"Игрок {i}: {' '.join(icons)}")
    if st.session_state.leftover_deck:
        icons = [CardClasses.label_to_icon(c) for c in st.session_state.leftover_deck]
        st.write("Лишние карты:", " ".join(icons))


def render_sort_menu() -> None:
    st.write("### Режим сортировки")
    for idx, item in enumerate(SORT_OPTIONS):
        prefix = "👉 " if idx == st.session_state.menu_index else "  "
        st.write(f"{prefix}{item}")
    st.multiselect(
        "Исключить карты",
        CardClasses.LABELS,
        key="exclude_select",
        format_func=CardClasses.label_to_icon,
    )


def render_sorted() -> None:
    st.write(f"Сортировка: {SORT_OPTIONS[st.session_state.sort_mode]}")
    if st.session_state.excluded_cards:
        icons = [CardClasses.label_to_icon(c) for c in st.session_state.excluded_cards]
        st.write("Исключённые:", " ".join(icons))
    if st.session_state.distributed_cards:
        for i, hand in enumerate(sorted_hands(), 1):
            icons = [CardClasses.label_to_icon(c) for c in hand]
            st.write(f"Игрок {i}: {' '.join(icons)}")
    else:
        icons = [CardClasses.label_to_icon(c) for c in st.session_state.deck]
        st.write("Оставшиеся:", " ".join(icons))


def render_count() -> None:
    from card_dealer.deck import count_cards

    st.write("### Подсчёт карт")
    st.write(f"В колоде {count_cards(st.session_state.deck)} карт")


def render_missing() -> None:
    from card_dealer.deck import find_missing_cards

    st.write("### Отсутствующие карты")
    missing = find_missing_cards(st.session_state.deck)
    if not missing:
        st.write("Все карты на месте")
    else:
        icons = [CardClasses.label_to_icon(c) for c in missing]
        st.write("Не хватает:", " ".join(icons))


def render_screen() -> None:
    screen = st.session_state.screen
    if screen == "main_menu":
        render_main_menu()
    elif screen == "set_players":
        render_set_players()
    elif screen == "set_cards":
        render_set_cards()
    elif screen == "set_mode":
        render_set_mode()
    elif screen == "deal":
        render_deal()
    elif screen == "sort":
        render_sort_menu()
    elif screen == "sorted":
        render_sorted()
    elif screen == "count":
        render_count()
    elif screen == "missing":
        render_missing()


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
    elif state.screen == "set_mode":
        if left:
            state.deal_mode = (state.deal_mode - 1) % len(DEAL_MODES)
        if right:
            state.deal_mode = (state.deal_mode + 1) % len(DEAL_MODES)

    if state.screen == "main_menu" and ok:
        idx = state.menu_index
        state.menu_index = 0
        if idx == 0:
            state.screen = "set_players"
        elif idx == 1:
            state.screen = "set_cards"
        elif idx == 2:
            state.screen = "set_mode"
        elif idx == 3:
            deal_cards()
        elif idx == 4:
            state.menu_index = state.sort_mode
            state.screen = "sort"
        elif idx == 4:
            state.screen = "count"
        elif idx == 5:
            state.screen = "missing"
    elif state.screen == "set_players" and ok:
        state.screen = "main_menu"
    elif state.screen == "set_cards" and ok:
        state.screen = "main_menu"
    elif state.screen == "set_mode" and ok:
        state.screen = "main_menu"
    elif state.screen == "deal" and ok:
        state.screen = "main_menu"
    elif state.screen == "sort" and ok:
        state.sort_mode = state.menu_index
        from card_dealer.deck import exclude_cards
        excluded, remaining = exclude_cards(state.deck, state.exclude_select)
        state.excluded_cards = excluded
        state.deck = remaining
        state.exclude_select = []
        state.screen = "sorted"
    elif state.screen == "sorted" and ok:
        state.screen = "main_menu"
    elif state.screen in {"count", "missing"} and ok:
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
