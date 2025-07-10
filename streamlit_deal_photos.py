import tempfile
from pathlib import Path

import streamlit as st
from typing import Sequence

from card_dealer.recognizer import recognize_card
from card_dealer.cards import CardClasses


def init_state(players: int) -> None:
    """Initialize persistent session variables."""
    if "players" not in st.session_state or st.session_state.players != players:
        st.session_state.players = players
        st.session_state.hands = [[] for _ in range(players)]
        st.session_state.index = 0
        st.session_state.prev_files = []


def reset_state(files: Sequence, players: int) -> None:
    """Reset saved progress if files or player count changed."""
    names = [getattr(f, "name", str(f)) for f in files]
    if names != st.session_state.get("prev_files"):
        st.session_state.hands = [[] for _ in range(players)]
        st.session_state.index = 0
        st.session_state.prev_files = names


def deal_card(file, players: int) -> None:
    """Recognize card image and assign it to the next player."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
        tmp.write(file.getvalue())
        tmp_path = Path(tmp.name)
    label = recognize_card(tmp_path)
    tmp_path.unlink(missing_ok=True)
    player = st.session_state.index % players
    st.session_state.hands[player].append(label)
    st.session_state.index += 1


def main() -> None:
    st.set_page_config(page_title="Раздача по фотографиям")
    st.title("Раздача карт с распознаванием")

    players = st.number_input("Количество игроков", min_value=1, value=2, step=1)
    uploaded = st.file_uploader(
        "Загрузите фотографии карт по порядку", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    init_state(int(players))
    if uploaded:
        reset_state(uploaded, int(players))
        idx = st.session_state.index
        if idx < len(uploaded):
            file = uploaded[idx]
            st.image(file, caption=f"Карта {idx + 1}")
            if st.button("Распознать и раздать", key=f"deal_{idx}"):
                deal_card(file, int(players))
                st.rerun()
        else:
            st.success("Все карты обработаны")

    for i, hand in enumerate(st.session_state.hands, 1):
        if hand:
            icons = [CardClasses.label_to_icon(lbl) for lbl in hand]
            st.write(f"Игрок {i}: {' '.join(icons)}")
        else:
            st.write(f"Игрок {i}: —")


if __name__ == "__main__":
    main()
