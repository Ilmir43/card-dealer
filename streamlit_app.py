"""Демо Streamlit для классификации изображений карт."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from predict import recognize_card_array

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


def label_to_icon(label: str) -> str:
    """Преобразовать название карты в удобный вид."""
    parts = label.replace("_", " ").split()
    if "of" in parts:
        rank = parts[0]
        suit = parts[-1]
        rank_char = RANK_SHORT.get(rank, rank)
        suit_icon = SUIT_ICONS.get(suit, "")
        return f"{rank_char}{suit_icon}"
    return label


def main() -> None:
    st.title("Распознавание игральных карт")
    model_path = st.sidebar.text_input("Файл модели", "model.pt")

    uploaded = st.file_uploader("Изображение карты", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Не удалось прочитать изображение")
        return

    result = recognize_card_array(img, model_path=model_path)

    st.image(img, channels="BGR")
    label = result.get("label", "Unknown")
    if label != "Unknown":
        st.success(f"Карта: {label_to_icon(label)}")
    else:
        st.info("Карта не распознана")


if __name__ == "__main__":  # pragma: no cover - скрипт
    main()
