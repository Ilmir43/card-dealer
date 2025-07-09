"""Демо Streamlit для классификации изображений карт."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from predict import recognize_card_array
import tempfile

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


def recognize_cards_in_video(file, model_path: str = "model.pt") -> list[str]:
    """Return labels of cards recognized in a video file."""
    cards: list[str] = []
    if file is None:
        return cards

    # Save uploaded file to a temporary location for cv2.VideoCapture
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return cards

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            result = recognize_card_array(frame, model_path=model_path)
            label = result.get("label", "Unknown")
            if label != "Unknown":
                cards.append(label)
    finally:
        cap.release()

    return cards

def main() -> None:
    st.title("Распознавание игральных карт")
    model_path = st.sidebar.text_input("Файл модели", "model.pt")

    uploaded = st.file_uploader("Изображение карты", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
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

    st.header("Видео")
    video_file = st.file_uploader(
        "Видео с картами", type=["mp4", "avi", "mov"], key="video"
    )
    if video_file is not None:
        labels = recognize_cards_in_video(video_file, model_path=model_path)
        if labels:
            st.success(" ".join(label_to_icon(l) for l in labels))
        else:
            st.info("Карты не распознаны")


if __name__ == "__main__":  # pragma: no cover - скрипт
    main()
