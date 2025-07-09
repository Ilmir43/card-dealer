"""Демо Streamlit для классификации изображений карт."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List

from pathlib import Path
from predict import recognize_card_array
from card_dealer import recognizer
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


def list_models() -> list[Path]:
    """Вернуть доступные файлы весов (.pt) в корне проекта."""
    return sorted(Path("./models").glob("*.pt"))


def list_project_cards() -> list[Path]:
    """Вернуть изображения карт из каталога проекта."""
    ds = recognizer.DATASET_DIR
    if not ds.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in ds.iterdir() if p.suffix.lower() in exts and not p.name.startswith("_"))


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

    models = list_models()
    if models:
        model_path = st.sidebar.selectbox(
            "Классификатор",
            models,
            format_func=lambda p: p.name,
        )
    else:
        model_path = st.sidebar.text_input("Файл модели", "model.pt")

    project_cards = list_project_cards()
    selected_card = None
    if project_cards:
        selected_card = st.sidebar.selectbox(
            "Карта из проекта",
            [None] + project_cards,
            format_func=lambda p: "-" if p is None else p.name,
        )

    uploaded = st.file_uploader("Изображение карты", type=["png", "jpg", "jpeg"])
    img = None
    source_path = None
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Не удалось прочитать изображение")
            return
    elif selected_card is not None:
        source_path = selected_card
        img = cv2.imread(str(selected_card))
        if img is None:
            st.error("Не удалось прочитать изображение")
            return

    if img is not None:
        result = recognize_card_array(img, model_path=str(model_path))

        st.image(img, channels="BGR")
        label = result.get("label", "Unknown")
        if label != "Unknown":
            st.success(f"Карта: {label_to_icon(label)}")
        else:
            st.info("Карта не распознана")

        if st.button("Отметить как неправильное"):
            if source_path is None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    cv2.imwrite(tmp.name, img)
                    source_path = Path(tmp.name)
            recognizer.record_incorrect(Path(source_path), label)
            st.info("Отмечено для переобучения")

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
