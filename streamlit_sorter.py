"""Интерфейс Streamlit для сортировки изображений карт по играм."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st

from card_dealer import recognizer
from card_dealer import sorter
from card_dealer.cards import CardClasses


st.title("Сортировка карт")

# Выбор карт, которые нужно исключить из обработки
excluded = st.multiselect(
    "Исключить карты",
    CardClasses.LABELS,
    format_func=CardClasses.label_to_icon,
)

uploaded_images = st.file_uploader(
    "Изображения карт", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)
video_file = st.file_uploader("Видео", type=["mp4", "avi", "mov"])

if st.button("Отсортировать"):
    temp_paths: list[Path] = []

    # Сохранение загруженных изображений
    if uploaded_images:
        for file in uploaded_images:
            tmp = Path(tempfile.gettempdir()) / file.name
            with open(tmp, "wb") as f:
                f.write(file.read())
            temp_paths.append(tmp)

    # Разбор видео на кадры
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = Path(tmp_vid.name)
        cap = cv2.VideoCapture(str(video_path))
        idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_path = Path(tempfile.gettempdir()) / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            temp_paths.append(frame_path)
            idx += 1
        cap.release()

    # Отфильтровать исключённые карты
    images_for_sort: list[Path] = []
    for path in temp_paths:
        label = recognizer.recognize_card(path)
        if label not in excluded:
            images_for_sort.append(path)

    groups = sorter.sort_cards(images_for_sort)
    for game, paths in groups.items():
        st.subheader(game)

        unique_labels: list[str] = []
        seen: set[str] = set()
        for p in paths:
            lbl = recognizer.recognize_card(p)
            if lbl in excluded or lbl in seen:
                continue
            unique_labels.append(lbl)
            seen.add(lbl)

        if unique_labels:
            icons = [CardClasses.label_to_icon(l) for l in unique_labels]
            cols = st.columns(len(icons))
            for col, icon in zip(cols, icons):
                col.write(icon)
        else:
            st.write("—")
