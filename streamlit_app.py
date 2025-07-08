"""Демо-приложение на Streamlit для распознавания карт."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

from detector import CardDetector
from classifier import CardClassifier, load_classifier

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
    """Convert card label like 'Ace of Spades' to icon string."""
    parts = label.replace("_", " ").split()
    if "of" in parts:
        rank = parts[0]
        suit = parts[-1]
        rank_char = RANK_SHORT.get(rank, rank)
        suit_icon = SUIT_ICONS.get(suit, "")
        return f"{rank_char}{suit_icon}"
    return label


@st.cache_resource  # pragma: no cover - внешняя функция
def load_models(detector_path: str, classifier_path: str | None = None):
    detector = CardDetector(detector_path)
    classifier = None
    if classifier_path:
        classifier = load_classifier(classifier_path)
    return detector, classifier


def draw_boxes(image: np.ndarray, boxes: List[tuple[int, int, int, int]]) -> np.ndarray:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def main() -> None:
    st.title("Распознавание игральных карт")
    detector_path = st.sidebar.text_input("Файл модели", "models/best.pt")
    classifier_path = st.sidebar.text_input("Файл классификатора", "")
    detector, classifier = load_models(detector_path, classifier_path or None)

    uploaded = st.file_uploader("Изображение или видео", type=["png", "jpg", "jpeg", "mp4"])
    if uploaded is None:
        return
    if uploaded.type.startswith("video"):
        data = uploaded.read()
        tmp = Path("uploaded.mp4")
        tmp.write_bytes(data)
        cap = cv2.VideoCapture(str(tmp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path("output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes = detector.detect(frame)
            if classifier:
                for x1, y1, x2, y2 in boxes:
                    roi = frame[y1:y2, x1:x2]
                    label = classifier.predict(roi)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )
            frame = draw_boxes(frame, boxes)
            writer.write(frame)
        cap.release()
        writer.release()
        st.video(str(out_path))
    else:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        boxes = detector.detect(img)
        detected_labels: List[str] = []
        if classifier:
            for x1, y1, x2, y2 in boxes:
                roi = img[y1:y2, x1:x2]
                label = classifier.predict(roi)
                detected_labels.append(label)
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
        img = draw_boxes(img, boxes)
        st.image(img, channels="BGR")
        if classifier and detected_labels:
            icons = [label_to_icon(lbl) for lbl in detected_labels]
            st.write("Обнаруженные карты: " + " ".join(icons))
        elif not boxes:
            st.info("Карты не обнаружены")
        elif classifier is None:
            st.info(f"Найдено {len(boxes)} карт(ы), классификатор не загружен")
        else:
            st.info("Карты не распознаны")


if __name__ == "__main__":  # pragma: no cover - скрипт
    main()
