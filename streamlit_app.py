"""Демо-приложение на Streamlit для распознавания карт."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

from detector import CardDetector
from classifier import CardClassifier, load_classifier


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
        tmp = Path("tmp_video.mp4")
        tmp.write_bytes(data)
        cap = cv2.VideoCapture(str(tmp))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes = detector.detect(frame)
            frame = draw_boxes(frame, boxes)
            frames.append(frame)
        cap.release()
        st.video(np.array(frames))
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
        if detected_labels:
            st.write("Обнаруженные карты: " + ", ".join(detected_labels))


if __name__ == "__main__":  # pragma: no cover - скрипт
    main()
