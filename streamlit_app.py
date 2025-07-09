"""Демо Streamlit для классификации и сбора изображений карт."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List
import tempfile
import datetime
import json

from predict import recognize_card_array
from recognize_card import load_embeddings, find_best
from card_dealer import recognizer
from card_dealer import camera as cam
from card_dealer.servo_controller import ServoController

import torch
from torch import nn
from torchvision import models, transforms
from model import IMAGE_SIZE

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

# Кэши для эмбеддингов и модели извлечения признаков
_EMBED_CACHE: dict[str, tuple[list[str], list[str], np.ndarray]] = {}
_FEATURE_MODEL: tuple[torch.nn.Module, str] | None = None
_array_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def get_servo(pin: int) -> ServoController | None:
    """Initialize a :class:`ServoController` for the given pin."""
    if "servo" not in st.session_state:
        try:
            st.session_state["servo"] = ServoController(pwm_pin=pin)
        except Exception as exc:  # pragma: no cover - hardware optional
            st.warning(f"Серво недоступно: {exc}")
            st.session_state["servo"] = None
    return st.session_state["servo"]


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
    """Вернуть доступные файлы весов (.pt) во всём проекте."""
    return sorted(p for p in Path(".").rglob("*.pt") if p.is_file())


def list_project_cards() -> list[Path]:
    """Вернуть изображения карт из каталога проекта."""
    ds = recognizer.DATASET_DIR
    if not ds.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in ds.iterdir() if p.suffix.lower() in exts and not p.name.startswith("_"))


def recognize_card_embeddings(image: np.ndarray, embeddings_path: str) -> dict[str, str]:
    """Распознать карту с помощью базы эмбеддингов."""
    global _EMBED_CACHE, _FEATURE_MODEL

    if embeddings_path not in _EMBED_CACHE:
        paths, labels, emb = load_embeddings(Path(embeddings_path))
        _EMBED_CACHE[embeddings_path] = (paths, labels, emb)
    paths, labels, emb_db = _EMBED_CACHE[embeddings_path]

    if _FEATURE_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = models.resnet18(pretrained=True)
        model.fc = nn.Identity()
        model.to(device)
        model.eval()
        _FEATURE_MODEL = (model, device)
    model, device = _FEATURE_MODEL

    tensor = _array_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()[0]

    results = find_best(feat, emb_db, labels, paths, top_k=1)
    label = results[0]["label"] if results else "Unknown"
    return {"type": "face", "label": label}


def recognize_cards_in_video(
    file, *, model_path: str = "model.pt", embeddings_path: str | None = None, method: str = "model"
) -> list[str]:
    """Распознать карты на видео."""
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
            if method == "embeddings" and embeddings_path:
                result = recognize_card_embeddings(frame, embeddings_path)
            else:
                result = recognize_card_array(frame, model_path=model_path)
            label = result.get("label", "Unknown")
            if label != "Unknown":
                cards.append(label)
    finally:
        cap.release()

    return cards

def dataset_page(angle: float, servo_pin: int) -> None:
    """Page for capturing images into the dataset."""

    st.header("Сбор изображений")
    game = st.text_input("Игра")
    card_name = st.text_input("Название карты")
    card_type = st.selectbox("Тип", ["face", "back"])
    expansion = st.text_input("Дополнение", "")
    tags = st.text_input("Теги через запятую", "")
    no_crop = st.checkbox("Не обрезать", value=False)

    if st.button("Следующая карта", key="dispense_dataset"):
        servo = get_servo(servo_pin)
        if servo:
            servo.dispense_card(angle=angle)

    img_data = st.camera_input("Сделать снимок")
    if img_data is not None and game and card_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("datasets") / game / card_name
        img_path = base_dir / f"{timestamp}.jpg"
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(img_data.getvalue())

        meta = {
            "game": game,
            "card_name": card_name,
            "type": card_type,
            "expansion": expansion or None,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
        }
        with open(img_path.with_suffix(".json"), "w", encoding="utf8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if not no_crop:
            arr = np.frombuffer(img_data.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            card = cam.find_card(img)
            if card is not None:
                crop_path = base_dir / f"{timestamp}_crop.jpg"
                cv2.imwrite(str(crop_path), card)

        st.success(f"Сохранено {img_path}")


def recognize_page(angle: float, servo_pin: int) -> None:
    st.header("Распознавание игральных карт")

    method = st.sidebar.radio(
        "Метод распознавания", ["model", "embeddings"],
        format_func=lambda m: "Модель" if m == "model" else "Эмбеддинги",
    )

    model_path = "model.pt"
    embeddings_path = "embeddings.pkl"
    if method == "model":
        models = list_models()
        if models:
            model_path = st.sidebar.selectbox(
                "Классификатор",
                models,
                format_func=lambda p: p.name,
            )
        else:
            model_path = st.sidebar.text_input("Файл модели", "model.pt")
    else:
        embeddings_path = st.sidebar.text_input("Файл embeddings", "embeddings.pkl")

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
        if method == "embeddings":
            result = recognize_card_embeddings(img, embeddings_path)
        else:
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

    if st.button("Следующая карта", key="dispense_rec"):
        servo = get_servo(servo_pin)
        if servo:
            servo.dispense_card(angle=angle)

    st.subheader("Видео")
    video_file = st.file_uploader(
        "Видео с картами", type=["mp4", "avi", "mov"], key="video"
    )
    if video_file is not None:
        labels = recognize_cards_in_video(
            video_file,
            model_path=model_path,
            embeddings_path=embeddings_path,
            method=method,
        )
        if labels:
            st.success(" ".join(label_to_icon(l) for l in labels))
        else:
            st.info("Карты не распознаны")

    if st.checkbox("Распознавать с камеры"):
        placeholder = st.empty()
        label_ph = st.empty()
        for frame in cam.stream_frames():
            if method == "embeddings":
                res = recognize_card_embeddings(frame, embeddings_path)
            else:
                res = recognize_card_array(frame, model_path=str(model_path))
            placeholder.image(frame, channels="BGR")
            label_ph.write(label_to_icon(res.get("label", "Unknown")))


def main() -> None:
    st.title("Card Dealer")
    page = st.sidebar.selectbox("Режим", ["Распознавание", "Сбор датасета"])
    servo_pin = st.sidebar.number_input("Servo PIN", value=11, step=1)
    angle = st.sidebar.slider("Угол сервопривода", 30, 150, 90)

    if page == "Сбор датасета":
        dataset_page(angle, servo_pin)
    else:
        recognize_page(angle, servo_pin)


if __name__ == "__main__":  # pragma: no cover - скрипт
    main()
