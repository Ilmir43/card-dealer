import json
import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from card_dealer.camera import find_card


def save_image(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main() -> None:
    st.title("Сбор изображений карт")

    game = st.text_input("Игра")
    card_name = st.text_input("Название карты")
    card_type = st.selectbox("Тип", ["face", "back"])
    expansion = st.text_input("Дополнение", "")
    tags = st.text_input("Теги через запятую", "")
    no_crop = st.checkbox("Не обрезать", value=False)

    img_data = st.camera_input("Сделать снимок")
    if img_data is not None and game and card_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("datasets") / game / card_name
        img_path = base_dir / f"{timestamp}.jpg"
        save_image(img_data.getvalue(), img_path)

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
            card = find_card(img)
            if card is not None:
                crop_path = base_dir / f"{timestamp}_crop.jpg"
                cv2.imwrite(str(crop_path), card)

        st.success(f"Сохранено {img_path}")


if __name__ == "__main__":
    main()
