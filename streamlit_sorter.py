import tempfile
import time
from pathlib import Path

import streamlit as st

from card_dealer import camera, recognizer
from card_dealer.servo_controller import ServoController


def get_cards() -> list[str]:
    """Вернуть названия карт из каталога ``dataset``."""
    try:
        templates = recognizer._load_templates()
    except Exception:
        return []
    return sorted(templates.keys())


def init_servo(key: str, pin: int) -> ServoController | None:
    """Создать и сохранить сервопривод в ``st.session_state``."""
    if key not in st.session_state:
        try:
            st.session_state[key] = ServoController(pwm_pin=pin)
        except Exception as exc:  # pragma: no cover - может не быть оборудования
            st.warning(f"Серво {pin} недоступно: {exc}")
            st.session_state[key] = None
    return st.session_state[key]


# Основной интерфейс
st.title("Сортировка карт")

available_cards = get_cards()
selected = st.multiselect("Исключить карты", available_cards)

pin_main = st.sidebar.number_input(
    "PIN основного сервопривода", value=11, step=1
)
pin_sort = st.sidebar.number_input(
    "PIN сортирующего сервопривода", value=12, step=1
)
angle_main = st.sidebar.slider(
    "Угол выдачи", 30, 150, 90
)
angle_sort = st.sidebar.slider(
    "Угол отклонения", 0, 180, 30
)

if st.button("Начать сортировку"):
    st.session_state["sorting"] = True
if st.button("Остановить"):
    st.session_state["sorting"] = False

if st.session_state.get("sorting"):
    servo_main = init_servo("servo_main", pin_main)
    servo_sort = init_servo("servo_sort", pin_sort)
    if servo_main and servo_sort:
        img_path = Path(tempfile.gettempdir()) / "current_card.png"
        camera.capture_image(img_path)
        label = recognizer.recognize_card(img_path)
        st.write(f"Распознано: {label}")

        if label in selected:
            servo_sort.dispense_card(angle=angle_sort)
        else:
            servo_sort.dispense_card(angle=0)
        servo_main.dispense_card(angle=angle_main)

        time.sleep(0.1)
        st.experimental_rerun()
