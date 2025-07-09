"""Функции для сортировки карт по ориентации."""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy optional
    np = None  # type: ignore

from .servo_controller import ServoController


_THRESHOLD = 40


def is_card_back(image: Any) -> bool:
    """Определить, является ли карта рубашкой.

    Функция оценивает разброс яркости по изображению. Если разброс
    меньше порога, предполагается, что на изображении рубашка карты.
    Поддерживаются объекты ``numpy.ndarray`` и вложенные списки,
    содержащие значения пикселей.
    """

    if np is not None and hasattr(image, "shape"):
        arr = image
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return float(arr.max() - arr.min()) < _THRESHOLD

    # Обработка списков без использования numpy
    values = []
    try:
        height = len(image)
        width = len(image[0]) if height else 0
    except Exception:
        return False

    for y in range(height):
        for x in range(width):
            pixel = image[y][x]
            try:
                value = sum(pixel) / len(pixel)
            except TypeError:
                value = pixel
            values.append(value)

    if not values:
        return False
    return max(values) - min(values) < _THRESHOLD


def sort_by_back(
    image: Any,
    servo_main: ServoController,
    servo_sort: ServoController,
    *,
    back_angle: float = 30,
    face_angle: float = 0,
    deal_angle: float = 90,
) -> bool:
    """Отправить карту в нужную стопку в зависимости от ориентации.

    Параметры
    ---------
    image:
        Образ карты в формате ``numpy.ndarray`` или вложенных списков.
    servo_main:
        Контроллер основного сервопривода, выдающего карту.
    servo_sort:
        Контроллер сервопривода, отклоняющего карту в стопку.
    back_angle:
        Угол отклонения для карт рубашкой вверх.
    face_angle:
        Угол отклонения для обычных карт.
    deal_angle:
        Угол срабатывания основного сервопривода.

    Возвращает
    ---------
    bool
        ``True``, если обнаружена рубашка, иначе ``False``.
    """

    back = is_card_back(image)
    angle = back_angle if back else face_angle
    servo_sort.dispense_card(angle=angle)
    servo_main.dispense_card(angle=deal_angle)
    return back
