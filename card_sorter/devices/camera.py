"""Utilities for interacting with a camera device."""

from pathlib import Path
from typing import Dict, Any, Generator

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - OpenCV might not be installed
    cv2 = None


_PROP_MAP: Dict[str, int] = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS if cv2 else 10,
    "contrast": cv2.CAP_PROP_CONTRAST if cv2 else 11,
    "saturation": cv2.CAP_PROP_SATURATION if cv2 else 12,
    "hue": cv2.CAP_PROP_HUE if cv2 else 13,
    "gain": cv2.CAP_PROP_GAIN if cv2 else 14,
}

from card_sorter.config import CameraSettings

# Default camera settings
DEFAULT_SETTINGS = CameraSettings()


def _apply_settings(cap: Any, settings: CameraSettings | None) -> None:
    """Apply camera property settings if provided."""
    if not settings:
        return
    for name, value in settings.__dict__.items():
        if name in {"width", "height"} or value is None:
            continue
        prop = _PROP_MAP.get(name)
        if prop is not None:
            cap.set(prop, value)


def find_card_bounds(frame: Any) -> tuple[int, int, int, int] | None:
    """Найти координаты карты на кадре.

    Возвращает кортеж ``(min_x, min_y, max_x, max_y)`` либо ``None`` если
    подходящая область не найдена. Если доступен OpenCV, используется поиск
    наибольшего контура. В противном случае выполняется простое сканирование
    ярких точек.
    """

    # Попытка использовать OpenCV для более точного определения границ
    if cv2 is not None and hasattr(frame, "shape"):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _ret, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area < frame.shape[0] * frame.shape[1] * 0.95:
                    return x, y, x + w - 1, y + h - 1
        except Exception:
            # если что-то пошло не так, продолжаем со старым методом
            pass

    # Старый алгоритм перебора пикселей
    try:
        height, width = frame.shape[:2]
    except AttributeError:
        height = len(frame)
        width = len(frame[0]) if height > 0 else 0

    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            pixel = frame[y][x]
            try:
                value = sum(pixel) / len(pixel)
            except TypeError:
                value = pixel
            if value > 30:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x < min_x or max_y < min_y:
        return None

    return min_x, min_y, max_x, max_y


def find_card(frame: Any) -> Any | None:
    """Найти изображение карты на кадре.

    Алгоритм пытается найти карту как наибольший яркий контур при помощи
    OpenCV. Если библиотека недоступна, применяется простое пороговое
    выделение. В обоих случаях возвращается фрагмент изображения,
    ограниченный минимальным прямоугольником. Если подходящая область не
    найдена, функция возвращает ``None``.
    """
    bounds = find_card_bounds(frame)
    if bounds is None:
        return None

    min_x, min_y, max_x, max_y = bounds
    try:
        card = frame[min_y : max_y + 1, min_x : max_x + 1]
        if hasattr(card, "copy"):
            card = card.copy()
        return card
    except Exception:
        return [row[min_x : max_x + 1] for row in frame[min_y : max_y + 1]]


def capture_image(
    output_path: Path,
    device_index: int = 0,
    api_preference: int | None = None,
    settings: CameraSettings | None = None,
) -> Path:
    """Capture a single frame from the specified camera device.

    Parameters
    ----------
    output_path:
        File where the captured image will be stored.
    settings:
        ``CameraSettings`` instance describing resolution and additional
        properties such as ``brightness`` or ``contrast``.

    Returns
    -------
    Path
        The path to the saved image.

    Notes
    -----
    The camera is opened using :func:`cv2.VideoCapture` with ``device_index``.
    On macOS можно указать ``api_preference=cv2.CAP_AVFOUNDATION`` для работы
    со встроенной камерой. Разрешение кадра по умолчанию снижено до
    ``320``×``180`` пикселей, что позволяет ускорить работу камеры.
    Для использования камеры iPhone через Continuity Camera выберите индекс
    устройства, соответствующий телефону (обычно ``1``), и при необходимости
    передайте ``api_preference=cv2.CAP_AVFOUNDATION``.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required to capture images")

    if api_preference is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index, api_preference)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device")

    # Configure resolution
    if settings is None:
        settings = DEFAULT_SETTINGS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    _apply_settings(cap, settings)

    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError("Failed to capture image from camera")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def stream_frames(
    device_index: int = 0,
    api_preference: int | None = None,
    settings: CameraSettings | None = None,
) -> Generator["cv2.typing.MatLike", None, None]:
    """Yield frames from the specified camera as numpy arrays.

    Parameters
    ----------
    device_index:
        Index of the camera device.
    api_preference:
        Optional backend API preference passed to :func:`cv2.VideoCapture`.
    settings:
        ``CameraSettings`` instance describing additional camera properties.

    Yields
    ------
    numpy.ndarray
        Subsequent frames captured from the camera. The stream stops when
        reading a frame fails.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required to stream video")

    if api_preference is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index, api_preference)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device")

    if settings is None:
        settings = DEFAULT_SETTINGS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    _apply_settings(cap, settings)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            yield frame
    finally:
        cap.release()


def record_video(
    output_path: Path,
    duration_seconds: float,
    fps: int = 30,
    device_index: int = 0,
    api_preference: int | None = None,
    settings: CameraSettings | None = None,
) -> Path:
    """Record a short video clip from the camera.

    Parameters
    ----------
    output_path:
        Target filename for the recorded video.
    duration_seconds:
        Duration of the clip in seconds.
    fps:
        Frames per second.
    device_index:
        Index of the camera device.
    api_preference:
        Optional backend API preference passed to :func:`cv2.VideoCapture`.
    settings:
        ``CameraSettings`` instance describing additional camera properties.

    Returns
    -------
    Path
        Path to the saved video file.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required to record video")

    if api_preference is None:
        cap = cv2.VideoCapture(device_index)
    else:
        cap = cv2.VideoCapture(device_index, api_preference)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device")

    if settings is None:
        settings = DEFAULT_SETTINGS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    _apply_settings(cap, settings)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (settings.width, settings.height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to open video writer")

    frame_count = max(1, int(duration_seconds * fps))
    try:
        for _ in range(frame_count):
            success, frame = cap.read()
            if not success:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return output_path

