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


def _apply_settings(cap: Any, settings: Dict[str, Any] | None) -> None:
    """Apply camera property settings if provided."""
    if not settings:
        return
    for name, value in settings.items():
        prop = _PROP_MAP.get(name)
        if prop is not None:
            cap.set(prop, value)


def capture_image(
    output_path: Path,
    device_index: int = 0,
    api_preference: int | None = None,
    camera_settings: Dict[str, Any] | None = None,
) -> Path:
    """Capture a single frame from the specified camera device.

    Parameters
    ----------
    output_path:
        File where the captured image will be stored.
    camera_settings:
        Optional dictionary mapping property names (``brightness``, ``contrast``,
        ``saturation``, ``hue``, ``gain``) to numeric values.

    Returns
    -------
    Path
        The path to the saved image.

    Notes
    -----
    The camera is opened using :func:`cv2.VideoCapture` with ``device_index``.
    On macOS можно указать ``api_preference=cv2.CAP_AVFOUNDATION`` для работы
    со встроенной камерой. Разрешение кадра устанавливается в ``1280``×``720``
    пикселей.
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _apply_settings(cap, camera_settings)

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
    camera_settings: Dict[str, Any] | None = None,
) -> Generator["cv2.typing.MatLike", None, None]:
    """Yield frames from the specified camera as numpy arrays.

    Parameters
    ----------
    device_index:
        Index of the camera device.
    api_preference:
        Optional backend API preference passed to :func:`cv2.VideoCapture`.
    camera_settings:
        Optional dictionary mapping property names (``brightness``, ``contrast``,
        ``saturation``, ``hue``, ``gain``) to numeric values.

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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _apply_settings(cap, camera_settings)

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
    camera_settings: Dict[str, Any] | None = None,
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
    camera_settings:
        Optional dictionary mapping property names (``brightness``, ``contrast``,
        ``saturation``, ``hue``, ``gain``) to numeric values.

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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _apply_settings(cap, camera_settings)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (1280, 720))
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

