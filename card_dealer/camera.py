"""Utilities for interacting with a camera device."""

from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - OpenCV might not be installed
    cv2 = None


def capture_image(
    output_path: Path,
    device_index: int = 0,
    api_preference: int | None = None,
) -> Path:
    """Capture a single frame from the specified camera device.

    Parameters
    ----------
    output_path:
        File where the captured image will be stored.

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
) -> "Generator[\"cv2.typing.MatLike\", None, None]":
    """Yield frames from the specified camera as numpy arrays.

    Parameters
    ----------
    device_index:
        Index of the camera device.
    api_preference:
        Optional backend API preference passed to :func:`cv2.VideoCapture`.

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

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            yield frame
    finally:
        cap.release()

