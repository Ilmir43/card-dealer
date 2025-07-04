"""Utilities for interacting with a camera device."""

from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - OpenCV might not be installed
    cv2 = None


def capture_image(output_path: Path) -> Path:
    """Capture a single frame from the default camera.

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
    The camera is opened using ``cv2.VideoCapture`` with device index ``0``.
    The frame width and height are set to ``1280`` x ``720`` pixels.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required to capture images")

    cap = cv2.VideoCapture(0)
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

