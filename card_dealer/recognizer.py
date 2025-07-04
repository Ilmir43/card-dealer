"""Playing card recognition utilities.

This module provides a very small stub implementation for recognizing the
rank and suit of a playing card from an image.  The goal is to allow tests and
examples to run without requiring a fully trained model.  The implementation is
based on naive template matching using OpenCV and a set of images stored in the
``dataset/`` directory.  It is intentionally simple and serves only as a
placeholder until a real model is trained.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, List

try:  # pragma: no cover - OpenCV/Numpy may not be installed
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore
    np = None  # type: ignore

# Directory that stores labelled training images.  The project layout places the
# ``dataset`` folder in the repository root.
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Cache for loaded templates so that multiple calls do not repeatedly read the
# same files from disk.
_TEMPLATES: Dict[str, List["np.ndarray"]] | None = None


def _load_templates() -> Dict[str, List["np.ndarray"]]:
    """Load image templates from :data:`DATASET_DIR`.

    The images are expected to be named using the pattern ``"<label>.<ext>"``
    where ``<label>`` is the card name (e.g. ``"Ace_of_Spades"``).  When
    multiple images with the same base name exist, e.g. ``"Ace_of_Spades_1"``,
    all of them are loaded and stored under the same label.  The function
    converts each image to grayscale and stores them in a dictionary keyed by
    the label.

    Returns
    -------
    dict
        Mapping of label to a list of grayscale image arrays.
    """

    global _TEMPLATES
    if _TEMPLATES is not None:
        return _TEMPLATES

    if cv2 is None:  # pragma: no cover - OpenCV optional
        raise RuntimeError("OpenCV is required for card recognition")

    templates: Dict[str, List["np.ndarray"]] = {}
    if DATASET_DIR.exists():
        for img_path in DATASET_DIR.iterdir():
            if not img_path.is_file():
                continue
            stem = img_path.stem
            base = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
            label = base.replace("_", " ")
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.setdefault(label, []).append(img)

    _TEMPLATES = templates
    return templates


def recognize_card(image_path: Path) -> str:
    """Recognize a playing card from an image path.

    This very small example uses OpenCV's :func:`matchTemplate` on a collection
    of labelled images stored in :data:`DATASET_DIR`.  The best matching label is
    returned.  If no templates are available, ``"Unknown"`` is returned.

    Parameters
    ----------
    image_path:
        Path to the captured card image.

    Returns
    -------
    str
        Predicted card name such as ``"Ace of Spades"``.  When prediction fails
        the string ``"Unknown"`` is returned.
    """

    if cv2 is None:  # pragma: no cover - OpenCV optional
        raise RuntimeError("OpenCV is required for card recognition")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    templates = _load_templates()
    if not templates:
        # No training data available
        return "Unknown"

    best_label = "Unknown"
    best_score = -1.0
    for label, templ_list in templates.items():
        for templ in templ_list:
            if image.shape[0] < templ.shape[0] or image.shape[1] < templ.shape[1]:
                continue
            result = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_label = label

    return best_label


def save_labeled_image(image_path: Path, label: str) -> Path:
    """Save an image with a label into :data:`DATASET_DIR`.

    The function copies ``image_path`` into :data:`DATASET_DIR` using a file name
    derived from ``label``.  Spaces in the label are replaced with underscores.
    If a file with the generated name already exists, a numeric suffix is
    appended to avoid overwriting existing data.

    Parameters
    ----------
    image_path:
        Path to the image that should be stored.
    label:
        The human readable label (e.g. ``"Ace of Spades"``).

    Returns
    -------
    Path
        The path of the copied file inside the dataset directory.
    """

    image_path = Path(image_path)

    if "/" in label or "\\" in label:
        raise ValueError(f"Invalid label: {label}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    base_name = label.replace(" ", "_")
    ext = image_path.suffix or ".png"
    dest = DATASET_DIR / f"{base_name}{ext}"
    counter = 1
    while dest.exists():
        dest = DATASET_DIR / f"{base_name}_{counter}{ext}"
        counter += 1

    shutil.copy(image_path, dest)
    # Invalidate template cache so the newly saved image becomes available
    # immediately and multiple templates per label are reloaded correctly
    global _TEMPLATES
    _TEMPLATES = None
    return dest

