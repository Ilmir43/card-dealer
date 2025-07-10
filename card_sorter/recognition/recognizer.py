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
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any

try:  # pragma: no cover - OpenCV/Numpy may not be installed
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    cv2 = None  # type: ignore
    np = None  # type: ignore

# Directory that stores labelled training images.  The project layout places the
# ``dataset`` folder in the repository root.
from card_sorter.config import DatasetSettings
from card_sorter.core.interfaces import CardRecognizer

DATASET_DIR = DatasetSettings().root

# Cache for loaded templates so that multiple calls do not repeatedly read the
# same files from disk.
_TEMPLATES: Dict[str, List["np.ndarray"]] | None = None

# минимальный коэффициент совпадения для уверенного распознавания
MIN_MATCH_SCORE = 0.2
NO_CARD_LABEL = "Нет карты"


@dataclass
class TemplateRecognizer(CardRecognizer):
    """Простейший распознаватель на основе шаблонов."""

    dataset_dir: Path = DATASET_DIR
    min_match_score: float = MIN_MATCH_SCORE
    _templates: Dict[str, List["np.ndarray"]] | None = field(default=None, init=False, repr=False)

    def _load_templates(self) -> Dict[str, List["np.ndarray"]]:
        global _TEMPLATES
        if self._templates is not None:
            return self._templates
        if _TEMPLATES is not None:
            self._templates = _TEMPLATES
            return self._templates
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("OpenCV is required for card recognition")
        templates: Dict[str, List["np.ndarray"]] = {}
        if self.dataset_dir.exists():
            for img_path in self.dataset_dir.iterdir():
                if not img_path.is_file() or img_path.name.startswith("_"):
                    continue
                stem = img_path.stem
                base = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1].isdigit() else stem
                label = base.replace("_", " ")
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates.setdefault(label, []).append(img)
        self._templates = templates
        _TEMPLATES = templates
        return templates

    # public API ---------------------------------------------------------
    def recognize(self, image: Any) -> str:
        if isinstance(image, (str, Path)):
            return self._recognize_path(Path(image))
        return self._recognize_array(image)

    # internal helpers ---------------------------------------------------
    def _recognize_path(self, image_path: Path) -> str:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("OpenCV is required for card recognition")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        return self._match_templates(image)

    def _recognize_array(self, image: "np.ndarray") -> str:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for card recognition")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._match_templates(gray)

    def _match_templates(self, image: "np.ndarray") -> str:
        templates = self._load_templates()
        if not templates:
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
        if best_score < self.min_match_score:
            return NO_CARD_LABEL
        return best_label

    def save_labeled_image(self, image_path: Path, label: str) -> Path:
        if "/" in label or "\\" in label:
            raise ValueError(f"Invalid label: {label}")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        base_name = label.replace(" ", "_")
        ext = image_path.suffix or ".png"
        dest = self.dataset_dir / f"{base_name}{ext}"
        counter = 1
        while dest.exists():
            dest = self.dataset_dir / f"{base_name}_{counter}{ext}"
            counter += 1
        shutil.copy(image_path, dest)
        self._templates = None
        global _TEMPLATES
        _TEMPLATES = None
        return dest

    def record_verification(self, image_path: Path, predicted: str, actual: str) -> Path:
        dest = self.save_labeled_image(image_path, actual)
        self._log("verify_log.csv", ["file", "predicted", "actual"], [dest.name, predicted, actual])
        return dest

    def record_incorrect(self, image_path: Path, predicted: str) -> Path:
        label = f"Incorrect_{predicted.replace(' ', '_')}"
        dest = self.save_labeled_image(image_path, label)
        self._log("incorrect_log.csv", ["file", "predicted"], [dest.name, predicted])
        return dest

    def _log(self, filename: str, header: List[str], row: List[str]) -> None:
        log_path = self.dataset_dir / filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()
        with log_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)


DEFAULT_RECOGNIZER = TemplateRecognizer()


def _load_templates() -> Dict[str, List["np.ndarray"]]:
    """Загрузить шаблоны через экземпляр :class:`TemplateRecognizer`."""

    return DEFAULT_RECOGNIZER._load_templates()


def recognize_card(image_path: Path) -> str:
    """Распознать карту по пути к изображению."""

    return DEFAULT_RECOGNIZER.recognize(image_path)


def recognize_card_array(image: "np.ndarray") -> str:
    """Распознать карту из массива изображения."""

    return DEFAULT_RECOGNIZER.recognize(image)


def save_labeled_image(image_path: Path, label: str) -> Path:
    """Сохранить изображение с меткой в каталог данных."""

    return DEFAULT_RECOGNIZER.save_labeled_image(image_path, label)


def log_verification(filename: str, predicted: str, actual: str) -> None:
    """Добавить запись о верификации распознавания."""

    DEFAULT_RECOGNIZER._log("verify_log.csv", ["file", "predicted", "actual"], [filename, predicted, actual])


def record_verification(image_path: Path, predicted: str, actual: str) -> Path:
    """Сохранить подтверждённое изображение и логировать результат."""

    return DEFAULT_RECOGNIZER.record_verification(image_path, predicted, actual)


def log_incorrect(filename: str, predicted: str) -> None:
    """Добавить запись о неверном предсказании."""

    DEFAULT_RECOGNIZER._log("incorrect_log.csv", ["file", "predicted"], [filename, predicted])


def record_incorrect(image_path: Path, predicted: str) -> Path:
    """Сохранить изображение с ошибкой и логировать её."""

    return DEFAULT_RECOGNIZER.record_incorrect(image_path, predicted)

