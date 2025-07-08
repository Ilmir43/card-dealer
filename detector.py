"""Модуль детектора карт на основе YOLOv8."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - ultralytics может отсутствовать
    YOLO = None  # type: ignore


class CardDetector:
    """Обертка над моделью YOLOv8 для поиска карт."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or "models/best.pt")
        self.model = None
        self.load_model(self.model_path)

    def load_model(self, path: Path) -> None:
        """Загрузить веса модели."""
        if YOLO is None:
            raise RuntimeError("ultralytics не установлен")
        if not path.exists():
            raise FileNotFoundError(path)
        self.model = YOLO(str(path))
        self.model_path = path

    def detect(self, image) -> List[Tuple[int, int, int, int]]:
        """Выполнить детекцию карт на изображении."""
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        results = self.model(image)
        boxes = []
        for result in results:
            for box in result.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, box[:4])
                boxes.append((x1, y1, x2, y2))
        return boxes

    def available_models(self) -> Iterable[Path]:  # pragma: no cover - простой вызов
        """Вернуть список доступных файлов с весами в папке models."""
        return self.model_path.parent.glob("*.pt")
