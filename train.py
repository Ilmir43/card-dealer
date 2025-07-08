"""Обучение YOLOv8 на датасете карт."""
from __future__ import annotations

from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


def train(data_cfg: str = "data.yaml", epochs: int = 50, weights: str = "models/best.pt") -> None:
    """Запустить дообучение YOLOv8."""
    if YOLO is None:
        raise RuntimeError("ultralytics не установлен")

    model = YOLO("yolov8n.pt")  # стартовые веса
    model.train(data=data_cfg, epochs=epochs)
    Path(weights).parent.mkdir(parents=True, exist_ok=True)
    model.save(weights)


if __name__ == "__main__":  # pragma: no cover - скрипт
    import argparse

    parser = argparse.ArgumentParser(description="Обучение YOLOv8")
    parser.add_argument("--data", default="data.yaml", help="Путь к data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--weights", default="models/best.pt")
    args = parser.parse_args()
    train(args.data, args.epochs, args.weights)
