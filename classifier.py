"""Классификатор ранга и масти карты."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


class CardClassifier(nn.Module):
    """Небольшая сеть для определения ранга и масти."""

    def __init__(self, num_classes: int = 52) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.model(x)

    def predict(self, image) -> str:
        """Предсказать ранг и масть по изображению."""
        if isinstance(image, (str, Path)):
            from PIL import Image

            image = Image.open(image).convert("RGB")
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        with torch.no_grad():
            output = self.forward(image.unsqueeze(0))
            cls = int(output.argmax())
        return str(cls)


def load_classifier(weights: Path | str) -> CardClassifier:
    """Загрузить веса классификатора."""
    model = CardClassifier()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
