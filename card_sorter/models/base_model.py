from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


IMAGE_SIZE = (224, 224)


def create_model(num_classes: int) -> nn.Module:
    """Вернуть модель ResNet18 с расширенной классификационной головой."""
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model

