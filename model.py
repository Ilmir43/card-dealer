from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


IMAGE_SIZE = (224, 224)


def create_model(num_classes: int) -> nn.Module:
    """Return a ResNet18 model with custom classification head."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

