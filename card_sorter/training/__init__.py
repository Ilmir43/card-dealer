from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from card_sorter.models.base_model import create_model, IMAGE_SIZE
from card_sorter.config import DatasetSettings


@dataclass
class TrainingConfig:
    """Параметры обучения классификатора."""

    data_dir: Path = DatasetSettings().root / "cards"
    epochs: int = 10
    batch_size: int = 32
    weights: Path = Path("models/classifier.pt")


def train_classifier(cfg: TrainingConfig) -> None:
    """Обучить модель классификации карт."""

    train_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(cfg.data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(cfg.data_dir / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(cfg.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    cfg.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.weights)
