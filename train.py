"""Обучение классификатора карт на базе ResNet18."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import create_model, IMAGE_SIZE


def train_classifier(
    data_dir: str = "datasets/cards",
    epochs: int = 10,
    weights: str = "models/classifier.pt",
    batch_size: int = 32,
) -> None:
    """Обучить классификатор на изображениях из ``data_dir``."""

    root = Path(data_dir)
    train_tf = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tf)

    print(f"Train images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print("Начинается обучение классификатора...")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

    Path(weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    print(f"Сохранение весов в {weights}")


def main() -> None:  # pragma: no cover - скрипт
    import argparse

    parser = argparse.ArgumentParser(description="Обучение классификатора карт")
    parser.add_argument("--data", default="datasets/cards", help="Каталог с изображениями")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weights", default="models/classifier.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_classifier(args.data, args.epochs, args.weights, args.batch_size)

    # Если позже появятся координаты объектов,
    # сюда можно добавить обучение YOLO как было ранее.


if __name__ == "__main__":
    main()
