"""Обучение классификатора карт на базе ResNet18 с расширенным логированием."""
from __future__ import annotations

from pathlib import Path
import time

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
    start_time = time.time()
    root = Path(data_dir)

    print(f"[INFO] Используется устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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

    print(f"[INFO] Загружаем датасеты из: {root}")
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tf)

    print(f"[INFO] Классы: {train_ds.classes}")
    print(f"[INFO] Количество классов: {len(train_ds.classes)}")
    print(f"[INFO] Кол-во изображений (train): {len(train_ds)}")
    print(f"[INFO] Кол-во изображений (val):   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"[INFO] Обучение начинается... Epochs: {epochs} | Batch size: {batch_size}")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"[Train][Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        elapsed = time.time() - epoch_start

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f} sec")

    Path(weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    print(f"[INFO] Обучение завершено. Веса сохранены в {weights}")
    print(f"[INFO] Общее время: {(time.time() - start_time):.2f} сек")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Обучение классификатора карт")
    parser.add_argument("--data", default="datasets/cards", help="Каталог с изображениями")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weights", default="models/classifier.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_classifier(args.data, args.epochs, args.weights, args.batch_size)


if __name__ == "__main__":
    main()
