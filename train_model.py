from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CardImageDataset
from model import create_model, IMAGE_SIZE
from utils import save_checkpoint


def build_datasets(csv_path: Path) -> tuple[CardImageDataset, CardImageDataset]:
    train_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    train_ds = CardImageDataset(csv_path, "train", transform=train_tf)
    val_ds = CardImageDataset(csv_path, "valid", transform=val_tf, class_to_idx=train_ds.class_to_idx)
    return train_ds, val_ds


def train(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    train_ds, val_ds = build_datasets(csv_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(train_ds.class_to_idx)).to(device)

    if args.resume and Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    save_checkpoint(model, train_ds.class_to_idx, Path(args.model_path))
    print(f"Saved model to {args.model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train card recognition model")
    parser.add_argument("--csv", default="cards.csv", help="Path to CSV manifest")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", default="model.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from existing model")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
