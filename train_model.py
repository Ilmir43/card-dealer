import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import CardClassifier

IMAGE_SIZE = (100, 150)

class CardDataset(Dataset):
    def __init__(self, entries: List[Tuple[Path, str]], class_to_idx: dict[str, int], transform=None):
        self.entries = entries
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]

def build_entries(dataset_dir: Path) -> Tuple[List[Tuple[Path, str]], dict[str, int]]:
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    entries: List[Tuple[Path, str]] = []
    for p in files:
        stem = p.stem
        base = stem.rsplit('_', 1)[0] if stem.rsplit('_', 1)[-1].isdigit() else stem
        label = base.replace('_', ' ')
        entries.append((p, label))
    classes = sorted({label for _, label in entries})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return entries, class_to_idx

def split_entries(entries: List[Tuple[Path, str]], val_split: float = 0.1):
    import random
    random.shuffle(entries)
    val_size = int(len(entries) * val_split)
    return entries[val_size:], entries[:val_size]

def main() -> None:
    parser = argparse.ArgumentParser(description="Train card recognition model")
    parser.add_argument('--dataset', default='dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-path', default='model.pt')
    parser.add_argument('--resume', action='store_true', help='Resume training from existing model')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    entries, class_to_idx = build_entries(dataset_dir)
    if not entries:
        raise RuntimeError(f'No images found in {dataset_dir}')

    train_entries, val_entries = split_entries(entries)

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

    train_ds = CardDataset(train_entries, class_to_idx, transform=train_tf)
    val_ds = CardDataset(val_entries, class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CardClassifier(num_classes=len(class_to_idx)).to(device)

    if args.resume and Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        state = checkpoint.get('model_state', checkpoint)
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

    checkpoint = {'model_state': model.state_dict(), 'class_to_idx': class_to_idx}
    torch.save(checkpoint, args.model_path)
    labels_path = Path(args.model_path).with_suffix('.json')
    with labels_path.open('w') as f:
        json.dump(class_to_idx, f)
    print(f"Saved model to {args.model_path}")

if __name__ == '__main__':
    main()
