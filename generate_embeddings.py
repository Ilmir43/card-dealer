import argparse
import logging
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from model import IMAGE_SIZE


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder, возвращающий вместе с изображением путь до файла."""

    def __getitem__(self, index):  # type: ignore[override]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def extract_embeddings(data_dir: Path, batch_size: int = 32) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    results: list[dict] = []
    splits = ["train", "val"]

    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            logging.warning("Пропускаю отсутствующую папку %s", split_dir)
            continue
        ds = ImageFolderWithPaths(split_dir, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        logging.info("Обрабатываю %s (%d изображений)", split, len(ds))
        with torch.no_grad():
            for imgs, labels, paths in loader:
                imgs = imgs.to(device)
                emb = model(imgs).cpu()
                for path, label, vec in zip(paths, labels, emb):
                    results.append({
                        "path": str(path),
                        "label": idx_to_class[label.item()],
                        "embedding": vec.numpy(),
                    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов карт")
    parser.add_argument("--data", default="datasets/cards", help="Каталог с изображениями")
    parser.add_argument("--output", default="embeddings.pkl", help="Файл для сохранения эмбеддингов")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_dir = Path(args.data)
    embeddings = extract_embeddings(data_dir, args.batch_size)
    with open(args.output, "wb") as f:
        pickle.dump(embeddings, f)
    logging.info("Сохранено %d эмбеддингов в %s", len(embeddings), args.output)


if __name__ == "__main__":  # pragma: no cover - script usage
    main()
