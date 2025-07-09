import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from model import IMAGE_SIZE


def _load_meta(img_path: Path) -> dict:
    meta_path = img_path.with_suffix(".json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf8") as f:
            return json.load(f)
    parts = img_path.parts
    game = parts[parts.index("datasets") + 1] if "datasets" in parts else parts[-3]
    card_name = img_path.parent.name
    return {"game": game, "card_name": card_name, "type": "face", "tags": []}


def extract_embeddings(data_dir: Path) -> list[dict]:
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
    images = [p for p in data_dir.rglob("*.jpg") if not p.name.endswith("_crop.jpg")]
    for img_path in tqdm(images, desc="images"):
        crop = img_path.with_name(img_path.stem + "_crop" + img_path.suffix)
        use_path = crop if crop.exists() else img_path
        img = Image.open(use_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(tensor).cpu().numpy()[0]
        meta = _load_meta(img_path)
        results.append({
            "path": str(use_path),
            "label": meta.get("card_name", ""),
            "game": meta.get("game"),
            "expansion": meta.get("expansion"),
            "type": meta.get("type"),
            "tags": meta.get("tags", []),
            "embedding": vec,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов карт")
    parser.add_argument("--data", default="datasets", help="Каталог с изображениями")
    parser.add_argument("--output", default="embeddings.pkl", help="Файл для сохранения")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    data_dir = Path(args.data)
    embeddings = extract_embeddings(data_dir)
    with open(args.output, "wb") as f:
        pickle.dump(embeddings, f)
    logging.info("Сохранено %d эмбеддингов в %s", len(embeddings), args.output)


if __name__ == "__main__":
    main()
