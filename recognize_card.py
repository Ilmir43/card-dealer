import argparse
import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models

from model import IMAGE_SIZE


def load_embeddings(path: Path) -> tuple[List[str], List[str], np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    paths = [row["path"] for row in data]
    labels = [row["label"] for row in data]
    emb = np.stack([row["embedding"] for row in data])
    return paths, labels, emb


def extract_feature(image_path: Path, model: torch.nn.Module, transform) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()[0]
    return feat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return a_norm @ b_norm


def find_best(embedding: np.ndarray, db_embeddings: np.ndarray, labels: List[str], paths: List[str], top_k: int = 3) -> List[dict]:
    sims = cosine_similarity(db_embeddings, embedding)
    idx = np.argsort(-sims)[:top_k]
    return [
        {"label": labels[i], "path": paths[i], "score": float(sims[i])}
        for i in idx
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание карты по эмбеддингам")
    parser.add_argument("image", help="Путь к изображению карты")
    parser.add_argument("--embeddings", default="embeddings.pkl", help="Файл с базой эмбеддингов")
    parser.add_argument("--top-k", type=int, default=3, help="Количество лучших совпадений")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    emb_path = Path(args.embeddings)
    logging.info("Загружаю эмбеддинги из %s", emb_path)
    paths, labels, emb_db = load_embeddings(emb_path)

    query_path = Path(args.image)
    logging.info("Извлекаю признаки из %s", query_path)
    query_emb = extract_feature(query_path, model, transform)

    results = find_best(query_emb, emb_db, labels, paths, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['label']} - {r['score']:.4f} ({r['path']})")


if __name__ == "__main__":  # pragma: no cover - script usage
    main()
