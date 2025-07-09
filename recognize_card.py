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


def load_embeddings(paths: List[Path]) -> tuple[List[str], List[str], np.ndarray, List[dict]]:
    all_paths: List[str] = []
    all_labels: List[str] = []
    all_emb: List[np.ndarray] = []
    meta: List[dict] = []
    for p in paths:
        with p.open("rb") as f:
            data = pickle.load(f)
        for row in data:
            all_paths.append(row.get("path", ""))
            all_labels.append(row.get("label", ""))
            all_emb.append(row["embedding"])
            meta.append(row)
    emb = np.stack(all_emb)
    return all_paths, all_labels, emb, meta


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


def find_best(
    embedding: np.ndarray,
    db_embeddings: np.ndarray,
    labels: List[str],
    paths: List[str],
    meta: List[dict],
    top_k: int = 3,
) -> List[dict]:
    sims = cosine_similarity(db_embeddings, embedding)
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        item = {
            "label": labels[i],
            "path": paths[i],
            "score": float(sims[i]),
        }
        item.update({k: meta[i].get(k) for k in ("game", "expansion")})
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Распознавание карты по эмбеддингам")
    parser.add_argument("image", help="Путь к изображению карты")
    parser.add_argument("--embeddings", nargs="+", default=["embeddings.pkl"], help="Файлы с базой эмбеддингов")
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

    emb_paths = [Path(p) for p in args.embeddings]
    logging.info("Загружаю эмбеддинги: %s", ", ".join(str(p) for p in emb_paths))
    paths, labels, emb_db, meta = load_embeddings(emb_paths)

    query_path = Path(args.image)
    logging.info("Извлекаю признаки из %s", query_path)
    query_emb = extract_feature(query_path, model, transform)

    results = find_best(query_emb, emb_db, labels, paths, meta, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        game = r.get("game", "?")
        print(f"{i}. {r['label']} [{game}] - {r['score']:.4f} ({r['path']})")


if __name__ == "__main__":  # pragma: no cover - script usage
    main()
