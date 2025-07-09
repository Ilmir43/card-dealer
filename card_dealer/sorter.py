
"""Автоматическая сортировка карт по играм с использованием эмбеддингов.

Модуль загружает базу эмбеддингов карт и определяет к какой игре относится
каждое изображение. Если похожего изображения в базе не находится, карта
помещается в отдельную стопку ``"unknown"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Sequence, Any, Callable

import pickle
try:  # pragma: no cover - optional dependency
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torchvision import models, transforms
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore

try:  # pragma: no cover - NumPy may be absent
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from model import IMAGE_SIZE
except ModuleNotFoundError:  # pragma: no cover - optional
    IMAGE_SIZE = (224, 224)  # sensible default

# Порог уверенности для распознавания
_DEFAULT_THRESHOLD = 0.6

# Файл с базой эмбеддингов по умолчанию
_DEFAULT_EMBEDDINGS = Path("embeddings.pkl")

# Кэшированные объекты для ускорения повторных вызовов
_model: nn.Module | None = None
_transform: transforms.Compose | None = None
_db_embeddings: np.ndarray | None = None
_db_games: List[str] | None = None


def is_card_back(
    image: Sequence[Sequence[Any]], *, classifier: Callable[[Sequence[Sequence[Any]]], bool]
) -> bool:
    """Определить по внешней модели, является ли изображение рубашкой.

    ``classifier`` должен вернуть ``True``, если карта показана рубашкой, иначе
    ``False``. Внутри функции никаких эвристик на основе пикселей не
    применяется.
    """

    return bool(classifier(image))


def sort_by_back(
    items: Iterable[Tuple[Sequence[Sequence[Any]], bool]],
) -> Dict[str, List[Sequence[Sequence[Any]]]]:
    """Разделить изображения по рубашке и лицевой стороне.

    Функция не определяет ориентацию карт самостоятельно. Она принимает
    от внешней модели признак ``is_back`` и распределяет изображения в
    соответствующие коллекции.
    """

    groups: Dict[str, List[Sequence[Sequence[Any]]]] = {"back": [], "face": []}
    for img, is_back_flag in items:
        if is_back_flag:
            groups["back"].append(img)
        else:
            groups["face"].append(img)
    return groups


def _load_model() -> tuple[nn.Module, transforms.Compose]:
    """Создать и вернуть модель для извлечения эмбеддингов."""
    global _model, _transform
    if _model is None or _transform is None:
        if torch is None or nn is None or models is None or transforms is None:
            raise RuntimeError("PyTorch is required for feature extraction")
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        tf = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        _model = model
        _transform = tf
    return _model, _transform


def _load_embeddings(path: Path) -> tuple[np.ndarray, List[str]]:
    """Загрузить базу эмбеддингов."""
    global _db_embeddings, _db_games
    if _db_embeddings is not None and _db_games is not None:
        return _db_embeddings, _db_games
    with path.open("rb") as f:
        data = pickle.load(f)
    embeddings = []
    games = []
    for row in data:
        embeddings.append(row["embedding"])
        games.append(row.get("game", "unknown"))
    if np is None:
        raise RuntimeError("NumPy is required for embeddings")
    _db_embeddings = np.stack(embeddings).astype(np.float32)
    _db_games = games
    return _db_embeddings, _db_games


def _extract_feature(image_path: Path) -> np.ndarray:
    model, transform = _load_model()
    if Image is None:
        raise RuntimeError("Pillow is required for feature extraction")
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()[0]
    return feat


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for cosine similarity")
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return y_norm @ x_norm


def detect_game(
    image_path: Path,
    *,
    embeddings_path: Path = _DEFAULT_EMBEDDINGS,
    threshold: float = _DEFAULT_THRESHOLD,
) -> str:
    """Определить игру по изображению карты."""
    emb_db, games = _load_embeddings(embeddings_path)
    feature = _extract_feature(image_path)
    sims = _cosine_similarity(feature, emb_db)
    if isinstance(sims, list):
        if not sims:
            return "unknown"
        best_score = max(sims)
        idx = sims.index(best_score)
    else:
        if sims.size == 0:
            return "unknown"
        idx = int(np.argmax(sims))
        best_score = float(sims[idx])
    if best_score < threshold:
        return "unknown"
    return games[idx]


def sort_card_image(image_path: Path, **kwargs) -> str:
    """Вернуть название игры для одной карты."""
    return detect_game(image_path, **kwargs)


def sort_cards(
    images: Iterable[Path],
    *,
    embeddings_path: Path = _DEFAULT_EMBEDDINGS,
    threshold: float = _DEFAULT_THRESHOLD,
) -> Dict[str, List[Path]]:
    """Разбить изображения карт по играм."""
    groups: Dict[str, List[Path]] = {}
    for img in images:
        game = detect_game(img, embeddings_path=embeddings_path, threshold=threshold)
        groups.setdefault(game, []).append(img)
    return groups
