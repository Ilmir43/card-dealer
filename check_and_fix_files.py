"""Проверка наличия файлов датасета с кешированием."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find_file(name: str, root: Path) -> Path | None:
    for path in root.rglob(name):
        if path.is_file():
            return path
    return None


def ensure_cards_csv(dataset_dir: Path, other_dir: Path) -> Path:
    csv_path = dataset_dir / "cards.csv"
    if csv_path.exists():
        return csv_path
    found = find_file("cards.csv", other_dir)
    if found:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dest = dataset_dir / "cards.csv"
        found.replace(dest)
        return dest
    raise FileNotFoundError("cards.csv not found")


if __name__ == "__main__":  # pragma: no cover - скрипт
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", default="dataset")
    parser.add_argument("other", default="dataset_other")
    args = parser.parse_args()
    path = ensure_cards_csv(Path(args.dataset), Path(args.other))
    print("Используется", path)
