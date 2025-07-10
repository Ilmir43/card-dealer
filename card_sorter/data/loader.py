"""Загрузчик файлов данных."""
from pathlib import Path
import csv


def load_cards_csv(path: Path) -> list[dict[str, str]]:
    """Загрузить ``cards.csv`` как список словарей."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]
