"""Пример обновления ``cards.csv``."""
from __future__ import annotations

from pathlib import Path
import csv


def add_record(path: Path, row: dict[str, str]) -> None:
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)
