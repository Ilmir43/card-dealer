"""Convert a classification dataset (cards.csv) to YOLOv8 detection format.
Each image becomes a single bounding box covering the whole card.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import csv


def _map_split(name: str) -> str:
    """Map dataset split names to YOLO conventions."""
    name = name.strip().lower()
    if name in {"valid", "validation"}:
        return "val"
    return name


def _unique_filename(src: Path) -> str:
    """Return a name composed of parent directory and file name."""
    parent = src.parent.name
    return f"{parent}_{src.name}"


def _read_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def convert(csv_path: Path, output_root: Path) -> None:
    """Convert classification CSV to YOLOv8 dataset structure with labels."""
    rows = _read_csv(csv_path)
    if not rows:
        return

    field_names = rows[0].keys()
    if "filepaths" not in field_names or "data set" not in field_names:
        raise ValueError("CSV должен содержать колонки 'filepaths' и 'data set'")

    if "class index" not in field_names:
        if "labels" not in field_names:
            raise ValueError("CSV должен содержать 'class index' или 'labels'")
        labels = sorted({row["labels"] for row in rows})
        mapping = {name: i for i, name in enumerate(labels)}
        for row in rows:
            row["class index"] = str(mapping[row["labels"]])

    splits = {_map_split(row["data set"]) for row in rows if row.get("data set")}

    for split in splits:
        subset = [r for r in rows if _map_split(r.get("data set", "")) == split]
        if not subset:
            continue

        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for row in subset:
            src = Path(row["filepaths"])
            if not src.exists():
                print(f"Файл {src} не найден, пропуск")
                continue

            new_name = _unique_filename(src)
            dst_img = img_dir / new_name
            dst_lbl = lbl_dir / (dst_img.stem + ".txt")

            try:
                shutil.copy(src, dst_img)
            except Exception as e:
                print(f"Ошибка при копировании {src}: {e}")
                continue

            cls = int(row["class index"])
            dst_lbl.write_text(f"{cls} 0.5 0.5 1 1\n")


if __name__ == "__main__":  # pragma: no cover - script usage
    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format")
    parser.add_argument("csv", help="Path to cards.csv")
    parser.add_argument("output", help="Output dataset directory")
    args = parser.parse_args()
    convert(Path(args.csv), Path(args.output))
