"""Convert a classification dataset (cards.csv) to YOLOv8 detection format.
Each image becomes a single bounding box covering the whole card.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import pandas as pd


def convert(csv_path: Path, output_root: Path) -> None:
    df = pd.read_csv(csv_path)
    for split in {"train", "valid", "test"}:
        subset = df[df["data set"] == split]
        if subset.empty:
            continue
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for _, row in subset.iterrows():
            src = Path(row["filepaths"])
            dst = img_dir / src.name
            if not dst.exists():
                shutil.copy(src, dst)
            label_file = lbl_dir / (src.stem + ".txt")
            label_file.write_text("0 0.5 0.5 1 1\n")


if __name__ == "__main__":  # pragma: no cover - script usage
    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format")
    parser.add_argument("csv", help="Path to cards.csv")
    parser.add_argument("output", help="Output dataset directory")
    args = parser.parse_args()
    convert(Path(args.csv), Path(args.output))
