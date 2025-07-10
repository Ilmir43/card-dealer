"""Convert UNO card dataset to project format with cards.csv."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _find_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def convert_uno_dataset(src_dir: Path, game: str, *, output_root: Path = Path("datasets"), val_ratio: float = 0.2) -> Path:
    """Convert UNO images to ``datasets/<game>`` and create ``cards.csv``.

    Parameters
    ----------
    src_dir : Path
        Directory with original UNO card images.
    game : str
        Name of the game used for subdirectory under ``datasets``.
    output_root : Path, optional
        Root directory for datasets, by default ``datasets``.
    val_ratio : float, optional
        Fraction of images per class to use as validation set, by default 0.2.

    Returns
    -------
    Path
        Path to the generated ``cards.csv`` file.
    """
    src_dir = Path(src_dir)
    out_dir = output_root / game
    images = _find_images(src_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {src_dir}")

    labels = sorted({img.stem for img in images})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    rows = []
    random.seed(0)
    for label in labels:
        imgs = [img for img in images if img.stem == label]
        random.shuffle(imgs)
        split = int(len(imgs) * (1 - val_ratio))
        if split == 0:
            split = len(imgs)
        for idx, img in enumerate(imgs):
            subset = "train" if idx < split else "valid"
            dest_dir = out_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / img.name
            shutil.copy(img, dest)
            rows.append({
                "class index": label_to_idx[label],
                "filepaths": str(dest.resolve()),
                "labels": label.replace("_", " "),
                "data set": subset,
            })

    csv_path = out_dir / "cards.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class index", "filepaths", "labels", "data set"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Convert UNO dataset")
    parser.add_argument("src", help="Path to UNO dataset")
    parser.add_argument("--game", default="uno", help="Game name for output directory")
    parser.add_argument("--output-root", default="datasets", help="Root directory for datasets")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    csv_path = convert_uno_dataset(Path(args.src), args.game, output_root=Path(args.output_root), val_ratio=args.val_ratio)
    print(f"Created {csv_path}")


if __name__ == "__main__":  # pragma: no cover - script usage
    main()
