from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch


def save_checkpoint(model: torch.nn.Module, class_to_idx: dict[str, int], out_path: Path) -> None:
    """Save model weights and label mapping."""
    checkpoint = {"model_state": model.state_dict(), "class_to_idx": class_to_idx}
    torch.save(checkpoint, out_path)
    labels_file = out_path.with_suffix(".json")
    with labels_file.open("w") as f:
        json.dump(class_to_idx, f)


def load_checkpoint(model_path: Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict[str, int]]:
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    return checkpoint, class_to_idx


def add_new_card(
    image_path: str | Path,
    label: str,
    card_type: str,
    dataset_split: str = "train",
    *,
    data_root: Path = Path("."),
    csv_file: Path = Path("cards.csv"),
    labels_file: Path = Path("labels.json"),
) -> Path:
    """Copy a new card image and update the dataset manifest."""
    image_path = Path(image_path)
    data_dir = data_root / dataset_split / label
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / image_path.name
    shutil.copy(image_path, dest)

    if labels_file.exists():
        with labels_file.open("r") as f:
            labels = json.load(f)
    else:
        labels = {}
    if label not in labels:
        labels[label] = max(labels.values(), default=-1) + 1
        with labels_file.open("w") as f:
            json.dump(labels, f)
    class_idx = labels[label]

    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["class index", "filepaths", "labels", "card type", "data set"])
    new_row = {
        "class index": class_idx,
        "filepaths": str(dest),
        "labels": label,
        "card type": card_type,
        "data set": dataset_split,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_file, index=False)
    return dest

