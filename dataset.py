import json
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CardImageDataset(Dataset):
    """Dataset that loads card images based on a CSV manifest."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        *,
        transform=None,
        class_to_idx: Optional[dict[str, int]] = None,
    ) -> None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        df = df[df["data set"] == split].reset_index(drop=True)
        self.df = df
        self.transform = transform
        if class_to_idx is None:
            class_to_idx = {
                row["labels"]: int(row["class index"])
                for _, row in df.drop_duplicates("labels").iterrows()
            }
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["filepaths"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["class index"])
        return image, label


def load_labels(labels_file: Path) -> dict[str, int]:
    if labels_file.exists():
        with labels_file.open("r") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict[str, int], labels_file: Path) -> None:
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    with labels_file.open("w") as f:
        json.dump(labels, f)


