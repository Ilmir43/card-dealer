from pathlib import Path
import csv

from convert_uno_dataset import convert_uno_dataset


def test_convert_uno_dataset(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # create dummy images
    (src / "Red_1.jpg").write_text("r1")
    (src / "Blue_2.jpg").write_text("b2")

    out_root = tmp_path / "out"
    csv_path = convert_uno_dataset(src, "uno", output_root=out_root, val_ratio=0.5)

    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    labels = {row["labels"] for row in rows}
    assert labels == {"Red 1", "Blue 2"}

    for row in rows:
        img_path = Path(row["filepaths"])
        assert img_path.exists()
        assert img_path.is_file()
        assert img_path.parent.parent == out_root / "uno"


