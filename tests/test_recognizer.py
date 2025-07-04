import pytest
from pathlib import Path

import card_dealer.recognizer as recognizer


def test_save_labeled_image_rejects_bad_label(tmp_path, monkeypatch):
    monkeypatch.setattr(recognizer, "DATASET_DIR", tmp_path / "dataset")
    img = tmp_path / "img.png"
    img.write_text("img")

    with pytest.raises(ValueError):
        recognizer.save_labeled_image(img, "../evil")

    assert not (tmp_path / "evil.png").exists()
    assert not (tmp_path / "dataset").exists()
