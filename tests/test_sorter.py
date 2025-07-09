from pathlib import Path

from card_dealer import sorter


def _fake_cosine(x, y):
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))

    import math

    x_norm = math.sqrt(dot(x, x))
    return [dot(x, v) / (x_norm * math.sqrt(dot(v, v))) for v in y]


def test_detect_game(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_text("img")

    emb_db = [[1.0, 0.0], [0.0, 1.0]]
    games = ["uno", "munchkin"]

    monkeypatch.setattr(sorter, "_load_embeddings", lambda path: (emb_db, games))
    monkeypatch.setattr(sorter, "_extract_feature", lambda p: [1.0, 0.0])
    monkeypatch.setattr(sorter, "_cosine_similarity", _fake_cosine)

    game = sorter.detect_game(img, embeddings_path=Path("db.pkl"), threshold=0.5)
    assert game == "uno"


def test_sort_cards(monkeypatch, tmp_path):
    imgs = [tmp_path / f"img{i}.png" for i in range(3)]
    for p in imgs:
        p.write_text("img")

    features = {
        imgs[0]: [1.0, 0.0],
        imgs[1]: [0.0, 1.0],
        imgs[2]: [0.2, 0.2],
    }

    emb_db = [[1.0, 0.0], [0.0, 1.0]]
    games = ["uno", "munchkin"]

    monkeypatch.setattr(sorter, "_load_embeddings", lambda path: (emb_db, games))
    monkeypatch.setattr(sorter, "_extract_feature", lambda p: features[p])
    monkeypatch.setattr(sorter, "_cosine_similarity", _fake_cosine)

    result = sorter.sort_cards(imgs, embeddings_path=Path("db.pkl"), threshold=0.8)

    assert result == {
        "uno": [imgs[0]],
        "munchkin": [imgs[1]],
        "unknown": [imgs[2]],
    }
