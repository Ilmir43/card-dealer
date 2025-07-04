from pathlib import Path

import card_dealer.recognizer as recognizer

class DummyImage:
    def __init__(self, ident):
        self.ident = ident
        self.shape = (1, 1)

class DummyCV2:
    IMREAD_GRAYSCALE = 0
    TM_CCOEFF_NORMED = 1

    def imread(self, path, flag):
        name = Path(path).name
        if name == "target.png":
            return DummyImage("target")
        elif name.startswith("Ace_of_Hearts"):
            # two template files
            if name.endswith("_1.png"):
                return DummyImage("templ2")
            return DummyImage("templ1")
        return None

    def matchTemplate(self, image, templ, method):
        # templ2 is a better match than templ1
        score = 0.9 if templ.ident == "templ2" else 0.2
        return [[score]]

    def minMaxLoc(self, result):
        val = result[0][0]
        return 0.0, val, (0, 0), (0, 0)

def test_multiple_templates(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    # two files for the same label
    (dataset / "Ace_of_Hearts.png").write_text("a")
    (dataset / "Ace_of_Hearts_1.png").write_text("b")

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(recognizer, "cv2", dummy_cv2)
    monkeypatch.setattr(recognizer, "DATASET_DIR", dataset)
    monkeypatch.setattr(recognizer, "_TEMPLATES", None)

    target = tmp_path / "target.png"
    target.write_text("img")

    result = recognizer.recognize_card(target)

    templates = recognizer._load_templates()

    assert result == "Ace of Hearts"
    assert len(templates["Ace of Hearts"]) == 2
