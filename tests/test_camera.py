from pathlib import Path

import card_dealer.camera as camera


class DummyVideoCapture:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, index: int, calls: dict):
        self.index = index
        self.calls = calls
        self.props = {}
        self.opened = True

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        self.calls["read"] = True
        return True, "frame"

    def release(self):
        self.calls["released"] = True


def test_capture_image(monkeypatch, tmp_path):
    calls = {}

    class DummyCV2:
        CAP_PROP_FRAME_WIDTH = DummyVideoCapture.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = DummyVideoCapture.CAP_PROP_FRAME_HEIGHT

        def VideoCapture(self, index):
            calls["index"] = index
            return DummyVideoCapture(index, calls)

        def imwrite(self, path, frame):
            calls["imwrite"] = path
            Path(path).write_text("image")
            return True

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(camera, "cv2", dummy_cv2)

    out_file = tmp_path / "out.png"
    result = camera.capture_image(out_file)

    assert result == out_file
    assert calls["index"] == 0
    assert calls["imwrite"] == str(out_file)
    assert out_file.exists()
