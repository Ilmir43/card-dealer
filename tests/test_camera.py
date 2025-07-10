from pathlib import Path

import card_sorter.devices.camera as camera


class DummyVideoCapture:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, index: int, api: int | None, calls: dict):
        self.index = index
        self.api = api
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

        def VideoCapture(self, index, api_preference=None):
            calls["index"] = index
            calls["api_preference"] = api_preference
            return DummyVideoCapture(index, api_preference, calls)

        def imwrite(self, path, frame):
            calls["imwrite"] = path
            Path(path).write_text("image")
            return True

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(camera, "cv2", dummy_cv2)

    out_file = tmp_path / "out.png"
    result = camera.capture_image(out_file, device_index=1, api_preference=42)

    assert result == out_file
    assert calls["index"] == 1
    assert calls["api_preference"] == 42
    assert calls["imwrite"] == str(out_file)
    assert out_file.exists()


def test_stream_frames(monkeypatch):
    calls = {"reads": []}

    class DummyCV2:
        CAP_PROP_FRAME_WIDTH = DummyVideoCapture.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = DummyVideoCapture.CAP_PROP_FRAME_HEIGHT

        def VideoCapture(self, index, api_preference=None):
            calls["index"] = index
            calls["api_preference"] = api_preference
            return StreamDummyVideoCapture(index, api_preference, calls)

    class StreamDummyVideoCapture(DummyVideoCapture):
        def __init__(self, index: int, api: int | None, calls: dict):
            super().__init__(index, api, calls)
            self.read_count = 0

        def read(self):
            self.read_count += 1
            if self.read_count <= 2:
                calls["reads"].append(self.read_count)
                return True, f"frame{self.read_count}"
            return False, None

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(camera, "cv2", dummy_cv2)

    gen = camera.stream_frames(device_index=2, api_preference=99)
    frames = list(gen)

    assert frames == ["frame1", "frame2"]
    assert calls["index"] == 2
    assert calls["api_preference"] == 99


def test_capture_image_with_settings(monkeypatch, tmp_path):
    calls = {}

    class DummyCV2:
        CAP_PROP_FRAME_WIDTH = DummyVideoCapture.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = DummyVideoCapture.CAP_PROP_FRAME_HEIGHT
        CAP_PROP_BRIGHTNESS = 10

        def VideoCapture(self, index, api_preference=None):
            cap = DummyVideoCapture(index, api_preference, calls)
            calls["cap"] = cap
            return cap

        def imwrite(self, path, frame):
            Path(path).write_text("image")
            return True

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(camera, "cv2", dummy_cv2)

    out_file = tmp_path / "out.png"
    camera.capture_image(out_file, camera_settings={"brightness": 0.5})

    prop_value = calls["cap"].props.get(dummy_cv2.CAP_PROP_BRIGHTNESS)
    assert prop_value == 0.5


def test_record_video(monkeypatch, tmp_path):
    calls = {}

    class DummyVideoWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = path
            self.frames = []
            self.opened = True
            calls["writer"] = self

        def isOpened(self):
            return self.opened

        def write(self, frame):
            self.frames.append(frame)

        def release(self):
            Path(self.path).write_text("video")

    class DummyCV2:
        CAP_PROP_FRAME_WIDTH = DummyVideoCapture.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = DummyVideoCapture.CAP_PROP_FRAME_HEIGHT

        def VideoCapture(self, index, api_preference=None):
            cap = DummyVideoCapture(index, api_preference, calls)
            calls["cap"] = cap
            return cap

        def VideoWriter_fourcc(self, *args):
            return 0

        def VideoWriter(self, path, fourcc, fps, size):
            return DummyVideoWriter(path, fourcc, fps, size)

    dummy_cv2 = DummyCV2()
    monkeypatch.setattr(camera, "cv2", dummy_cv2)

    video_file = tmp_path / "out.mp4"
    result = camera.record_video(video_file, duration_seconds=0.1, fps=2)

    assert result == video_file
    assert video_file.exists()
    # should have written some frames
    assert calls["writer"].frames


def test_find_card_detects_rectangle():
    frame = [[[0, 0, 0] for _ in range(10)] for _ in range(10)]
    for y in range(2, 6):
        for x in range(3, 5):
            frame[y][x] = [200, 200, 200]

    card = camera.find_card(frame)

    assert card is not None
    assert len(card) == 4
    assert len(card[0]) == 2
    assert all(pixel == [200, 200, 200] for row in card for pixel in row)


def test_find_card_returns_none():
    frame = [[[0, 0, 0] for _ in range(5)] for _ in range(5)]

    card = camera.find_card(frame)

    assert card is None
