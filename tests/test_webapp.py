import io
import pytest
from pathlib import Path

try:
    from card_dealer import webapp
except ModuleNotFoundError:
    pytest.skip("Flask not available", allow_module_level=True)


def test_video_frames(monkeypatch):
    frames = ["img1", "img2"]

    def dummy_stream_frames():
        for f in frames:
            yield f

    calls = {}

    def dummy_find_card(frame):
        calls.setdefault("find", []).append(frame)
        return f"card-{frame}"

    def dummy_recognize(image):
        calls.setdefault("rec", []).append(image)
        return "Ace"

    class DummyCV2:
        FONT_HERSHEY_SIMPLEX = 0

        def putText(self, frame, text, pos, font, scale, color, thickness):
            pass

        def imencode(self, ext, frame):
            return True, DummyBuffer(frame)

    class DummyBuffer:
        def __init__(self, frame):
            self.frame = frame

        def tobytes(self):
            return b"data" + bytes(self.frame, "utf-8")

    monkeypatch.setattr(webapp.camera, "stream_frames", dummy_stream_frames)
    monkeypatch.setattr(webapp.camera, "find_card", dummy_find_card)
    monkeypatch.setattr(webapp.recognizer, "recognize_card_array", dummy_recognize)
    monkeypatch.setattr(webapp.camera, "cv2", DummyCV2())

    gen = webapp._video_frames()
    data = b"".join([next(gen), next(gen)])
    assert b"data" in data
    assert calls["find"] == ["img1", "img2"]
    assert calls["rec"] == ["card-img1", "card-img2"]


def test_video_frames_handles_errors(monkeypatch):
    frames = ["img1", "img2"]

    def dummy_stream_frames():
        for f in frames:
            yield f

    def failing_recognize(image, **kwargs):
        raise RuntimeError("boom")

    class DummyCV2:
        FONT_HERSHEY_SIMPLEX = 0

        def putText(self, frame, text, pos, font, scale, color, thickness):
            pass

        def imencode(self, ext, frame):
            return True, DummyBuffer(frame)

    class DummyBuffer:
        def __init__(self, frame):
            self.frame = frame

        def tobytes(self):
            return b"data" + bytes(self.frame, "utf-8")

    monkeypatch.setattr(webapp.camera, "stream_frames", dummy_stream_frames)
    monkeypatch.setattr(webapp.camera, "find_card", lambda f: None)
    monkeypatch.setattr(webapp.predict, "recognize_card_array", failing_recognize)
    monkeypatch.setattr(webapp.camera, "cv2", DummyCV2())
    monkeypatch.setattr(webapp, "_MODEL_PATH", Path("model.pt"))

    gen = webapp._video_frames()
    data = next(gen)
    assert b"data" in data


def test_video_frames_mirror(monkeypatch):
    frames = ["img"]

    def dummy_stream_frames():
        for f in frames:
            yield f

    def dummy_recognize(image):
        return "Ace"

    calls = {}

    class DummyCV2:
        FONT_HERSHEY_SIMPLEX = 0

        def flip(self, frame, mode):
            calls["flip"] = (frame, mode)
            return frame

        def putText(self, frame, text, pos, font, scale, color, thickness):
            pass

        def imencode(self, ext, frame):
            return True, DummyBuffer(frame)

    class DummyBuffer:
        def __init__(self, frame):
            self.frame = frame

        def tobytes(self):
            return b"data" + bytes(self.frame, "utf-8")

    monkeypatch.setattr(webapp.camera, "stream_frames", dummy_stream_frames)
    monkeypatch.setattr(webapp.camera, "find_card", lambda f: None)
    monkeypatch.setattr(webapp.recognizer, "recognize_card_array", dummy_recognize)
    monkeypatch.setattr(webapp.camera, "cv2", DummyCV2())
    monkeypatch.setattr(webapp, "_MIRROR", True)

    gen = webapp._video_frames()
    next(gen)
    assert calls["flip"] == ("img", 1)


def test_verify_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp.recognizer, "DATASET_DIR", tmp_path / "ds")

    def dummy_recognize(path):
        assert path.name == webapp._UPLOAD_NAME
        return "Queen"

    monkeypatch.setattr(webapp.recognizer, "recognize_card", dummy_recognize)

    client = webapp.app.test_client()
    data = {"image": (io.BytesIO(b"img"), "card.png")}
    resp = client.post("/verify", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert b"Queen" in resp.data
    assert (tmp_path / "ds" / webapp._UPLOAD_NAME).exists()


def test_verify_upload_with_model(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp.recognizer, "DATASET_DIR", tmp_path / "ds")

    def dummy_predict(path, model_path="model.pt", device=None):
        assert path.name == webapp._UPLOAD_NAME
        assert model_path == "model.pt"
        return {"label": "Jack"}

    monkeypatch.setattr(webapp.predict, "recognize_card", dummy_predict)
    monkeypatch.setattr(webapp.recognizer, "recognize_card", lambda p: "Wrong")
    monkeypatch.setattr(webapp, "_MODEL_PATH", Path("model.pt"))

    client = webapp.app.test_client()
    data = {"image": (io.BytesIO(b"img"), "card.png")}
    resp = client.post("/verify", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert b"Jack" in resp.data


def test_confirm_logs_result(monkeypatch, tmp_path):
    ds = tmp_path / "ds"
    monkeypatch.setattr(webapp.recognizer, "DATASET_DIR", ds)
    ds.mkdir(parents=True, exist_ok=True)

    tmp_img = ds / webapp._UPLOAD_NAME
    tmp_img.write_text("img")

    client = webapp.app.test_client()
    data = {
        "image_name": webapp._UPLOAD_NAME,
        "label": "Ace",
        "prediction": "King",
    }
    resp = client.post("/confirm", data=data)

    assert resp.status_code == 302
    saved = ds / "Ace.png"
    assert saved.exists()
    log_path = ds / "verify_log.csv"
    assert log_path.exists()
    rows = log_path.read_text().splitlines()
    assert rows[1].split(",") == [saved.name, "King", "Ace"]
