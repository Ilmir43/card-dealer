import pytest

try:
    from card_dealer import webapp
except ModuleNotFoundError:
    pytest.skip("Flask not available", allow_module_level=True)


def test_video_frames(monkeypatch):
    frames = ["img1", "img2"]

    def dummy_stream_frames():
        for f in frames:
            yield f

    def dummy_recognize(image):
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
    monkeypatch.setattr(webapp.recognizer, "recognize_card_array", dummy_recognize)
    monkeypatch.setattr(webapp.camera, "cv2", DummyCV2())

    gen = webapp._video_frames()
    data = b"".join([next(gen), next(gen)])
    assert b"data" in data
