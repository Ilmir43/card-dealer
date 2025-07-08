from __future__ import annotations

from pathlib import Path

from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)

from . import camera, recognizer
import predict

# List of all standard playing cards used for manual labeling
_RANKS = [
    "Ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
]
_SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
CARD_NAMES = [f"{rank} of {suit}" for suit in _SUITS for rank in _RANKS]

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Temporary capture filename inside dataset directory
_CAPTURE_NAME = "_capture.png"
# Temporary filename for uploaded verification images
_UPLOAD_NAME = "_upload.png"
# Path to a CNN model used for live recognition; None means template recognizer
_MODEL_PATH: Path | None = None
# Latest label recognized in the video stream
_LATEST_LABEL: str = ""
# Mirror video feed horizontally when True
_MIRROR: bool = False


def _list_models() -> list[Path]:
    """Return available model files in the project root."""
    return sorted(ROOT_DIR.glob("*.pt"))


def _find_card_image(label: str) -> Path | None:
    """Return path to an example image for a recognized label."""
    if not label:
        return None
    base = label.replace(" ", "_")
    for path in recognizer.DATASET_DIR.glob(f"{base}*"):
        if path.is_file():
            return path
    return None


# Choose the first available model as default
_models = _list_models()
if _MODEL_PATH is None and _models:
    _MODEL_PATH = _models[0]


@app.route("/")
def index() -> str:
    """Redirect to the capture route."""
    return redirect(url_for("capture"))


@app.route("/capture")
def capture() -> str:
    """Capture an image and display the recognition result."""
    image_path = recognizer.DATASET_DIR / _CAPTURE_NAME
    camera.capture_image(image_path)
    prediction = recognizer.recognize_card(image_path)
    return render_template(
        "confirm.html",
        image_name=_CAPTURE_NAME,
        prediction=prediction,
        card_names=CARD_NAMES,
        back_url=url_for("capture"),
    )


@app.route("/dataset/<path:filename>")
def dataset_file(filename: str):
    """Serve files from the dataset directory."""
    return send_from_directory(recognizer.DATASET_DIR, filename)


@app.route("/verify", methods=["GET", "POST"])
def verify_upload() -> str:
    """Upload an image for recognition verification."""
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = recognizer.DATASET_DIR / _UPLOAD_NAME
            image_path.parent.mkdir(parents=True, exist_ok=True)
            file.save(str(image_path))
            prediction = recognizer.recognize_card(image_path)
            return render_template(
                "confirm.html",
                image_name=_UPLOAD_NAME,
                prediction=prediction,
                card_names=CARD_NAMES,
                back_url=url_for("verify_upload"),
            )
        return redirect(url_for("verify_upload"))
    return render_template("upload.html")


def _video_frames():
    """Generate JPEG frames with recognition overlay."""
    global _LATEST_LABEL
    for frame in camera.stream_frames():
        if _MIRROR:
            frame = camera.cv2.flip(frame, 1)
        bounds = camera.find_card_bounds(frame)
        card = camera.find_card(frame) if bounds is not None else None
        if bounds is not None:
            x1, y1, x2, y2 = bounds
            camera.cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        source = card if card is not None else frame
        try:
            if _MODEL_PATH is not None:
                result = predict.recognize_card_array(
                    source, model_path=str(_MODEL_PATH)
                )
                label = result.get("label", "Unknown")
            else:
                label = recognizer.recognize_card_array(source)
        except Exception:  # pragma: no cover - safety net for video feed
            label = "Error"
        _LATEST_LABEL = label
        camera.cv2.putText(
            frame,
            label,
            (10, 30),
            camera.cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        success, buffer = camera.cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed() -> Response:
    """Stream video from the camera with recognition results."""
    return Response(_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live", methods=["GET", "POST"])
def live() -> str:
    """Display a page with the live video feed and model selection."""
    global _MODEL_PATH, _MIRROR
    if request.method == "POST":
        path = request.form.get("model_path", "").strip()
        _MODEL_PATH = Path(path) if path else None
        _MIRROR = bool(request.form.get("mirror"))
        return redirect(url_for("live"))
    return render_template(
        "live.html", model_path=_MODEL_PATH, models=_list_models(), mirror=_MIRROR
    )


@app.route("/current_label")
def current_label() -> dict[str, str]:
    """Return the latest recognized label as JSON."""
    return {"label": _LATEST_LABEL}


@app.route("/current_card")
def current_card() -> dict[str, str | None]:
    """Return the latest recognized label and image."""
    image_path = _find_card_image(_LATEST_LABEL)
    image_url = (
        url_for("dataset_file", filename=image_path.name) if image_path else None
    )
    return {"label": _LATEST_LABEL, "image": image_url}




@app.route("/confirm", methods=["POST"])
def confirm() -> str:
    """Save the labeled image provided by the user."""
    label = request.form.get("label", "").strip() or "Unknown"
    predicted = request.form.get("prediction", "").strip() or label
    image_path = recognizer.DATASET_DIR / request.form.get("image_name", _CAPTURE_NAME)
    dest = recognizer.record_verification(image_path, predicted, label)
    if image_path.exists():
        image_path.unlink()
    return redirect(url_for("next_card"))


@app.route("/next")
def next_card() -> str:
    """Display a page indicating that processing is complete."""
    return render_template("next.html")


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
