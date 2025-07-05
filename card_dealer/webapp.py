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

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Temporary capture filename inside dataset directory
_CAPTURE_NAME = "_capture.png"
# Path to a CNN model used for live recognition; None means template recognizer
_MODEL_PATH: Path | None = None
# Latest label recognized in the video stream
_LATEST_LABEL: str = ""


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
    )


@app.route("/dataset/<path:filename>")
def dataset_file(filename: str):
    """Serve files from the dataset directory."""
    return send_from_directory(recognizer.DATASET_DIR, filename)


def _video_frames():
    """Generate JPEG frames with recognition overlay."""
    global _LATEST_LABEL
    for frame in camera.stream_frames():
        if _MODEL_PATH is not None:
            result = predict.recognize_card_array(frame, model_path=str(_MODEL_PATH))
            label = result.get("label", "Unknown")
        else:
            label = recognizer.recognize_card_array(frame)
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


@app.route("/live")
def live() -> str:
    """Display a page with the live video feed."""
    return render_template("live.html")


@app.route("/current_label")
def current_label() -> dict[str, str]:
    """Return the latest recognized label as JSON."""
    return {"label": _LATEST_LABEL}


@app.route("/model", methods=["GET", "POST"])
def select_model() -> str:
    """Select the model used for live recognition."""
    global _MODEL_PATH
    if request.method == "POST":
        path = request.form.get("model_path", "").strip()
        _MODEL_PATH = Path(path) if path else None
        return redirect(url_for("live"))
    return render_template("select_model.html", model_path=_MODEL_PATH)


@app.route("/confirm", methods=["POST"])
def confirm() -> str:
    """Save the labeled image provided by the user."""
    label = request.form.get("label", "").strip()
    image_path = recognizer.DATASET_DIR / request.form.get("image_name", _CAPTURE_NAME)
    if label:
        recognizer.save_labeled_image(image_path, label)
    if image_path.exists():
        image_path.unlink()
    return redirect(url_for("next_card"))


@app.route("/next")
def next_card() -> str:
    """Display a page indicating that processing is complete."""
    return render_template("next.html")


if __name__ == "__main__":  # pragma: no cover
    app.run(debug=True)
