from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

from . import camera, recognizer

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# Temporary capture filename inside dataset directory
_CAPTURE_NAME = "_capture.png"


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
