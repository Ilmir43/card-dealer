"""Main control loop for the card dealing system."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading

from flask import Flask, redirect, render_template_string, request, url_for

from .devices import camera, servo_controller
from .recognition import recognizer


def run() -> None:
    """Run the card dealing loop with a simple Flask interface."""

    app = Flask(__name__)

    latest: dict[str, str] = {"label": ""}
    confirm_event = threading.Event()

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        if request.method == "POST":
            confirm_event.set()
            return redirect(url_for("index"))
        return render_template_string(
            """
            <html>
            <body>
            <h1>Recognized card: {{ label }}</h1>
            <form method="post">
              <button type="submit">Dispense next card</button>
            </form>
            </body>
            </html>
            """,
            label=latest["label"],
        )

    def start_app() -> None:
        app.run(debug=False, use_reloader=False)

    thread = threading.Thread(target=start_app, daemon=True)
    thread.start()

    image_file = Path(tempfile.gettempdir()) / "captured_card.png"
    servo = servo_controller.ServoController(pwm_pin=12)

    try:
        while True:
            camera.capture_image(image_file)
            latest["label"] = recognizer.recognize_card(image_file)
            confirm_event.clear()
            confirm_event.wait()
            servo.dispense_card()
    finally:
        servo.cleanup()


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    run()

