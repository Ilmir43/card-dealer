# Card Dealer

This repository contains the initial scaffold for a playing card dealing system.

## Project layout

```
card_dealer/            # Python package for the project
    __init__.py
    camera.py
    recognizer.py
    servo_controller.py
    webapp.py
    main.py

dataset/                # Captured card images

templates/              # HTML templates used by the Flask web app
```

## Dataset

The ``dataset/`` directory stores labeled reference images used by the simple
template matcher in :mod:`card_dealer.recognizer`.  File names follow the
``<rank>_of_<suit>.<ext>`` pattern where ``<ext>`` is typically ``png``.  For
example a picture of the ace of spades would be saved as
``dataset/Ace_of_Spades.png``.


Most modules are still placeholders.  The :mod:`card_dealer.camera` module can
capture a single frame from the default camera, and
:mod:`card_dealer.servo_controller` contains a very small helper used to
activate a servo for dispensing cards.

## Installation

1. Install **Python 3.9** or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are ``opencv-python`` (or ``opencv-python-headless`` on a
headless system), ``flask`` for the web interface, ``picamera`` when using the
Raspberry Pi camera module, ``numpy`` and the optional ``RPi.GPIO`` and
``pyserial`` libraries for servo control.

## Camera configuration

The :mod:`card_dealer.camera` module provides a ``capture_image`` function
that grabs a single frame from the default camera using ``cv2.VideoCapture``.
The capture uses device index ``0`` with a resolution of **1280x720** pixels.

## Hardware setup

### Camera

* Connect a USB webcam or Raspberry Pi camera so that it is available as
  ``/dev/video0``.  If your camera uses a different index adjust the call to
  :func:`cv2.VideoCapture` in :mod:`card_dealer.camera`.
* Ensure the camera is capable of 1280x720 resolution or update the resolution
  in ``capture_image``.

### Servo

``card_dealer.servo_controller`` supports two connection methods:

1. **GPIO PWM** &ndash; pass the GPIO pin number to ``ServoController``.  This
   requires the ``RPi.GPIO`` library.
2. **Serial** &ndash; provide the serial port name (e.g. ``/dev/ttyUSB0``).  This
   requires ``pyserial``.

Wire the servo to your chosen control method and make sure the power supply is
adequate for the servo current draw.

## Running the program

The repository ships only minimal example code.  A simple test run can capture
an image and attempt to recognize it:

```python
from pathlib import Path
from card_dealer.camera import capture_image
from card_dealer.recognizer import recognize_card

img = capture_image(Path("test.png"))
print("Detected card:", recognize_card(img))
```

To dispense a card using a servo connected to GPIO pin ``11``:

```python
from card_dealer.servo_controller import ServoController

controller = ServoController(pwm_pin=11)
controller.dispense_card()
controller.cleanup()
```

These examples can be executed in a Python REPL after installing the
dependencies.

## Running tests

The repository contains a small test suite.  After installing the
dependencies, run `pytest` from the project root to execute the tests:

```bash
pytest
```

## Dataset

The :mod:`card_dealer.recognizer` module expects labeled training
images in the :file:`dataset/` directory. Each image file name should
use the pattern ``<label>.<ext>`` where ``<label>`` is the card name and
``<ext>`` is any supported image extension (for example
``Ace_of_Spades.png``). Spaces in the label are replaced with
underscores.

To collect training images manually, start the web interface and follow
the capture workflow as shown below:

```bash
python -m card_dealer.webapp
```

After capturing a card, edit the label field if necessary and press the
save button. The image is stored inside :file:`dataset/` using the label
you provided.

## Web interface

The project includes a small Flask application for capturing images and
confirming their labels. Launch it with:

```bash
python -m card_dealer.webapp
```

Navigate to ``http://localhost:5000/`` and capture a card. The page
shows the predicted card name. Enter the correct name if the prediction
is wrong and click **Save**. The labeled image is written to
:file:`dataset/` with spaces replaced by underscores.

