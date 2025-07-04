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

Most modules are still placeholders.  The :mod:`card_dealer.camera` module can
capture a single frame from the default camera, and
:mod:`card_dealer.servo_controller` contains a very small helper used to
activate a servo for dispensing cards.

## Camera configuration

The :mod:`card_dealer.camera` module provides a ``capture_image`` function
that grabs a single frame from the default camera using ``cv2.VideoCapture``.
The capture uses device index ``0`` with a resolution of **1280x720** pixels.

