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

All modules are currently empty and act as placeholders for future
implementation.

## Camera configuration

The :mod:`card_dealer.camera` module provides a ``capture_image`` function
that grabs a single frame from the default camera using ``cv2.VideoCapture``.
The capture uses device index ``0`` with a resolution of **1280x720** pixels.

