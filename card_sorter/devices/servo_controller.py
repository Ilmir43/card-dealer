"""Servo control utilities for ejecting playing cards."""

from __future__ import annotations

from typing import Optional
import time

try:  # pragma: no cover - RPi.GPIO optional
    import RPi.GPIO as GPIO
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    GPIO = None  # type: ignore

try:  # pragma: no cover - pyserial optional
    import serial
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    serial = None  # type: ignore


class _HardwareInterface:
    """Abstract hardware interface used by :class:`ServoController`."""

    def dispense(self, angle: float = 90) -> None:
        """Move the servo to eject a single card.

        Parameters
        ----------
        angle:
            Desired rotation angle in degrees. Implementations may ignore the
            value if the hardware does not support variable angles.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any hardware resources."""
        # Optional for subclasses
        pass


class _GPIODriver(_HardwareInterface):
    """Servo driver using ``RPi.GPIO`` PWM control."""

    def __init__(self, pwm_pin: int, frequency: int = 50) -> None:
        if GPIO is None:
            raise RuntimeError("RPi.GPIO is required for GPIO servo control")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pwm_pin, GPIO.OUT)
        self._pwm = GPIO.PWM(pwm_pin, frequency)
        self._pwm.start(0)

    def dispense(self, angle: float = 90) -> None:
        """Rotate servo to ``angle`` degrees and back."""
        duty = 2.5 + angle / 18.0
        self._pwm.ChangeDutyCycle(duty)
        time.sleep(0.5)
        self._pwm.ChangeDutyCycle(2.5)
        time.sleep(0.5)
        self._pwm.ChangeDutyCycle(0)

    def cleanup(self) -> None:
        self._pwm.stop()
        GPIO.cleanup()


class _SerialDriver(_HardwareInterface):
    """Servo driver that sends commands over a serial connection."""

    def __init__(self, port: str, baudrate: int = 9600) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for serial servo control")
        self._serial = serial.Serial(port, baudrate)

    def dispense(self, angle: float = 90) -> None:
        cmd = f"DISPENSE:{angle}\n".encode()
        self._serial.write(cmd)
        self._serial.flush()

    def cleanup(self) -> None:
        self._serial.close()


from card_sorter.config import ServoSettings


class ServoController:
    """Control a servo motor used to dispense playing cards."""

    def __init__(
        self,
        settings: ServoSettings,
        *,
        driver: Optional[_HardwareInterface] = None,
    ) -> None:
        """Create a new controller for the servo.

        Parameters
        ----------
        settings:
            Dataclass with connection parameters for the servo.
        driver:
            Optional custom hardware driver implementing the
            :class:`_HardwareInterface` protocol.  Primarily used for testing.
        """

        if driver is not None:
            self._driver = driver
        elif settings.serial_port is not None:
            self._driver = _SerialDriver(settings.serial_port)
        elif settings.pwm_pin is not None:
            self._driver = _GPIODriver(settings.pwm_pin)
        else:
            raise ValueError("Either pwm_pin or serial_port must be specified")

    def dispense_card(self, angle: float = 90) -> None:
        """Dispense a single card by activating the servo.

        Parameters
        ----------
        angle:
            Rotation angle passed to the underlying driver.
        """

        self._driver.dispense(angle)

    def cleanup(self) -> None:
        """Release resources allocated by the controller."""

        self._driver.cleanup()
