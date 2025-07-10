from card_sorter.devices.servo_controller import ServoController, _HardwareInterface
from card_sorter.config import ServoSettings

class DummyDriver(_HardwareInterface):
    def __init__(self):
        self.dispense_args = []
        self.cleanup_called = 0

    def dispense(self, angle: float = 90) -> None:
        self.dispense_args.append(angle)

    def cleanup(self) -> None:
        self.cleanup_called += 1


def test_servo_controller_uses_driver():
    driver = DummyDriver()
    controller = ServoController(settings=ServoSettings(), driver=driver)
    controller.dispense_card(angle=45)
    controller.cleanup()
    assert driver.dispense_args == [45]
    assert driver.cleanup_called == 1
