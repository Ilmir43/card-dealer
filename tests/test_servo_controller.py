from card_dealer.servo_controller import ServoController, _HardwareInterface

class DummyDriver(_HardwareInterface):
    def __init__(self):
        self.dispense_called = 0
        self.cleanup_called = 0
    def dispense(self) -> None:
        self.dispense_called += 1
    def cleanup(self) -> None:
        self.cleanup_called += 1


def test_servo_controller_uses_driver():
    driver = DummyDriver()
    controller = ServoController(driver=driver)
    controller.dispense_card()
    controller.cleanup()
    assert driver.dispense_called == 1
    assert driver.cleanup_called == 1
