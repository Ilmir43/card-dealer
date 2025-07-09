from card_dealer.sorter import is_card_back, sort_by_back
from card_dealer.servo_controller import ServoController, _HardwareInterface


class DummyDriver(_HardwareInterface):
    def __init__(self):
        self.angles = []

    def dispense(self, angle: float = 90) -> None:  # pragma: no cover - simple
        self.angles.append(angle)


def test_is_card_back():
    back_img = [[[10 for _ in range(3)] for _ in range(5)] for _ in range(5)]
    face_img = [[[10 for _ in range(3)] for _ in range(5)] for _ in range(5)]
    face_img[2][2] = [200, 200, 200]

    assert is_card_back(back_img)
    assert not is_card_back(face_img)


def test_sort_by_back():
    driver_main = DummyDriver()
    driver_sort = DummyDriver()
    servo_main = ServoController(driver=driver_main)
    servo_sort = ServoController(driver=driver_sort)

    back_img = [[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)]
    face_img = [[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)]
    face_img[0][0] = [255, 255, 255]

    sort_by_back(back_img, servo_main, servo_sort, back_angle=30, face_angle=0, deal_angle=45)
    assert driver_sort.angles[-1] == 30
    assert driver_main.angles[-1] == 45

    sort_by_back(face_img, servo_main, servo_sort, back_angle=30, face_angle=0, deal_angle=45)
    assert driver_sort.angles[-1] == 0
    assert driver_main.angles[-1] == 45
