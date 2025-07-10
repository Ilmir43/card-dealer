from card_sorter.recognition.sorter import is_card_back, sort_by_back


def _dummy_classifier(img):
    return all(ch == 0 for ch in img[0][0])


def test_is_card_back():
    back_img = [[[0 for _ in range(3)] for _ in range(5)] for _ in range(5)]
    face_img = [[[0 for _ in range(3)] for _ in range(5)] for _ in range(5)]
    face_img[0][0] = [1, 1, 1]

    assert is_card_back(back_img, classifier=_dummy_classifier)
    assert not is_card_back(face_img, classifier=_dummy_classifier)


def test_sort_by_back():
    back_img = [[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)]
    face_img = [[[0 for _ in range(3)] for _ in range(4)] for _ in range(4)]
    face_img[0][0] = [255, 255, 255]

    groups = sort_by_back([
        (back_img, is_card_back(back_img, classifier=_dummy_classifier)),
        (face_img, is_card_back(face_img, classifier=_dummy_classifier)),
    ])
    assert groups["back"] == [back_img]
    assert groups["face"] == [face_img]
