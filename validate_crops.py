from pathlib import Path
import cv2


def validate_image(path: Path) -> None:
    crop_path = path.with_name(path.stem + "_crop" + path.suffix)
    if not crop_path.exists():
        print(f"Нет обрезки для {path}")
        return
    orig = cv2.imread(str(path))
    crop = cv2.imread(str(crop_path))
    if orig is None or crop is None:
        print(f"Ошибка чтения {path}")
        return
    print(f"{path} -> {crop.shape[1]}x{crop.shape[0]}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Проверить обрезанные изображения")
    parser.add_argument("path", type=Path, help="Файл или каталог")
    args = parser.parse_args()
    p = args.path
    if p.is_file():
        validate_image(p)
    else:
        for img in p.rglob("*.jpg"):
            if img.name.endswith("_crop.jpg"):
                continue
            validate_image(img)


if __name__ == "__main__":
    main()
