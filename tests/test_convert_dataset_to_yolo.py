from pathlib import Path
import csv
from convert_dataset_to_yolo import convert, _unique_filename


def test_unique_filename():
    src = Path('/path/to/class1/image.jpg')
    assert _unique_filename(src) == 'class1_image.jpg'


def test_convert_creates_files(tmp_path):
    img1 = tmp_path / 'img1.png'
    img2 = tmp_path / 'img2.png'
    img1.write_text('a')
    img2.write_text('b')
    csv_path = tmp_path / 'cards.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class index', 'filepaths', 'labels', 'data set'])
        writer.writerow([0, str(img1), 'a', 'train'])
        writer.writerow([1, str(img2), 'b', 'valid'])

    out_dir = tmp_path / 'out'
    convert(csv_path, out_dir)

    train_img = next((out_dir / 'images' / 'train').glob('*.png'))
    val_img = next((out_dir / 'images' / 'val').glob('*.png'))

    assert train_img.exists()
    assert val_img.exists()

    train_lbl = out_dir / 'labels' / 'train' / (train_img.stem + '.txt')
    val_lbl = out_dir / 'labels' / 'val' / (val_img.stem + '.txt')

    assert train_lbl.read_text().strip() == '0 0.5 0.5 1 1'
    assert val_lbl.read_text().strip() == '1 0.5 0.5 1 1'
