# Загрузка и использование моделей YOLOv8

1. Установите библиотеку `ultralytics`:
   ```bash
   pip install ultralytics
   ```
2. При наличии готовых весов из проекта [Playing-Cards-Object-Detection](https://github.com/TeogopK/Playing-Cards-Object-Detection) скачайте файл `best.pt` и поместите его в каталог `models/`:
   ```bash
   wget -O models/best.pt https://github.com/TeogopK/Playing-Cards-Object-Detection/raw/main/best.pt
   ```
3. Если весов нет, обучите модель самостоятельно:
   ```bash
   python train.py --data data.yaml --epochs 50
   ```
4. После получения файла `models/best.pt` используйте класс `CardDetector` для детекции карт:
   ```python
   from detector import CardDetector
   detector = CardDetector('models/best.pt')
   boxes = detector.detect(image)
   ```

Файл `data.yaml` описывает структуру датасета. Вы можете дополнить его своими изображениями, предварительно сконвертировав датасет скриптом `convert_dataset_to_yolo.py`.

## Подготовка датасета

1. Сформируйте таблицу `cards.csv` с колонками `filepaths`, `labels`, `class index` и `data set`.
   В ней перечислены пути к изображениям карт и разбиение на подмножества `train` и `valid`.
2. Выполните конвертацию в формат YOLOv8:
   ```bash
   python convert_dataset_to_yolo.py cards.csv datasets/cards
   ```
   В каталоге `datasets/cards` появятся папки `images/` и `labels/` с соответствующими подкаталогами `train` и `val`.
   Каждое изображение копируется под уникальным именем, а `.txt`‑файлы содержат
   индекс класса из CSV.
3. Убедитесь, что параметр `path` в `data.yaml` указывает на созданный каталог `datasets/cards`.

## Дообучение YOLOv8

1. Для старта обучения используйте скрипт `train.py`, который загружает базовые веса `yolov8n.pt` и сохраняет результат в `models/best.pt`:
   ```bash
   python train.py --data data.yaml --epochs 50
   ```
2. Чтобы продолжить обучение уже существующих весов, можно воспользоваться CLI пакета `ultralytics`:
   ```bash
   yolo task=detect mode=train model=models/best.pt data=data.yaml epochs=50
   ```

После обучения полученные веса можно использовать через класс `CardDetector`.
