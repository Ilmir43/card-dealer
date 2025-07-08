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
