# Card Dealer

**Внимание!** Вся документация в этом проекте должна быть на русском языке.

Данный репозиторий содержит заготовку системы для распознавания и раздачи игральных карт.

## Структура проекта

```
card_dealer/            # Python-пакет с исходным кодом
    __init__.py
    camera.py           # захват изображений с камеры
    recognizer.py       # простое распознавание карт по шаблонам
    servo_controller.py # управление сервоприводом
    webapp.py           # веб-интерфейс на Flask
    main.py             # пример запуска

dataset/                # изображения карт и временные снимки
templates/              # HTML-шаблоны для веб-приложения
train_model.py          # обучение сверточной сети
predict.py              # использование обученной модели
model.py                # архитектура нейронной сети
utils.py                # утилиты сохранения модели
```

## Установка

1. Требуется **Python 3.9** или новее.
2. При необходимости создайте и активируйте виртуальное окружение.
3. Установите зависимости:

```bash
pip install -r requirements.txt
```

Основные зависимости: `opencv-python`, `flask`, `numpy`, `torch` и `torchvision`.
Пакеты `picamera` и `RPi.GPIO` нужны только на Raspberry Pi и
автоматически пропускаются при установке на macOS и других системах.

## Конфигурация камеры

Модуль `card_dealer.camera` предоставляет функцию `capture_image`, снимающую кадр с устройства `0` (обычно `/dev/video0`) в разрешении **1280x720**. На macOS с чипом M1 можно передать параметр `api_preference=cv2.CAP_AVFOUNDATION`, чтобы задействовать встроенную камеру.
Функция `stream_frames` использует те же параметры и позволяет получать видео
для веб‑страницы `/live`.

## Аппаратная часть

### Камера

- Подключите USB‑камеру или камеру Raspberry Pi, чтобы она была доступна как `/dev/video0`.
- При необходимости измените индекс устройства или разрешение в функции `capture_image`.

### Серво

`card_dealer.servo_controller` позволяет управлять сервомотором двумя способами:

1. **GPIO PWM** — укажите номер GPIO‑пина при создании `ServoController`.
2. **Последовательный порт** — укажите имя порта (например, `/dev/ttyUSB0`).
   Для этого режима необходим пакет `pyserial`.

## Запуск примеров

Простейший пример захватывает изображение и пытается определить карту:

```python
from pathlib import Path
from card_dealer.camera import capture_image
import cv2
from card_dealer.recognizer import recognize_card

# Для macOS можно указать api_preference=cv2.CAP_AVFOUNDATION
img = capture_image(Path("test.png"), api_preference=cv2.CAP_AVFOUNDATION)
print("Обнаружена карта:", recognize_card(img))
```

Для выдачи карты сервоприводом на GPIO‑пине `11`:

```python
from card_dealer.servo_controller import ServoController

controller = ServoController(pwm_pin=11)
controller.dispense_card()
controller.cleanup()
```

## Тесты

После установки зависимостей запустите тесты командой:

```bash
pytest
```

## Сбор датасета

Каталог `dataset/` содержит изображения карт, используемые распознавателем. Сбор новых изображений осуществляется через веб‑интерфейс:

```bash
python -m card_dealer.webapp
```

После съёмки отредактируйте название карты при необходимости и нажмите **Save**. Файл будет сохранён в `dataset/` с названием вида `Имя_карты.png`.

## Веб‑интерфейс

Запустите:

```bash
python -m card_dealer.webapp
```

Перейдите на `http://localhost:5000/`, снимите карту, проверьте название и сохраните изображение. При подтверждении фотография попадёт в `dataset/`.

Для просмотра видео с распознаванием в реальном времени откройте страницу
`http://localhost:5000/live`. На кадрах будет отображаться название найденной
карты.

## Обучение нейронной сети

Для более точного распознавания предусмотрена простая сверточная сеть. Скрипт `train_model.py` читает разметку из CSV‑файла (по умолчанию `cards.csv`) и изображения из папки `dataset/`. Обучение запускается так:

```bash
python train_model.py --csv cards.csv --epochs 10 --model-path model.pt
```

Параметр `--resume` позволяет продолжить обучение, если модель уже существует. После завершения веса сохраняются в `model.pt`, а соответствие меток — в том же файле.

## Использование модели

Для распознавания карты по изображению используйте `predict.py`:

```python
from predict import recognize_card

result = recognize_card("some_image.png", model_path="model.pt")
print(result["label"])
```

Файл `model.pt` должен находиться в корне проекта, если не указано иное.

