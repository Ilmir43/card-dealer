# Руководство по разработке

Этот проект использует архитектуру, основанную на принципах SOLID и паттернах
проектирования. Новые модули должны строго соблюдать следующие правила.

## Общие рекомендации

- Пишите код на Python 3.9+ и придерживайтесь стиля PEP 8.
- Соблюдайте DRY и KISS: избегайте дублирования и усложнения логики.
- Все параметры и пути берите из конфигурации в `card_sorter/config`.
- Используйте `dataclass` и `typing` для явных контрактов.
- Покрывайте ключевые компоненты модульными тестами в каталоге `tests/`.

## Структура каталога `card_sorter`

- `core` – базовые интерфейсы и утилиты.
- `devices` – взаимодействие с оборудованием (камера, сервоприводы и т.п.).
- `models` – модели машинного обучения и фабрики для их создания.
- `recognition` – бизнес‑логика распознавания и сортировки.
- `ui` – веб‑интерфейс (Streamlit/Flask) без бизнес‑логики.

При добавлении новых функций выбирайте каталог исходя из их назначения.

## Использование интерфейсов

Все публичные компоненты должны реализовывать соответствующие протоколы из
`card_sorter.core.interfaces`.  Например, собственный распознаватель карт должен
поддерживать `CardRecognizer`, а контроллер сервопривода – `DeviceController`.
Конкретные реализации создавайте через фабрики (см. `models.factory` и другие).

Такой подход позволяет заменять зависимости без изменения остального кода и
упрощает тестирование.

## Входная точка

Приложение запускается через `main.py` в корне репозитория. Все скрипты и
инструменты должны вызывать функции из пакета `card_sorter`, не дублируя логику.

## Документация

Документируйте публичные функции и классы докстрингами на русском языке и
обновляйте `README.md`, если добавляются новые возможности.
