from __future__ import annotations

"""\
Утилита для конвертации сохранений моделей между форматами ``.h5`` (Keras)
и ``.pt`` (PyTorch). Конвертер предполагает одинаковую архитектуру сети в
обоих фреймворках. В проекте используется ResNet18, поэтому в примерах
создаётся эквивалентная модель.

Зависимости ``tensorflow`` и ``torch`` загружаются только при вызове функций,
что позволяет импортировать модуль даже при отсутствии библиотек.
"""

from pathlib import Path


def _create_keras_resnet18(num_classes: int):
    """Версия ResNet18 на Keras для конвертации весов."""
    import tensorflow as tf

    Layer = tf.keras.layers

    def _block(x, filters, stride=1, name="block"):
        y = Layer.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                         name=f"{name}_conv1")(x)
        y = Layer.BatchNormalization(name=f"{name}_bn1")(y)
        y = Layer.ReLU(name=f"{name}_relu1")(y)
        y = Layer.Conv2D(filters, 3, padding="same", use_bias=False,
                         name=f"{name}_conv2")(y)
        y = Layer.BatchNormalization(name=f"{name}_bn2")(y)

        if stride != 1 or x.shape[-1] != filters:
            shortcut = Layer.Conv2D(filters, 1, strides=stride, use_bias=False,
                                   name=f"{name}_down_conv")(x)
            shortcut = Layer.BatchNormalization(name=f"{name}_down_bn")(shortcut)
        else:
            shortcut = x

        y = Layer.Add(name=f"{name}_add")([y, shortcut])
        return Layer.ReLU(name=f"{name}_relu2")(y)

    def _make_layer(x, filters, blocks, stride=1, name="layer"):
        x = _block(x, filters, stride=stride, name=f"{name}0")
        for i in range(1, blocks):
            x = _block(x, filters, stride=1, name=f"{name}{i}")
        return x

    inputs = Layer.Input(shape=(224, 224, 3))
    x = Layer.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="conv1")(inputs)
    x = Layer.BatchNormalization(name="bn1")(x)
    x = Layer.ReLU(name="relu")(x)
    x = Layer.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _make_layer(x, 64, 2, stride=1, name="layer1_")
    x = _make_layer(x, 128, 2, stride=2, name="layer2_")
    x = _make_layer(x, 256, 2, stride=2, name="layer3_")
    x = _make_layer(x, 512, 2, stride=2, name="layer4_")

    x = Layer.GlobalAveragePooling2D()(x)
    outputs = Layer.Dense(num_classes, name="fc")(x)
    return tf.keras.Model(inputs, outputs)


def h5_to_pt(h5_path: str | Path, pt_path: str | Path) -> None:
    """Преобразовать модель из формата Keras ``.h5`` в ``.pt``."""
    import tensorflow as tf
    import torch
    from model import create_model

    h5_path = Path(h5_path)
    pt_path = Path(pt_path)

    keras_model = tf.keras.models.load_model(h5_path)
    num_classes = keras_model.output_shape[-1]

    torch_model = create_model(num_classes)
    torch_state = torch_model.state_dict()

    new_state = {}
    for (name, param), w in zip(torch_state.items(), keras_model.weights):
        arr = w.numpy()
        if arr.ndim == 4:  # Conv2D weights
            arr = arr.transpose(3, 2, 0, 1)
        elif arr.ndim == 2:  # Dense weights
            arr = arr.transpose()
        new_state[name] = torch.tensor(arr, dtype=param.dtype)
    torch_model.load_state_dict(new_state)

    torch.save({"model_state": torch_model.state_dict()}, pt_path)


def pt_to_h5(pt_path: str | Path, h5_path: str | Path) -> None:
    """Преобразовать модель из ``.pt`` в ``.h5`` для Keras."""
    import tensorflow as tf
    import torch
    from model import create_model

    pt_path = Path(pt_path)
    h5_path = Path(h5_path)

    checkpoint = torch.load(pt_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    num_classes = state_dict["fc.weight"].shape[0]

    torch_model = create_model(num_classes)
    torch_model.load_state_dict(state_dict)

    keras_model = _create_keras_resnet18(num_classes)

    weight_list = []
    for tensor in torch_model.state_dict().values():
        arr = tensor.cpu().numpy()
        if arr.ndim == 4:
            arr = arr.transpose(2, 3, 1, 0)
        elif arr.ndim == 2:
            arr = arr.transpose()
        weight_list.append(arr)

    keras_model.set_weights(weight_list)
    keras_model.save(h5_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Конвертер моделей между .h5 и .pt")
    parser.add_argument("src", help="Исходный файл модели")
    parser.add_argument("dst", help="Файл назначения")
    parser.add_argument("--reverse", action="store_true", help="Конвертировать .pt -> .h5")
    args = parser.parse_args()

    if args.reverse:
        pt_to_h5(args.src, args.dst)
    else:
        h5_to_pt(args.src, args.dst)
