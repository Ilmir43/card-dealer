import pytest
pytest.importorskip("torch")
import torch
from torch import nn

import predict
from model import create_model


def test_load_model_simple(tmp_path):
    model = create_model(3, simple_head=True)
    ckpt = {"model_state": model.state_dict()}
    path = tmp_path / "simple.pt"
    torch.save(ckpt, path)

    loaded, _ = predict._load_model(str(path), "cpu")
    assert isinstance(loaded.fc, nn.Linear)


def test_load_model_extended(tmp_path):
    model = create_model(3, simple_head=False)
    ckpt = {"model_state": model.state_dict()}
    path = tmp_path / "extended.pt"
    torch.save(ckpt, path)

    loaded, _ = predict._load_model(str(path), "cpu")
    assert isinstance(loaded.fc, nn.Sequential)
