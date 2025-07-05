from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from model import create_model, IMAGE_SIZE

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Transformation for numpy array inputs in RGB format
_array_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def recognize_card(image_path: str | Path, model_path: str = "model.pt", device: Optional[str] = None) -> dict[str, str]:
    """Predict the card label and type for a given image."""
    image_path = Path(image_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = create_model(num_classes=len(class_to_idx))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    label = idx_to_class.get(pred, "Unknown")
    card_type = "back" if label.endswith("_back") else "face"
    return {"type": card_type, "label": label}


def recognize_card_array(
    image: "np.ndarray", model_path: str = "model.pt", device: Optional[str] = None
) -> dict[str, str]:
    """Predict the card label for an image array."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = create_model(num_classes=len(class_to_idx))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    tensor = _array_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    label = idx_to_class.get(pred, "Unknown")
    card_type = "back" if label.endswith("_back") else "face"
    return {"type": card_type, "label": label}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Predict card from image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="model.pt", dest="model")
    args = parser.parse_args()
    result = recognize_card(args.image, args.model)
    print(result)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
