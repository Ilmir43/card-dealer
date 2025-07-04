from pathlib import Path
from typing import Optional

import torch
from torchvision import transforms
from PIL import Image

from model import CardClassifier

IMAGE_SIZE = (100, 150)

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def recognize_card(image_path: str | Path, model_path: str = "model.pt", device: Optional[str] = None) -> str:
    """Load a trained model and predict the card label for ``image_path``."""
    image_path = Path(image_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model = CardClassifier(num_classes=len(class_to_idx))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return idx_to_class.get(pred, "Unknown")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict card from image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="model.pt", dest="model")
    args = parser.parse_args()
    result = recognize_card(args.image, args.model)
    print(result)


if __name__ == "__main__":
    main()
