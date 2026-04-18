"""Inference helper for the crack classifier."""

from __future__ import annotations

from typing import Dict, Union

import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import get_eval_transform
from src.model import CLASS_NAMES


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, object]:
    """Run a single-image prediction.

    Args:
        image: An RGB PIL image. Non-RGB images are converted automatically.
        model: A trained classifier in ``eval`` mode.
        device: Device to run inference on.

    Returns:
        Dictionary with keys:
            - ``label`` (str): predicted class name
            - ``confidence`` (float): probability of the predicted class [0, 1]
            - ``probs`` (list[float]): per-class probabilities ordered as
              ``src.model.CLASS_NAMES``
            - ``class_index`` (int): index of the predicted class
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_eval_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    class_index = int(torch.tensor(probs).argmax().item())
    return {
        "label": CLASS_NAMES[class_index],
        "confidence": float(probs[class_index]),
        "probs": [float(p) for p in probs],
        "class_index": class_index,
    }
