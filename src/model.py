"""Model architecture and checkpoint loading for the crack classifier.

Uses transfer learning from an ImageNet-pretrained ResNet18. The final fully
connected layer is replaced with a 2-way classifier (No Crack / Crack).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

CLASS_NAMES = ["No Crack", "Crack"]


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with its classifier head replaced for ``num_classes``.

    Args:
        num_classes: Number of output classes. Defaults to 2.
        pretrained: If ``True``, loads ImageNet weights for the backbone.

    Returns:
        A ``torch.nn.Module`` ready for training or inference.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the final classifier (``model.fc``)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter so the whole network can be fine-tuned."""
    for param in model.parameters():
        param.requires_grad = True


def load_model(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    num_classes: int = 2,
) -> nn.Module:
    """Instantiate the architecture and load trained weights.

    Args:
        checkpoint_path: Path to a ``.pt`` state-dict saved by ``train.py``.
        device: Device to load the model on.
        num_classes: Number of output classes (must match the checkpoint).

    Returns:
        A ``torch.nn.Module`` in ``eval`` mode on the requested device.
    """
    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
