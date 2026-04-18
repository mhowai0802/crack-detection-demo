"""Lightweight Grad-CAM implementation for ResNet18.

Grad-CAM highlights the spatial regions of the input that most strongly
influence a specific class score. For a crack classifier this gives a visual
answer to "where did the model see the crack?" which is valuable for both
interview discussion and field trust in the prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.dataset import get_eval_transform


class GradCAM:
    """Grad-CAM for a single target layer.

    Parameters
    ----------
    model:
        A classifier in eval mode.
    target_layer:
        The layer whose activations we want to visualise. For ResNet18 this is
        typically ``model.layer4``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def compute(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Return a normalised [H, W] heatmap in ``[0, 1]``."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_index]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target layer.")

        activations = self._activations[0]  # [C, H, W]
        gradients = self._gradients[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # global-avg-pool over spatial dims

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max
        return cam_np


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """Blend a Grad-CAM heatmap on top of the original RGB image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    base = np.array(image)
    h, w = base.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = np.uint8((1 - alpha) * base + alpha * heatmap_color)
    return Image.fromarray(blended)


def compute_gradcam_overlay(
    image: Image.Image,
    model: nn.Module,
    class_index: int,
    device: Union[str, torch.device] = "cpu",
    alpha: float = 0.45,
) -> Image.Image:
    """Convenience wrapper: preprocess, compute CAM, and return an overlay."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_eval_transform()
    tensor = transform(image).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    target_layer = model.layer4  # type: ignore[attr-defined]
    with GradCAM(model, target_layer) as cam:
        heatmap = cam.compute(tensor, class_index=class_index)

    return overlay_heatmap(image, heatmap, alpha=alpha)
