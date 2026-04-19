"""Segmentation inference and mask statistics.

Given a trained :class:`src.seg_model.UNet` and a PIL image, produce a
binary crack mask at the input resolution plus a small bag of
shape-level statistics the AI report can quote.

The statistics deliberately stay in **pixel units**. Converting to
millimetres requires a calibration scale (``px_per_mm``) that the model
cannot infer from a single uncalibrated photo, so we surface the
``px_per_mm`` hook for callers who do have that number (e.g. a tape
measure visible in frame) but otherwise report the raw pixel values
with a loud "requires on-site measurement" caveat in the prompt.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision import tv_tensors
except ImportError as exc:
    raise ImportError(
        "torchvision >= 0.17 is required for tv_tensors."
    ) from exc

from src.seg_dataset import DEFAULT_SEG_IMAGE_SIZE, get_eval_transform

__all__ = [
    "predict_mask",
    "mask_stats",
]


@torch.inference_mode()
def predict_mask(
    image: Image.Image,
    model: nn.Module,
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.5,
    image_size: int = DEFAULT_SEG_IMAGE_SIZE,
    resize_to_original: bool = True,
) -> np.ndarray:
    """Return a binary ``uint8`` mask (values 0 or 1).

    Parameters
    ----------
    image:
        PIL RGB image. Will be converted if needed.
    model:
        A :class:`src.seg_model.UNet` in eval mode.
    threshold:
        Probability threshold applied to the sigmoid output.
    image_size:
        Side length used for network input.
    resize_to_original:
        If ``True`` (default) the returned mask matches the original
        ``image.size`` — convenient for overlay rendering. If ``False``
        the mask stays at ``image_size x image_size``.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_eval_transform(image_size)
    img_tv = tv_tensors.Image(image)
    mask_stub = tv_tensors.Mask(
        torch.zeros(image.size[1], image.size[0], dtype=torch.uint8)
    )
    img_tv, _ = transform(img_tv, mask_stub)
    tensor = img_tv.unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8)

    if resize_to_original and (
        mask.shape != (image.size[1], image.size[0])
    ):
        mask_img = Image.fromarray(mask * 255, mode="L").resize(
            image.size, resample=Image.NEAREST
        )
        mask = (np.array(mask_img) > 0).astype(np.uint8)

    return mask


def _connected_components(binary_mask: np.ndarray) -> int:
    """Count 8-connected components using OpenCV."""
    if binary_mask.sum() == 0:
        return 0
    num_labels, _ = cv2.connectedComponents(
        binary_mask.astype(np.uint8), connectivity=8
    )
    return int(num_labels - 1)


def _estimate_max_width_px(binary_mask: np.ndarray) -> float:
    """Rough max crack width in pixels.

    Uses ``cv2.distanceTransform`` to compute, for every crack pixel,
    the Euclidean distance to the nearest background pixel. Maximum
    such distance multiplied by two approximates the widest part of the
    crack (since the widest point is roughly twice its half-width, i.e.
    distance to the nearest edge).
    """
    if binary_mask.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(
        binary_mask.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3
    )
    return float(dist.max() * 2.0)


def _morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """Lantuejoul skeleton of a binary mask — pure cv2, no contrib dep.

    The skeleton approximates the crack's medial axis. Summing its
    pixel count gives a much better "length" proxy than the raw pixel
    area, especially for wider cracks where area overcounts length by
    the width factor.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(binary_mask, dtype=np.uint8)
    img = binary_mask.astype(np.uint8).copy()
    while img.any():
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, skel)
        img = eroded
    return skeleton


def mask_stats(
    binary_mask: np.ndarray,
    px_per_mm: Optional[float] = None,
) -> Dict[str, float]:
    """Compute summary stats for a binary crack mask.

    Returns a dict with:

    - ``crack_pixel_ratio`` — fraction of pixels labelled crack (0..1)
    - ``num_components`` — number of disjoint crack regions (8-conn.)
    - ``area_px`` — total number of crack pixels (filled area)
    - ``length_px`` — pixel count along the morphological skeleton, a
      much better length proxy than raw area for non-hairline cracks
    - ``max_width_px`` — estimated max crack width in pixels, via
      ``2 * max(distance-to-background)``
    - ``max_width_mm`` / ``length_mm`` — only present if ``px_per_mm``
      is provided. No calibration supplied → not in the dict.
    - ``image_height_px`` / ``image_width_px`` — mask dimensions for
      downstream px↔mm scale calculations.

    All values are ``int`` or ``float`` so the dict serialises cleanly
    to JSON for the FastAPI response.
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            f"Expected 2-D mask, got shape {binary_mask.shape}"
        )

    mask = (binary_mask > 0).astype(np.uint8)
    total = mask.size
    area = int(mask.sum())
    h, w = mask.shape

    skeleton = _morphological_skeleton(mask) if area > 0 else mask
    length_px = int(skeleton.sum())

    stats: Dict[str, float] = {
        "crack_pixel_ratio": (area / total) if total else 0.0,
        "num_components": _connected_components(mask),
        "area_px": int(area),
        "length_px": int(length_px),
        "max_width_px": _estimate_max_width_px(mask),
        "image_height_px": int(h),
        "image_width_px": int(w),
    }

    if px_per_mm and px_per_mm > 0:
        stats["max_width_mm"] = stats["max_width_px"] / float(px_per_mm)
        stats["length_mm"] = stats["length_px"] / float(px_per_mm)

    return stats
