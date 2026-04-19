"""Render a binary crack mask over the original image.

Parallel to :func:`src.gradcam.overlay_heatmap` but for a discrete mask.
The overlay uses a solid red fill with adjustable alpha so the user can
visually verify where the segmentation model thinks the crack is,
regardless of the underlying surface colour.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

__all__ = ["overlay_mask"]


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Blend a binary mask as a coloured tint on top of the image.

    Parameters
    ----------
    image:
        Source RGB image. Converted to RGB if necessary.
    mask:
        2-D array; non-zero entries are treated as crack pixels.
    alpha:
        Opacity of the coloured fill over crack pixels (0..1).
    color:
        RGB tuple for the fill. Default red.

    Returns
    -------
    PIL Image with the same size as ``image``.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    base = np.array(image)
    h, w = base.shape[:2]

    if mask.shape != (h, w):
        mask_img = Image.fromarray(
            (mask > 0).astype(np.uint8) * 255, mode="L"
        ).resize((w, h), resample=Image.NEAREST)
        mask = (np.array(mask_img) > 0).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    alpha = float(max(0.0, min(1.0, alpha)))
    color_arr = np.array(color, dtype=np.float32)

    out = base.astype(np.float32)
    mask_3c = mask[..., None].astype(np.float32)
    out = out * (1.0 - alpha * mask_3c) + color_arr * (alpha * mask_3c)

    return Image.fromarray(np.uint8(np.clip(out, 0, 255)))


def draw_mask_contours(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> Image.Image:
    """Draw the mask outline on top of the image — crisper than a fill.

    Useful when the crack is thin (1-3 px): the contour trick the eye
    into seeing the localisation clearly without covering the crack
    pixels with colour.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    base = np.array(image)
    h, w = base.shape[:2]

    if mask.shape != (h, w):
        mask_img = Image.fromarray(
            (mask > 0).astype(np.uint8) * 255, mode="L"
        ).resize((w, h), resample=Image.NEAREST)
        mask_resized = (np.array(mask_img) > 0).astype(np.uint8)
    else:
        mask_resized = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(base, contours, -1, color, thickness=thickness)
    return Image.fromarray(base)
