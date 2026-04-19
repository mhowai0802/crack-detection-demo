"""Shared dependencies for the FastAPI routers.

Keeps the model + device objects in module-level caches so a single
worker serves many requests without reloading weights. The helpers
accept a PIL image (already decoded from the request) and return plain
Python data that Pydantic/FastAPI can serialise.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from backend import config
from src.model import build_model, load_model
from src.seg_model import load_seg_model


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return torch.device(config.resolve_device())


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    """Load the trained classifier once, falling back to ImageNet weights.

    If ``models/crack_classifier.pt`` is missing, we still return a
    working ResNet18 so the API can boot — predictions will be random,
    same as the old Streamlit fallback behaviour.
    """
    device = get_device()
    if config.MODEL_PATH.exists():
        return load_model(config.MODEL_PATH, device=device)
    model = build_model(pretrained=True)
    model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_seg_model() -> Optional[torch.nn.Module]:
    """Load the trained segmentation U-Net, or ``None`` if missing.

    Unlike the classifier we don't fall back to random weights for
    segmentation: a random U-Net outputs noise that would mislead the
    inspector. Callers should treat ``None`` as "segmentation feature
    unavailable" and degrade gracefully.
    """
    if not config.SEG_MODEL_PATH.exists():
        return None
    device = get_device()
    return load_seg_model(config.SEG_MODEL_PATH, device=device)


def has_trained_weights() -> bool:
    return config.MODEL_PATH.exists()


def has_seg_weights() -> bool:
    return config.SEG_MODEL_PATH.exists()


def load_image_bytes(payload: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(payload)).convert("RGB")


def resolve_sample(sample_name: str) -> Path:
    """Validate + resolve a sample filename, preventing path traversal."""
    if "/" in sample_name or "\\" in sample_name or sample_name.startswith("."):
        raise ValueError("Invalid sample name.")
    candidate = (config.SAMPLE_DIR / sample_name).resolve()
    try:
        candidate.relative_to(config.SAMPLE_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Sample path escapes SAMPLE_DIR.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Sample not found: {sample_name}")
    return candidate


def load_image_from_source(
    *, file_bytes: Optional[bytes], sample: Optional[str]
) -> Image.Image:
    """Get a PIL image from either an uploaded file or a sample name."""
    if file_bytes:
        return load_image_bytes(file_bytes)
    if sample:
        return Image.open(resolve_sample(sample)).convert("RGB")
    raise ValueError("Either an uploaded file or a sample name is required.")
