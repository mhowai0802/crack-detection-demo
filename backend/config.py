"""Backend configuration and on-disk paths.

Environment variables (all optional, defaults in parens):
    BACKEND_HOST      — uvicorn bind host (``0.0.0.0``)
    BACKEND_PORT      — uvicorn bind port (``8000``)
    BACKEND_CORS_ORIGINS  — comma-separated CORS origins (``*``)
    MODEL_PATH        — override the checkpoint path
    SAMPLE_DIR        — override the sample-images directory
    DATA_DIR          — override the dataset root (``data/``)
    TORCH_DEVICE      — ``cpu`` | ``cuda`` | ``auto`` (``auto``)

``.env`` at the repo root is loaded via ``python-dotenv`` so shared
secrets (like ``HKBU_API_KEY``) stay in one file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


MODEL_PATH: Path = _path_from_env("MODEL_PATH", ROOT / "models" / "crack_classifier.pt")
SEG_MODEL_PATH: Path = _path_from_env(
    "SEG_MODEL_PATH", ROOT / "models" / "crack_segmenter.pt"
)
SAMPLE_DIR: Path = _path_from_env("SAMPLE_DIR", ROOT / "sample_images")
DATA_DIR: Path = _path_from_env("DATA_DIR", ROOT / "data")

HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT: int = int(os.getenv("BACKEND_PORT", "8000"))


def _split_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


CORS_ORIGINS: List[str] = _split_csv(os.getenv("BACKEND_CORS_ORIGINS", "*"))


def resolve_device() -> str:
    """Pick a torch device string: honour ``TORCH_DEVICE`` or auto-detect CUDA."""
    requested = os.getenv("TORCH_DEVICE", "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
