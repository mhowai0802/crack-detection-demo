"""Liveness + config probe."""

from __future__ import annotations

from fastapi import APIRouter

from backend.deps import get_device, has_seg_weights, has_trained_weights
from backend.schemas import HealthResponse
from src.constants import CLASS_NAMES

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        has_trained_weights=has_trained_weights(),
        has_seg_weights=has_seg_weights(),
        device=str(get_device()),
        class_names=list(CLASS_NAMES),
    )
