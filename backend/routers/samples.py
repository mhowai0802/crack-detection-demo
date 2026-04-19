"""List + serve the demo sample images bundled with the repo."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend import config
from backend.deps import resolve_sample
from backend.schemas import SamplesResponse

router = APIRouter(prefix="/samples", tags=["samples"])

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@router.get("", response_model=SamplesResponse)
def list_samples() -> SamplesResponse:
    if not config.SAMPLE_DIR.exists():
        return SamplesResponse(samples=[])
    names = sorted(
        p.name
        for p in config.SAMPLE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    return SamplesResponse(samples=names)


@router.get("/{name}")
def get_sample(name: str) -> FileResponse:
    try:
        path = resolve_sample(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path)
