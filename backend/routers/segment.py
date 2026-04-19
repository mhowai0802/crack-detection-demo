"""Segmentation endpoint.

``POST /api/segment`` accepts either an uploaded file or a bundled
``sample=<name>`` and returns a JSON object carrying:

- ``overlay_png_b64``: the original image with a red crack mask
  overlay (base64 PNG)
- ``mask_png_b64``: the raw binary mask (base64 PNG, 0 or 255)
- ``stats``: :class:`backend.schemas.SegStats` — shape / area / length
  / max-width metrics in pixel units

If the segmentation checkpoint is missing the endpoint returns 503 so
the frontend can degrade gracefully (e.g. hide the section rather than
show a scary error).
"""

from __future__ import annotations

import base64
import io
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.deps import get_device, get_seg_model, load_image_from_source
from backend.schemas import SegmentResponse, SegStats
from src.seg_infer import mask_stats, predict_mask
from src.seg_viz import overlay_mask

router = APIRouter(tags=["segment"])


async def _read_file(file: Optional[UploadFile]) -> Optional[bytes]:
    if file is None:
        return None
    data = await file.read()
    if not data:
        return None
    return data


def _png_bytes_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/segment", response_model=SegmentResponse)
async def run_segment(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(None),
    alpha: float = Query(0.5, ge=0.1, le=0.9),
    threshold: float = Query(0.5, ge=0.05, le=0.95),
) -> SegmentResponse:
    model = get_seg_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Segmentation model is not loaded. Run "
                "`python -m src.seg_train` to produce "
                "`models/crack_segmenter.pt`."
            ),
        )

    try:
        image = load_image_from_source(
            file_bytes=await _read_file(file), sample=sample
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    device = get_device()
    try:
        mask = predict_mask(image, model=model, device=device, threshold=threshold)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=500, detail=f"Segmentation failed: {exc}"
        ) from exc

    stats_dict = mask_stats(mask)
    stats = SegStats(
        crack_pixel_ratio=float(stats_dict["crack_pixel_ratio"]),
        num_components=int(stats_dict["num_components"]),
        area_px=int(stats_dict["area_px"]),
        length_px=int(stats_dict["length_px"]),
        max_width_px=float(stats_dict["max_width_px"]),
        image_height_px=int(stats_dict["image_height_px"]),
        image_width_px=int(stats_dict["image_width_px"]),
    )

    overlay = overlay_mask(image, mask, alpha=alpha)

    from PIL import Image as _PILImage

    mask_image = _PILImage.fromarray((mask * 255).astype("uint8"), mode="L")

    return SegmentResponse(
        overlay_png_b64=_png_bytes_b64(overlay),
        mask_png_b64=_png_bytes_b64(mask_image),
        stats=stats,
    )
