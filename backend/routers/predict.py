"""Predict + Grad-CAM endpoints.

Both accept either a multipart file upload (``file``) or a ``sample=<name>``
query parameter so the frontend can reference bundled demo images without
a round-trip upload.
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from backend.deps import (
    get_device,
    get_model,
    load_image_from_source,
)
from backend.schemas import PredictionResponse
from src.constants import CLASS_NAMES, CRACK_INDEX
from src.gradcam import (
    compute_gradcam_map,
    compute_gradcam_overlay,
    dominant_quadrant,
)
from src.predict import predict

router = APIRouter(tags=["predict"])


async def _read_file(file: Optional[UploadFile]) -> Optional[bytes]:
    if file is None:
        return None
    data = await file.read()
    if not data:
        return None
    return data


@router.post("/predict", response_model=PredictionResponse)
async def run_predict(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(
        None, description="Optional bundled sample image name to predict on."
    ),
) -> PredictionResponse:
    try:
        image = load_image_from_source(
            file_bytes=await _read_file(file), sample=sample
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = get_model()
    device = get_device()
    prediction = predict(image, model=model, device=device)

    focus: Optional[str] = None
    try:
        heatmap = compute_gradcam_map(
            image, model=model, class_index=CRACK_INDEX, device=device
        )
        focus = dominant_quadrant(heatmap)
    except Exception:  # pragma: no cover - focus hint is best effort
        focus = None

    return PredictionResponse(
        label=prediction["label"],
        class_index=int(prediction["class_index"]),
        confidence=float(prediction["confidence"]),
        probs=[float(p) for p in prediction["probs"]],
        class_names=list(CLASS_NAMES),
        gradcam_focus=focus,
    )


@router.post(
    "/gradcam",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def run_gradcam(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(None),
    alpha: float = Query(0.45, ge=0.05, le=0.95),
) -> Response:
    try:
        image = load_image_from_source(
            file_bytes=await _read_file(file), sample=sample
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = get_model()
    device = get_device()
    try:
        overlay = compute_gradcam_overlay(
            image, model=model, class_index=CRACK_INDEX, device=device, alpha=alpha
        )
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(
            status_code=500, detail=f"Grad-CAM failed: {exc}"
        ) from exc

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
