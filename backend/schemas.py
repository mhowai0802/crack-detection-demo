"""Pydantic request / response schemas for the FastAPI routers."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class name (argmax).")
    class_index: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    probs: List[float] = Field(..., description="Probabilities ordered by class_names.")
    class_names: List[str]
    gradcam_focus: Optional[str] = Field(
        None,
        description='One of "top-left", "top-right", "bottom-left", '
        '"bottom-right", "centre-heavy", "uniform".',
    )


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


class SamplesResponse(BaseModel):
    samples: List[str]


# ---------------------------------------------------------------------------
# AI (LLM)
# ---------------------------------------------------------------------------

Language = Literal["zh", "en"]


class PredictionContext(BaseModel):
    label: str
    probs: List[float]
    confidence: float


class ReportHeader(BaseModel):
    """Administrative header that the LLM renders at the top of the report.

    All fields are optional. Missing fields are simply omitted from the
    rendered header block so the inspector can fill in as much or as
    little context as they have on hand.
    """

    report_id: Optional[str] = Field(
        None, description="E.g. CRK-2026-0419-001 — free-form string."
    )
    inspection_date: Optional[str] = Field(
        None, description='Free-form date string, typically "YYYY-MM-DD".'
    )
    location: Optional[str] = Field(
        None, description='Site / address / unit, e.g. "Site B – Podium lvl 3".'
    )
    element: Optional[str] = Field(
        None,
        description=(
            "Structural element: Slab / Beam / Column / Wall / "
            "Staircase / Other."
        ),
    )
    inspector: Optional[str] = Field(
        None, description="Inspector name or staff id."
    )


class SegStats(BaseModel):
    """Shape statistics for a predicted crack mask.

    Values are in pixel units because the model cannot recover an
    absolute mm scale from a single uncalibrated photo. Callers who do
    have a scale reference (e.g. a ruler in the frame) can convert in
    the frontend before rendering or passing into the LLM prompt.
    """

    crack_pixel_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Crack pixels / total pixels."
    )
    num_components: int = Field(
        ..., ge=0, description="Disjoint crack regions (8-connectivity)."
    )
    area_px: int = Field(..., ge=0, description="Total number of crack pixels.")
    length_px: int = Field(
        ..., ge=0, description="Morphological skeleton pixel count."
    )
    max_width_px: float = Field(
        ..., ge=0.0, description="2 * max distance-to-background."
    )
    image_height_px: int = Field(..., gt=0)
    image_width_px: int = Field(..., gt=0)


class ReportRequest(BaseModel):
    prediction: PredictionContext
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    lang: Language = "zh"
    grad_cam_hint: Optional[str] = None
    header: Optional[ReportHeader] = None
    seg_stats: Optional[SegStats] = None
    max_tokens: int = Field(
        900,
        ge=32,
        le=2048,
        description=(
            "Must fit the 6-section HK compliance report; "
            "raising this costs more tokens but avoids truncation."
        ),
    )
    temperature: float = Field(0.3, ge=0.0, le=2.0)


class ReportResponse(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ..., description="Prior user/assistant turns (no system role needed)."
    )
    prediction: PredictionContext
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    lang: Language = "zh"
    grad_cam_hint: Optional[str] = None
    max_tokens: int = Field(350, ge=32, le=2048)
    temperature: float = Field(0.5, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluateRequest(BaseModel):
    val_ratio: float = Field(0.15, ge=0.0, le=0.5)
    test_ratio: float = Field(0.15, ge=0.0, le=0.5)
    seed: int = Field(42, ge=0)
    batch_size: int = Field(64, ge=1, le=512)
    split: Literal["val", "test"] = Field(
        "test",
        description=(
            "Which held-out slice to score. 'test' is the unbiased "
            "number; 'val' replays the checkpoint-selection slice."
        ),
    )


class EvaluateResponse(BaseModel):
    probs: List[List[float]]
    targets: List[int]
    sample_paths: List[str]
    class_names: List[str]
    split: Literal["val", "test"]
    num_train: int
    num_val: int
    num_test: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    has_trained_weights: bool
    has_seg_weights: bool = False
    device: str
    class_names: List[str]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class SegmentResponse(BaseModel):
    """Response from ``POST /api/segment``.

    The overlay PNG is base64-encoded so the response stays a single
    JSON object (simpler for the Streamlit client than multipart /
    separate endpoints for image vs stats).
    """

    overlay_png_b64: str = Field(
        ..., description="Base64-encoded PNG: image blended with red crack mask."
    )
    mask_png_b64: str = Field(
        ..., description="Base64-encoded PNG: raw binary mask (0 or 255)."
    )
    stats: SegStats
