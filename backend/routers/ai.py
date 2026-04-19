"""HKBU GenAI-backed endpoints: inspection report + prediction chat."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas import (
    ChatRequest,
    ChatResponse,
    PredictionContext,
    ReportRequest,
    ReportResponse,
)
from src.ai_prompts import (
    SYSTEM_CHAT_EN,
    SYSTEM_CHAT_ZH,
    SYSTEM_REPORT_EN,
    SYSTEM_REPORT_ZH,
    build_prediction_context,
)
from src.llm import LLMConfigError, LLMRequestError, chat, chat_messages

router = APIRouter(prefix="/ai", tags=["ai"])


def _context_dict(pred: PredictionContext) -> dict:
    """Adapter: Pydantic PredictionContext -> the dict shape src.ai_prompts expects."""
    return {
        "label": pred.label,
        "probs": list(pred.probs),
        "confidence": pred.confidence,
    }


@router.post("/report", response_model=ReportResponse)
def generate_report(req: ReportRequest) -> ReportResponse:
    system = SYSTEM_REPORT_ZH if req.lang == "zh" else SYSTEM_REPORT_EN
    header = req.header.model_dump() if req.header else None
    seg_stats = req.seg_stats.model_dump() if req.seg_stats else None
    user = build_prediction_context(
        _context_dict(req.prediction),
        req.threshold,
        req.grad_cam_hint,
        header=header,
        seg_stats=seg_stats,
    )
    try:
        text = chat(
            system=system,
            user=user,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ReportResponse(text=text)


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty.")

    base_system = SYSTEM_CHAT_ZH if req.lang == "zh" else SYSTEM_CHAT_EN
    pred_context = build_prediction_context(
        _context_dict(req.prediction), req.threshold, req.grad_cam_hint
    )
    system_prompt = f"{base_system}\n\n{pred_context}"

    payload = [{"role": "system", "content": system_prompt}]
    payload.extend(msg.model_dump() for msg in req.messages)

    try:
        reply = chat_messages(
            payload,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except LLMConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(reply=reply)
