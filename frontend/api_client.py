"""HTTP client wrapping the FastAPI backend.

The backend URL defaults to ``http://localhost:8000`` and can be
overridden via the ``BACKEND_URL`` environment variable (or a ``.env``
file at the repo root). Every helper raises :class:`BackendError` with
a human-readable message so the Streamlit pages can show red banners
instead of crashing.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api"
DEFAULT_TIMEOUT = 60.0


class BackendError(RuntimeError):
    """Raised when the backend is unreachable or returns a non-2xx response."""


def _url(path: str) -> str:
    return f"{BASE_URL}{API_PREFIX}{path}"


def _handle(resp: requests.Response) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise BackendError(f"{resp.status_code}: {detail}")
    return resp.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthInfo:
    status: str
    has_trained_weights: bool
    has_seg_weights: bool
    device: str
    class_names: List[str]


def health() -> HealthInfo:
    try:
        data = _handle(requests.get(_url("/health"), timeout=5.0))
    except requests.RequestException as exc:
        raise BackendError(f"Cannot reach backend at {BASE_URL}: {exc}") from exc
    return HealthInfo(
        status=data["status"],
        has_trained_weights=bool(data["has_trained_weights"]),
        has_seg_weights=bool(data.get("has_seg_weights", False)),
        device=str(data["device"]),
        class_names=list(data["class_names"]),
    )


def list_samples() -> List[str]:
    try:
        data = _handle(requests.get(_url("/samples"), timeout=DEFAULT_TIMEOUT))
    except requests.RequestException as exc:
        raise BackendError(f"Failed to list samples: {exc}") from exc
    return list(data.get("samples", []))


def sample_image_bytes(name: str) -> bytes:
    try:
        resp = requests.get(_url(f"/samples/{name}"), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"Failed to fetch sample '{name}': {exc}") from exc
    if resp.status_code >= 400:
        raise BackendError(f"{resp.status_code}: {resp.text}")
    return resp.content


def predict_sample(name: str) -> Dict[str, Any]:
    return _predict(sample=name, file=None, filename=None)


def predict_upload(data: bytes, filename: str = "upload.png") -> Dict[str, Any]:
    return _predict(sample=None, file=data, filename=filename)


def _predict(
    *, sample: Optional[str], file: Optional[bytes], filename: Optional[str]
) -> Dict[str, Any]:
    params: Dict[str, str] = {}
    files = None
    if sample:
        params["sample"] = sample
    elif file is not None:
        files = {"file": (filename or "upload.png", file, "application/octet-stream")}
    else:
        raise ValueError("predict requires either a sample name or file bytes.")

    try:
        resp = requests.post(
            _url("/predict"), params=params, files=files, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as exc:
        raise BackendError(f"Predict request failed: {exc}") from exc
    return _handle(resp)


def gradcam_sample(name: str, *, alpha: float) -> bytes:
    return _gradcam(sample=name, file=None, filename=None, alpha=alpha)


def gradcam_upload(
    data: bytes, filename: str = "upload.png", *, alpha: float
) -> bytes:
    return _gradcam(sample=None, file=data, filename=filename, alpha=alpha)


def _gradcam(
    *,
    sample: Optional[str],
    file: Optional[bytes],
    filename: Optional[str],
    alpha: float,
) -> bytes:
    params: Dict[str, Any] = {"alpha": alpha}
    files = None
    if sample:
        params["sample"] = sample
    elif file is not None:
        files = {"file": (filename or "upload.png", file, "application/octet-stream")}
    else:
        raise ValueError("gradcam requires either a sample name or file bytes.")

    try:
        resp = requests.post(
            _url("/gradcam"), params=params, files=files, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as exc:
        raise BackendError(f"Grad-CAM request failed: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise BackendError(f"{resp.status_code}: {detail}")
    return resp.content


def segment_sample(name: str) -> Tuple[bytes, bytes, Dict[str, Any]]:
    return _segment(sample=name, file=None, filename=None)


def segment_upload(
    data: bytes, filename: str = "upload.png"
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    return _segment(sample=None, file=data, filename=filename)


def _segment(
    *,
    sample: Optional[str],
    file: Optional[bytes],
    filename: Optional[str],
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """Call ``/api/segment`` and return ``(overlay_png, mask_png, stats)``.

    Returns the overlay image bytes + raw mask bytes + stats dict. The
    backend sends both PNGs as base64 inside a JSON payload so we decode
    here and hide that transport detail from the UI layer.
    """
    params: Dict[str, str] = {}
    files = None
    if sample:
        params["sample"] = sample
    elif file is not None:
        files = {
            "file": (filename or "upload.png", file, "application/octet-stream")
        }
    else:
        raise ValueError("segment requires either a sample name or file bytes.")

    try:
        resp = requests.post(
            _url("/segment"), params=params, files=files, timeout=DEFAULT_TIMEOUT
        )
    except requests.RequestException as exc:
        raise BackendError(f"Segment request failed: {exc}") from exc
    data = _handle(resp)
    overlay = base64.b64decode(data["overlay_png_b64"])
    mask = base64.b64decode(data["mask_png_b64"])
    stats = dict(data["stats"])
    return overlay, mask, stats


def ai_report(
    prediction: Mapping[str, Any],
    *,
    threshold: float,
    lang: str,
    grad_cam_hint: Optional[str],
    header: Optional[Mapping[str, Optional[str]]] = None,
    seg_stats: Optional[Mapping[str, Any]] = None,
) -> str:
    body: Dict[str, Any] = {
        "prediction": {
            "label": prediction["label"],
            "probs": list(prediction["probs"]),
            "confidence": float(prediction["confidence"]),
        },
        "threshold": threshold,
        "lang": lang,
        "grad_cam_hint": grad_cam_hint,
    }
    if header:
        cleaned = {
            k: (v.strip() if isinstance(v, str) else v)
            for k, v in header.items()
            if v not in (None, "")
        }
        if cleaned:
            body["header"] = cleaned
    if seg_stats:
        body["seg_stats"] = dict(seg_stats)
    try:
        resp = requests.post(_url("/ai/report"), json=body, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"AI report request failed: {exc}") from exc
    return _handle(resp)["text"]


def ai_chat(
    messages: List[Mapping[str, str]],
    prediction: Mapping[str, Any],
    *,
    threshold: float,
    lang: str,
    grad_cam_hint: Optional[str],
) -> str:
    body = {
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ],
        "prediction": {
            "label": prediction["label"],
            "probs": list(prediction["probs"]),
            "confidence": float(prediction["confidence"]),
        },
        "threshold": threshold,
        "lang": lang,
        "grad_cam_hint": grad_cam_hint,
    }
    try:
        resp = requests.post(_url("/ai/chat"), json=body, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise BackendError(f"AI chat request failed: {exc}") from exc
    return _handle(resp)["reply"]


def evaluate(
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    split: str = "test",
) -> Dict[str, Any]:
    body = {
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "batch_size": batch_size,
        "split": split,
    }
    try:
        resp = requests.post(
            _url("/evaluate"), json=body, timeout=600.0
        )
    except requests.RequestException as exc:
        raise BackendError(f"Evaluate request failed: {exc}") from exc
    return _handle(resp)


def dataset_image_bytes(path: str) -> bytes:
    try:
        resp = requests.get(
            _url("/dataset-image"),
            params={"path": path},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise BackendError(f"Dataset image fetch failed: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise BackendError(f"{resp.status_code}: {detail}")
    return resp.content
