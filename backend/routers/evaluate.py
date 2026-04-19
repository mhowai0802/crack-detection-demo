"""Evaluation endpoint + dataset image server for the Evaluation page.

Running a full forward pass over the held-out split is expensive, so
we wrap it with an in-process LRU cache keyed by the checkpoint mtime +
split config (ratios, seed, batch size, which slice). Threshold-
dependent metrics are derived frontend-side from the returned
probability matrix.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend import config
from backend.schemas import EvaluateRequest, EvaluateResponse
from src.constants import CLASS_NAMES
from src.evaluate import EvaluationResult, SplitName, evaluate_checkpoint

router = APIRouter(tags=["evaluate"])


@lru_cache(maxsize=8)
def _cached_evaluate(
    model_mtime: float,
    data_dir: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    split: SplitName,
) -> EvaluationResult:
    return evaluate_checkpoint(
        config.MODEL_PATH,
        data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        device="cpu",
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if not config.MODEL_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Checkpoint not found at {config.MODEL_PATH}. Train with "
                "`python -m src.train --data-dir data/`."
            ),
        )
    if not config.DATA_DIR.exists() or not any(config.DATA_DIR.iterdir()):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset not found under {config.DATA_DIR}. Run "
                "`python -m scripts.prepare_data` first."
            ),
        )
    if req.split == "test" and req.test_ratio <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "split='test' requires test_ratio > 0. Pass a positive "
                "test_ratio or switch split='val'."
            ),
        )

    try:
        result = _cached_evaluate(
            model_mtime=config.MODEL_PATH.stat().st_mtime,
            data_dir=str(config.DATA_DIR),
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
            seed=req.seed,
            batch_size=req.batch_size,
            split=req.split,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvaluateResponse(
        probs=result.probs.tolist(),
        targets=result.targets.astype(int).tolist(),
        sample_paths=[str(p) for p in result.sample_paths],
        class_names=list(CLASS_NAMES),
        split=result.split,
        num_train=result.num_train,
        num_val=result.num_val,
        num_test=result.num_test,
    )


@router.get("/dataset-image")
def dataset_image(path: str = Query(..., description="Absolute path returned by /evaluate.")) -> FileResponse:
    """Serve a file from within ``DATA_DIR`` given its absolute path.

    Prevents traversal by requiring the resolved target to live inside
    the configured data root.
    """
    try:
        target = Path(path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad path: {exc}") from exc

    data_root = config.DATA_DIR.resolve()
    try:
        target.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path outside DATA_DIR.") from exc

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Not found: {target.name}")
    return FileResponse(target)
