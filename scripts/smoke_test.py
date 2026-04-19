"""End-to-end smoke test for every backend module.

Run with ``python -m scripts.smoke_test`` from the repo root. The script
exercises each module with realistic inputs and prints a PASS / FAIL line
per check so a single command can verify the backend is healthy. It
includes one live HKBU GenAI call guarded by the ``HKBU_API_KEY`` env
var; that single network check is skipped automatically if the key is
absent.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_images"
MODEL_PATH = ROOT / "models" / "crack_classifier.pt"
SEG_MODEL_PATH = ROOT / "models" / "crack_segmenter.pt"
DATA_DIR = ROOT / "data"
DATA_SEG_DIR = ROOT / "data_seg"


Result = Tuple[str, bool, str]


def _check(name: str, fn: Callable[[], str]) -> Result:
    try:
        detail = fn() or "ok"
        return name, True, detail
    except Exception as exc:  # pragma: no cover - diagnostic only
        tb = traceback.format_exc(limit=3)
        return name, False, f"{type(exc).__name__}: {exc}\n{tb}"


def test_model_build() -> str:
    from src.model import CLASS_NAMES, build_model

    model = build_model(pretrained=False)
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2), f"unexpected output shape {out.shape}"
    assert CLASS_NAMES == ["No Crack", "Crack"]
    return f"forward -> {tuple(out.shape)}"


def test_model_load() -> str:
    from src.model import load_model

    if not MODEL_PATH.exists():
        return f"skipped (no checkpoint at {MODEL_PATH.relative_to(ROOT)})"
    model = load_model(MODEL_PATH, device="cpu")
    assert not model.training, "loaded model should be in eval mode"
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 2)
    return "checkpoint loaded + eval forward ok"


def test_dataset_transforms() -> str:
    from src.dataset import (
        CrackDataset,
        get_eval_transform,
        get_train_transform,
    )

    image = Image.new("RGB", (128, 128), color=(64, 128, 200))
    eval_tensor = get_eval_transform()(image)
    train_tensor = get_train_transform()(image)
    assert eval_tensor.shape == (3, 224, 224)
    assert train_tensor.shape == (3, 224, 224)

    dataset_detail = "dataset dir missing"
    if DATA_DIR.exists():
        ds = CrackDataset(DATA_DIR, transform=get_eval_transform())
        assert len(ds) > 0
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert label in (0, 1)
        dataset_detail = f"CrackDataset size={len(ds)}"
    return f"transforms 3x224x224; {dataset_detail}"


def test_predict_on_sample() -> str:
    from src.model import CLASS_NAMES, build_model, load_model
    from src.predict import predict

    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    assert samples, "no sample images available"
    img = Image.open(samples[0]).convert("RGB")

    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH, device="cpu")
    else:
        model = build_model(pretrained=False).eval()

    out = predict(img, model=model, device="cpu")
    assert set(out.keys()) >= {"label", "confidence", "probs", "class_index"}
    assert out["label"] in CLASS_NAMES
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["probs"]) == 2
    assert abs(sum(out["probs"]) - 1.0) < 1e-4
    return (
        f"{samples[0].name} -> {out['label']} "
        f"({out['confidence'] * 100:.1f}%)"
    )


def test_gradcam_pipeline() -> str:
    from src.gradcam import (
        compute_gradcam_map,
        compute_gradcam_overlay,
        dominant_quadrant,
    )
    from src.model import CLASS_NAMES, build_model, load_model

    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    assert samples, "no sample images available"
    img = Image.open(samples[0]).convert("RGB")

    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH, device="cpu")
    else:
        model = build_model(pretrained=False).eval()

    cam = compute_gradcam_map(
        img, model=model, class_index=CLASS_NAMES.index("Crack"), device="cpu"
    )
    assert isinstance(cam, np.ndarray) and cam.ndim == 2
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    quadrant = dominant_quadrant(cam)
    assert quadrant in {
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
        "centre-heavy",
        "uniform",
    }

    overlay = compute_gradcam_overlay(
        img,
        model=model,
        class_index=CLASS_NAMES.index("Crack"),
        device="cpu",
    )
    assert overlay.size == img.size
    assert overlay.mode == "RGB"

    assert dominant_quadrant(np.zeros((7, 7), dtype=np.float32)) == "uniform"
    tl = np.zeros((8, 8), dtype=np.float32)
    tl[:4, :4] = 1.0
    assert dominant_quadrant(tl) == "top-left"
    return f"cam {cam.shape} -> {quadrant}; overlay {overlay.size}"


def test_ai_prompts() -> str:
    from src.ai_prompts import (
        SYSTEM_CHAT_EN,
        SYSTEM_CHAT_ZH,
        SYSTEM_REPORT_EN,
        SYSTEM_REPORT_ZH,
        build_prediction_context,
    )

    for text in (
        SYSTEM_REPORT_ZH,
        SYSTEM_REPORT_EN,
        SYSTEM_CHAT_ZH,
        SYSTEM_CHAT_EN,
    ):
        assert isinstance(text, str) and len(text) > 20

    prediction = {
        "label": "Crack",
        "confidence": 0.9,
        "probs": [0.1, 0.9],
        "class_index": 1,
    }
    ctx = build_prediction_context(
        prediction, threshold=0.5, grad_cam_hint="bottom-right"
    )
    assert "CRACK DETECTED" in ctx
    assert "90.0%" in ctx
    assert "bottom-right" in ctx
    assert "50%" in ctx

    no_crack = {"label": "No Crack", "confidence": 0.8, "probs": [0.8, 0.2]}
    ctx_no = build_prediction_context(no_crack, threshold=0.5)
    assert "NO CRACK" in ctx_no
    assert "[Grad-CAM focus]" not in ctx_no
    return "ZH/EN prompts present; context formatting ok"


def test_llm_config() -> str:
    from src.llm import LLMConfigError, _require_api_key

    try:
        _require_api_key()
    except LLMConfigError as exc:
        return f"no API key detected ({exc})"
    return "HKBU_API_KEY present"


def test_llm_live_call() -> str:
    import os

    from src.llm import chat, chat_messages

    if not os.getenv("HKBU_API_KEY"):
        return "skipped (HKBU_API_KEY not set)"

    single = chat(
        system="You reply in one short word.",
        user="Reply with the single word PONG.",
        max_tokens=10,
        temperature=0.0,
    )
    assert isinstance(single, str) and single.strip()

    multi = chat_messages(
        [
            {"role": "system", "content": "You reply in one short word."},
            {"role": "user", "content": "Respond with the single word OK."},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    assert isinstance(multi, str) and multi.strip()
    return f"chat={single.strip()[:20]!r}; chat_messages={multi.strip()[:20]!r}"


def test_evaluate_module() -> str:
    import numpy as np

    from src.evaluate import evaluate_checkpoint, metrics_at_threshold

    if not MODEL_PATH.exists():
        return f"skipped (no checkpoint at {MODEL_PATH.relative_to(ROOT)})"
    if not DATA_DIR.exists():
        return f"skipped (no dataset at {DATA_DIR.relative_to(ROOT)})"

    result = evaluate_checkpoint(
        MODEL_PATH,
        DATA_DIR,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        split="test",
        batch_size=64,
        num_workers=0,
        device="cpu",
    )
    assert result.num_samples > 0
    assert result.split == "test"
    assert result.num_test == result.num_samples
    assert result.num_train > 0 and result.num_val > 0
    assert result.probs.shape == (result.num_samples, 2)
    assert result.targets.shape == (result.num_samples,)
    assert len(result.sample_paths) == result.num_samples
    assert np.all(result.probs >= -1e-4) and np.all(result.probs <= 1.0 + 1e-4)
    row_sums = result.probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)

    metrics = metrics_at_threshold(
        result.probs, result.targets, threshold=0.5
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert int(cm.sum()) == result.num_samples
    assert set(metrics["per_class"].keys()) == {"No Crack", "Crack"}

    high = metrics_at_threshold(result.probs, result.targets, 0.95)
    low = metrics_at_threshold(result.probs, result.targets, 0.05)
    assert high["confusion_matrix"].sum() == low["confusion_matrix"].sum()

    val_result = evaluate_checkpoint(
        MODEL_PATH,
        DATA_DIR,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        split="val",
        batch_size=64,
        num_workers=0,
        device="cpu",
    )
    assert val_result.split == "val"
    assert val_result.num_val == val_result.num_samples
    test_indices = {str(p) for p in result.sample_paths}
    val_indices = {str(p) for p in val_result.sample_paths}
    assert not (test_indices & val_indices), (
        "val and test slices must be disjoint"
    )

    return (
        f"test_size={result.num_samples} val_size={val_result.num_samples} "
        f"test_acc={metrics['accuracy'] * 100:.2f}% "
        f"crack_f1={metrics['per_class']['Crack']['f1']:.3f}"
    )


def test_seg_module() -> str:
    """Shape + stats sanity for the U-Net segmentation pipeline.

    Covers ``build_unet`` / ``load_seg_model`` / ``predict_mask`` /
    ``mask_stats`` / ``overlay_mask``. Uses a tiny random U-Net with
    ``base_channels=4`` to keep this fast (<1s on CPU); the trained
    checkpoint is loaded separately so both code paths are exercised.
    """
    from src.seg_infer import mask_stats, predict_mask
    from src.seg_model import (
        CLASS_NAMES_SEG,
        build_unet,
        count_parameters,
        load_seg_model,
    )
    from src.seg_viz import draw_mask_contours, overlay_mask

    assert CLASS_NAMES_SEG == ["background", "crack"]

    tiny = build_unet(base_channels=4)
    params = count_parameters(tiny)
    assert 10_000 < params < 1_000_000, f"tiny U-Net params out of range: {params}"
    with torch.no_grad():
        logits = tiny(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 1, 64, 64), f"unexpected logit shape {logits.shape}"

    img = Image.new("RGB", (96, 80), color=(120, 140, 160))
    mask = predict_mask(
        img, model=tiny.eval(), device="cpu", threshold=0.5, image_size=64
    )
    assert mask.shape == (80, 96), f"mask resized wrongly: {mask.shape}"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 1})

    synthetic = np.zeros((20, 40), dtype=np.uint8)
    synthetic[9:11, 5:35] = 1
    synthetic[5:15, 30:33] = 1
    stats = mask_stats(synthetic)
    expected_keys = {
        "crack_pixel_ratio",
        "num_components",
        "area_px",
        "length_px",
        "max_width_px",
        "image_height_px",
        "image_width_px",
    }
    assert expected_keys <= set(stats.keys())
    assert stats["image_height_px"] == 20
    assert stats["image_width_px"] == 40
    assert stats["area_px"] == int(synthetic.sum())
    assert stats["num_components"] == 1
    assert stats["max_width_px"] >= 2.0
    assert stats["length_px"] >= 20

    stats_mm = mask_stats(synthetic, px_per_mm=10.0)
    assert "max_width_mm" in stats_mm and stats_mm["max_width_mm"] > 0
    assert "length_mm" in stats_mm and stats_mm["length_mm"] > 0

    overlay = overlay_mask(img, mask, alpha=0.5)
    assert overlay.size == img.size and overlay.mode == "RGB"
    contoured = draw_mask_contours(img, mask)
    assert contoured.size == img.size

    empty_stats = mask_stats(np.zeros((10, 10), dtype=np.uint8))
    assert empty_stats["area_px"] == 0 and empty_stats["num_components"] == 0
    assert empty_stats["max_width_px"] == 0.0

    ckpt_detail = "skipped (no segmenter checkpoint)"
    if SEG_MODEL_PATH.exists():
        trained = load_seg_model(SEG_MODEL_PATH, device="cpu")
        assert not trained.training
        with torch.no_grad():
            out = trained(torch.zeros(1, 3, 128, 128))
        assert out.shape == (1, 1, 128, 128)
        ckpt_detail = f"trained params={count_parameters(trained):,}"

    return (
        f"tiny U-Net params={params:,}; synthetic stats "
        f"(cc={stats['num_components']}, len={stats['length_px']}, "
        f"max_w={stats['max_width_px']:.1f}); {ckpt_detail}"
    )


def test_segment_endpoint() -> str:
    """Hit ``POST /api/segment`` in-process via FastAPI ``TestClient``.

    Uses the real backend app so router wiring, schema validation, and
    base64 encoding all go through the production code path. Skipped if
    the segmenter checkpoint is missing (the endpoint correctly
    returns 503 in that case — we verify that too).
    """
    import base64

    from fastapi.testclient import TestClient

    from backend.main import app

    samples = sorted(SAMPLE_DIR.glob("*.jpg"))
    assert samples, "no sample images available"
    sample_name = samples[0].name

    with TestClient(app) as client:
        resp = client.post("/api/segment", params={"sample": sample_name})

        if not SEG_MODEL_PATH.exists():
            assert resp.status_code == 503, (
                f"expected 503 when seg checkpoint missing, got "
                f"{resp.status_code}: {resp.text}"
            )
            return "no segmenter checkpoint; endpoint correctly 503s"

        assert resp.status_code == 200, (
            f"unexpected status {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        assert {"overlay_png_b64", "mask_png_b64", "stats"} <= set(body.keys())

        overlay_bytes = base64.b64decode(body["overlay_png_b64"])
        mask_bytes = base64.b64decode(body["mask_png_b64"])
        assert overlay_bytes.startswith(b"\x89PNG"), "overlay is not a PNG"
        assert mask_bytes.startswith(b"\x89PNG"), "mask is not a PNG"

        stats = body["stats"]
        for key in (
            "crack_pixel_ratio",
            "num_components",
            "area_px",
            "length_px",
            "max_width_px",
            "image_height_px",
            "image_width_px",
        ):
            assert key in stats, f"stats missing {key}: {stats}"
        assert 0.0 <= stats["crack_pixel_ratio"] <= 1.0
        assert stats["image_height_px"] > 0 and stats["image_width_px"] > 0

        health = client.get("/api/health").json()
        assert health.get("has_seg_weights") is True, (
            f"health endpoint should report seg weights: {health}"
        )

    return (
        f"{sample_name} -> "
        f"coverage={stats['crack_pixel_ratio'] * 100:.2f}%, "
        f"components={stats['num_components']}, "
        f"overlay={len(overlay_bytes):,}B mask={len(mask_bytes):,}B"
    )


def test_prepare_data_script() -> str:
    import importlib

    module = importlib.import_module("scripts.prepare_data")
    for attr in ("PATCH_GRID", "CRACK_MIN_PIXELS", "POSITIVE_DIR", "NEGATIVE_DIR"):
        assert hasattr(module, attr), f"scripts.prepare_data missing {attr}"

    seg_module = importlib.import_module("scripts.prepare_seg_data")
    assert hasattr(seg_module, "prepare_deepcrack"), (
        "scripts.prepare_seg_data missing prepare_deepcrack"
    )
    return (
        "scripts.prepare_data importable; scripts.prepare_seg_data "
        "importable with prepare_deepcrack"
    )


def test_splits_module() -> str:
    from src.splits import (
        DEFAULT_SEED,
        DEFAULT_TEST_RATIO,
        DEFAULT_VAL_RATIO,
        split_sizes,
        three_way_split_indices,
    )

    n_train, n_val, n_test = split_sizes(1000, 0.15, 0.15)
    assert (n_train, n_val, n_test) == (700, 150, 150)
    assert n_train + n_val + n_test == 1000

    train, val, test = three_way_split_indices(
        1000,
        val_ratio=DEFAULT_VAL_RATIO,
        test_ratio=DEFAULT_TEST_RATIO,
        seed=DEFAULT_SEED,
    )
    assert (len(train), len(val), len(test)) == (700, 150, 150)
    all_idx = set(train) | set(val) | set(test)
    assert len(all_idx) == 1000, "splits must be disjoint and exhaustive"
    assert all_idx == set(range(1000))

    train2, val2, test2 = three_way_split_indices(
        1000, val_ratio=0.15, test_ratio=0.15, seed=DEFAULT_SEED
    )
    assert train == train2 and val == val2 and test == test2, (
        "three_way_split_indices must be deterministic for a given seed"
    )

    legacy_train, legacy_val, legacy_test = three_way_split_indices(
        1000, val_ratio=0.2, test_ratio=0.0, seed=DEFAULT_SEED
    )
    assert len(legacy_test) == 0
    assert len(legacy_train) == 800 and len(legacy_val) == 200

    return (
        f"70/15/15 split OK; deterministic; "
        f"legacy 80/20 reproduced (test_ratio=0)"
    )


CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("src.model.build_model", test_model_build),
    ("src.model.load_model", test_model_load),
    ("src.dataset transforms + CrackDataset", test_dataset_transforms),
    ("src.predict.predict", test_predict_on_sample),
    ("src.gradcam pipeline + dominant_quadrant", test_gradcam_pipeline),
    ("src.ai_prompts", test_ai_prompts),
    ("src.llm config (.env)", test_llm_config),
    ("src.llm live HKBU call", test_llm_live_call),
    ("src.splits three-way partition", test_splits_module),
    ("src.evaluate on current checkpoint", test_evaluate_module),
    ("scripts.prepare_data import", test_prepare_data_script),
    ("src.seg_model + seg_infer + seg_viz", test_seg_module),
    ("POST /api/segment (TestClient)", test_segment_endpoint),
]


def main() -> int:
    print(f"Running backend smoke tests from {ROOT}\n")
    results: List[Result] = []
    for label, fn in CHECKS:
        name, ok, detail = _check(label, fn)
        status = "PASS" if ok else "FAIL"
        first = detail.splitlines()[0] if detail else ""
        print(f"[{status}] {name} :: {first}")
        if not ok:
            rest = "\n".join(detail.splitlines()[1:])
            if rest.strip():
                print(rest)
        results.append((name, ok, detail))

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
