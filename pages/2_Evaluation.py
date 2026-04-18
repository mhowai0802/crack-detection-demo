"""Model evaluation dashboard (Streamlit page).

Loads ``models/crack_classifier.pt`` and re-runs it on the seeded 80/20
validation split from training. Shows:

- Dataset + split summary.
- Headline accuracy.
- A decision-threshold slider that recomputes confusion matrix +
  per-class precision / recall / F1 from stored probabilities without a
  second forward pass.
- 2x2 confusion matrix + classification report table.
- A Grad-CAM contact sheet of the worst misclassified samples so you
  can see *why* the model fails on specific patches.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.evaluate import (
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    EvaluationResult,
    evaluate_checkpoint,
    metrics_at_threshold,
)
from src.model import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "crack_classifier.pt"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="模型評估 - 混凝土裂縫偵測",
    page_icon=":construction:",
    layout="wide",
)

st.title("模型評估 / Model Evaluation")
st.caption(
    "用同訓練時一樣嘅 seeded 80 / 20 split,再跑一次 validation,"
    "即時睇番 checkpoint 嘅 accuracy、每類嘅 precision / recall / F1、"
    "同 confusion matrix。"
)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

missing: List[str] = []
if not MODEL_PATH.exists():
    missing.append(
        f"`{MODEL_PATH.relative_to(ROOT)}` — 訓練一次先: "
        "`python -m src.train --data-dir data/`"
    )
if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    missing.append(
        f"`{DATA_DIR.relative_to(ROOT)}/` 入面要有 `Positive/` 同 `Negative/` "
        "兩個 folder: `python -m scripts.prepare_data` 會幫你整好。"
    )
if missing:
    st.error("無辦法做 evaluation,因為:\n\n- " + "\n- ".join(missing))
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

sidebar = st.sidebar
sidebar.title(":material/analytics: Evaluation")
sidebar.caption("Re-score `models/crack_classifier.pt` on the held-out split.")
sidebar.divider()

with sidebar.expander(":material/dataset: Validation split", expanded=True):
    val_ratio = st.slider(
        "Validation ratio",
        min_value=0.1,
        max_value=0.4,
        value=DEFAULT_VAL_RATIO,
        step=0.05,
        help="一定要同 training 時一樣(預設 0.2),唔係數字就對唔到上。",
    )
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=999_999,
        value=DEFAULT_SEED,
        step=1,
        help="split 用嘅 RNG seed,要同 training 時一樣。",
    )
    st.caption(
        "Changing either field invalidates the cache and triggers a fresh run."
    )

with sidebar.expander(":material/bolt: Runtime", expanded=True):
    batch_size = st.select_slider(
        "Batch size", options=[16, 32, 64, 128], value=64
    )
    rerun = st.button(
        "Re-run evaluation",
        use_container_width=True,
        icon=":material/refresh:",
    )

with sidebar.expander(":material/tune: Decision rule", expanded=True):
    threshold = st.slider(
        "Decision threshold — P(Crack) ≥ ?",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.01,
        help=(
            "Argmax (即係 0.5) 係 training 報告用嘅準則。"
            "調高會減少 false positive,調低會減少 false negative。"
        ),
    )
    st.caption(f"Flag as **Crack** when P(Crack) ≥ **{threshold:.2f}**.")

sidebar.divider()
sidebar.caption(
    "Threshold 唔會再跑一次 model —— metrics 直接喺 cached probabilities "
    "入面即時計出嚟。"
)


# ---------------------------------------------------------------------------
# Cached evaluation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="跑緊 validation set...")
def _run_eval_cached(
    model_mtime: float,
    data_dir_str: str,
    val_ratio: float,
    seed: int,
    batch_size: int,
) -> EvaluationResult:
    """Wrapper so Streamlit caches by split config + checkpoint timestamp."""
    return evaluate_checkpoint(
        MODEL_PATH,
        data_dir_str,
        val_ratio=val_ratio,
        seed=seed,
        batch_size=batch_size,
        num_workers=0,
        device="cpu",
    )


if rerun:
    _run_eval_cached.clear()

result = _run_eval_cached(
    model_mtime=MODEL_PATH.stat().st_mtime,
    data_dir_str=str(DATA_DIR),
    val_ratio=val_ratio,
    seed=seed,
    batch_size=batch_size,
)
metrics = metrics_at_threshold(result.probs, result.targets, threshold=threshold)


# ---------------------------------------------------------------------------
# Split summary
# ---------------------------------------------------------------------------

st.subheader("1. Dataset 同 split")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total patches", f"{result.num_train + result.num_val}")
col_b.metric("Train", f"{result.num_train}")
col_c.metric("Validation", f"{result.num_val}")
col_d.metric(
    "Val class balance",
    f"Crack {int((result.targets == CLASS_NAMES.index('Crack')).sum())} / "
    f"No Crack {int((result.targets == CLASS_NAMES.index('No Crack')).sum())}",
)


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

st.subheader("2. 總體準確率 / Headline metrics")

crack_idx = CLASS_NAMES.index("Crack")
per_class = metrics["per_class"]
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
col2.metric("Crack recall", f"{per_class['Crack']['recall'] * 100:.2f}%",
            help="真裂縫入面,有幾多個捉到? 對施工安全最重要。")
col3.metric("Crack precision", f"{per_class['Crack']['precision'] * 100:.2f}%",
            help="模型話係裂縫果堆,入面有幾多個真係裂縫?")

st.caption(
    f"Threshold = {threshold:.2f} · Validation size = {result.num_samples} · "
    f"Seed = {seed} · Val ratio = {val_ratio:.2f}"
)


# ---------------------------------------------------------------------------
# Per-class table
# ---------------------------------------------------------------------------

st.subheader("3. 每類詳細數字 / Classification report")
report_df = pd.DataFrame(
    {
        name: {
            "precision": row["precision"],
            "recall": row["recall"],
            "f1-score": row["f1"],
            "support": row["support"],
        }
        for name, row in per_class.items()
    }
).T
report_df.loc["accuracy"] = [
    metrics["accuracy"],
    metrics["accuracy"],
    metrics["accuracy"],
    result.num_samples,
]

def _fmt(v: float) -> str:
    return f"{v:.4f}" if isinstance(v, float) else f"{int(v)}"

st.dataframe(
    report_df.style.format(
        {col: _fmt for col in ["precision", "recall", "f1-score", "support"]}
    ),
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

st.subheader("4. Confusion matrix")
cm: np.ndarray = metrics["confusion_matrix"]
cm_df = pd.DataFrame(
    cm,
    index=[f"true: {c}" for c in CLASS_NAMES],
    columns=[f"pred: {c}" for c in CLASS_NAMES],
)

col_cm_table, col_cm_chart = st.columns([1, 1])
with col_cm_table:
    st.dataframe(cm_df, use_container_width=True)
    tn = int(cm[CLASS_NAMES.index("No Crack"), CLASS_NAMES.index("No Crack")])
    fp = int(cm[CLASS_NAMES.index("No Crack"), crack_idx])
    fn = int(cm[crack_idx, CLASS_NAMES.index("No Crack")])
    tp = int(cm[crack_idx, crack_idx])
    st.caption(
        f"TP={tp} · FP={fp} · FN={fn} · TN={tn}. "
        "FN = 漏報裂縫(最危險),FP = 誤報裂縫(浪費人力)。"
    )

with col_cm_chart:
    cm_long = cm_df.reset_index().melt(
        id_vars="index", var_name="pred", value_name="count"
    )
    cm_long.rename(columns={"index": "true"}, inplace=True)
    st.bar_chart(cm_long, x="pred", y="count", color="true")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

st.subheader("5. Threshold sweep")
st.caption(
    "掃一次 0.05 → 0.95,睇下應唔應該用其他 threshold。當前嘅 threshold 會"
    "喺圖上面以紅線標出。"
)
thresholds = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds:
    m = metrics_at_threshold(result.probs, result.targets, float(t))
    rows.append(
        {
            "threshold": float(t),
            "accuracy": m["accuracy"],
            "crack_recall": m["per_class"]["Crack"]["recall"],
            "crack_precision": m["per_class"]["Crack"]["precision"],
            "crack_f1": m["per_class"]["Crack"]["f1"],
        }
    )
sweep_df = pd.DataFrame(rows).set_index("threshold")
st.line_chart(sweep_df)


# ---------------------------------------------------------------------------
# Misclassified samples
# ---------------------------------------------------------------------------

st.subheader("6. 錯得最離譜嘅樣本 / Worst misclassifications")
crack_probs = result.probs[:, crack_idx]
preds = metrics["preds"]
wrong_mask = preds != result.targets
wrong_indices = np.where(wrong_mask)[0]


def _confidence_distance(i: int) -> float:
    """How far the wrong prediction was past the threshold, larger = worse."""
    return abs(float(crack_probs[i]) - threshold)


if wrong_indices.size == 0:
    st.success(
        f"Threshold {threshold:.2f} 之下,驗證集 {result.num_samples} 張全部"
        " 答啱。唔使睇失敗例子。"
    )
else:
    ranked: List[Tuple[int, float]] = sorted(
        ((int(i), _confidence_distance(int(i))) for i in wrong_indices),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_k = st.slider(
        "顯示幾多張?", min_value=4, max_value=min(16, len(ranked)),
        value=min(8, len(ranked)), step=1
    )

    cols_per_row = 4
    for row_start in range(0, top_k, cols_per_row):
        cols = st.columns(cols_per_row)
        for offset, col in enumerate(cols):
            i = row_start + offset
            if i >= top_k:
                break
            idx, _dist = ranked[i]
            path = result.sample_paths[idx]
            try:
                img = Image.open(path).convert("RGB")
            except Exception as exc:
                col.warning(f"讀唔到 {path.name}: {exc}")
                continue
            true_label = CLASS_NAMES[int(result.targets[idx])]
            pred_label = CLASS_NAMES[int(preds[idx])]
            p_crack = float(crack_probs[idx])
            col.image(img, use_column_width=True)
            col.caption(
                f"**{path.name}**\n\n"
                f"真實: `{true_label}`\n\n"
                f"預測: `{pred_label}` · P(Crack)={p_crack * 100:.1f}%"
            )

    st.caption(
        f"總共有 {wrong_indices.size} 個錯誤 (accuracy = "
        f"{metrics['accuracy'] * 100:.2f}%)。"
        "上面排序係按照「預測信心離 threshold 有幾遠」— 信心越大越錯就越值得睇。"
    )
