"""Model evaluation dashboard (Streamlit page).

Thin HTTP client over ``POST /api/evaluate``. The backend does the heavy
forward pass + returns a raw probability matrix; threshold-dependent
metrics (confusion matrix, precision / recall / F1) are computed here
so the threshold slider stays snappy without a second network call.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from frontend import api_client
from frontend.api_client import BackendError
from src.constants import CLASS_NAMES, CRACK_INDEX
from src.metrics import metrics_at_threshold, point_on_roc, roc_curve_points
from src.splits import DEFAULT_SEED, DEFAULT_TEST_RATIO, DEFAULT_VAL_RATIO

st.set_page_config(
    page_title="模型評估 - 混凝土裂縫偵測",
    page_icon=":construction:",
    layout="wide",
)

st.title("模型評估 / Model Evaluation")
st.caption(
    "用同訓練時一樣嘅 seeded **train / val / test** split (預設 70 / 15 / 15),"
    "由 backend 跑揀到嗰 slice,再喺前端即時計返 accuracy、per-class "
    "precision / recall / F1、同 confusion matrix。"
)

st.info(
    "**Dataset · Özgenel Concrete Crack Images for Classification (CCIC)**  \n"
    "本頁嘅數字全部由呢個 dataset 計出嚟:\n\n"
    "- **作者 / Source:** Özgenel, Ç.F. (2019). *Concrete Crack Images for "
    "Classification* [Data set]. Mendeley Data, v2.  \n"
    "  [`doi:10.17632/5y9wdsg2zt.2`](https://doi.org/10.17632/5y9wdsg2zt.2) · "
    "METU (Middle East Technical University, Ankara) 校園 facade 拍攝\n"
    "- **License:** CC-BY-4.0 — attribution required, free to reuse\n"
    "- **Demo 實際用到嘅 subset:** 3,200 張 `224×224` patch "
    "(HuggingFace `Vizuara/concrete-crack-dataset` 嘅 800 張相 × 2×2 patch,"
    "mask 判斷 crack / no crack),由 `scripts/prepare_data.py` 生成。\n"
    "- **Split:** `src.splits.three_way_split_indices(seed=42)` 做 70 / 15 / 15 "
    "train / val / test,training 同 evaluation 共用同一個 seed,所以保證 test "
    "slice 完全冇入過訓練。\n"
    "- **Important caveat for HK deployment:** 全部相都係 Ankara 校園乾淨 "
    "close-up,**唔係** HK MBIS 嘅 70s-80s 瓷磚外牆;呢個 slice 嘅 accuracy "
    "應該視為 *upper bound under in-distribution*,真地盤相會差明顯一截。",
    icon=":material/dataset:",
)
st.caption(
    "完整 dataset provenance(包括 DeepCrack segmentation dataset)見 repo "
    "入面嘅 `data/DATASETS.md` · 本 demo 冇 redistribute 任何相,"
    "只 redistribute training script。"
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE = 64
EVAL_SPLIT = "test"

val_ratio = DEFAULT_VAL_RATIO
test_ratio = DEFAULT_TEST_RATIO
seed = DEFAULT_SEED
batch_size = EVAL_BATCH_SIZE

sidebar = st.sidebar
sidebar.title(":material/analytics: Evaluation")
sidebar.caption(
    f"Scoring the held-out **test** slice "
    f"({int(DEFAULT_TEST_RATIO * 100)}%, seed={DEFAULT_SEED}). "
    "Probabilities are cached; only the threshold below re-computes metrics live."
)

threshold = sidebar.slider(
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
sidebar.caption(f"Flag as **Crack** when P(Crack) ≥ **{threshold:.2f}**.")


# ---------------------------------------------------------------------------
# Cached evaluation call
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Backend 跑緊 evaluation...")
def _run_eval_cached(
    val_ratio: float,
    test_ratio: float,
    seed: int,
    batch_size: int,
    split: str,
) -> dict:
    return api_client.evaluate(val_ratio, test_ratio, seed, batch_size, split)


try:
    raw = _run_eval_cached(
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        batch_size=batch_size,
        split=EVAL_SPLIT,
    )
except BackendError as exc:
    st.error(f"Backend refused to evaluate: {exc}")
    st.stop()

probs = np.asarray(raw["probs"], dtype=np.float32)
targets = np.asarray(raw["targets"], dtype=np.int64)
sample_paths: List[str] = list(raw["sample_paths"])
num_train = int(raw["num_train"])
num_val = int(raw["num_val"])
num_test = int(raw["num_test"])
scored_split: str = str(raw["split"])
num_samples = int(targets.shape[0])

metrics = metrics_at_threshold(probs, targets, threshold=threshold)

split_label = {
    "val": "Validation (selection)",
    "test": "Test (held-out)",
}[scored_split]


# ---------------------------------------------------------------------------
# Split summary
# ---------------------------------------------------------------------------

st.subheader("1. Dataset 同 split")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total patches", f"{num_train + num_val + num_test}")
col_b.metric("Train", f"{num_train}")
col_c.metric("Validation", f"{num_val}")
col_d.metric("Test", f"{num_test}" if num_test > 0 else "—")

st.caption(
    f"Currently scoring **{split_label}** · "
    f"{num_samples} samples · "
    f"Crack {int((targets == CLASS_NAMES.index('Crack')).sum())} / "
    f"No Crack {int((targets == CLASS_NAMES.index('No Crack')).sum())}. "
    + (
        "Test 係完全冇掂過嘅 slice,最代表真 generalisation。"
        if scored_split == "test"
        else "Val 係揀 best checkpoint 嗰 slice,報出嚟會樂觀少少 — "
        "打 test 數字請撳上面 slider 切去 Test。"
    )
)


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------

st.subheader("2. 總體準確率 / Headline metrics")

per_class = metrics["per_class"]
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
col2.metric(
    "Crack recall",
    f"{per_class['Crack']['recall'] * 100:.2f}%",
    help="真裂縫入面,有幾多個捉到? 對施工安全最重要。",
)
col3.metric(
    "Crack precision",
    f"{per_class['Crack']['precision'] * 100:.2f}%",
    help="模型話係裂縫果堆,入面有幾多個真係裂縫?",
)

st.caption(
    f"Threshold = {threshold:.2f} · "
    f"Scored on **{split_label}** (N={num_samples})"
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
    num_samples,
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
    fp = int(cm[CLASS_NAMES.index("No Crack"), CRACK_INDEX])
    fn = int(cm[CRACK_INDEX, CLASS_NAMES.index("No Crack")])
    tp = int(cm[CRACK_INDEX, CRACK_INDEX])
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
# ROC curve
# ---------------------------------------------------------------------------

st.subheader("5. ROC curve")
st.caption(
    "掃晒所有 threshold,畫 TPR (Crack recall) vs FPR (False-alarm rate)。"
    "完美 classifier 貼住左上角;隨機 classifier 就係對角線。"
    "**AUC** = 隨機抽一張 Crack 相同一張 No Crack 相,model 俾 Crack 相"
    "較高分嘅機率。"
)

roc = roc_curve_points(probs, targets)
auc_value = float(roc["auc"])

if np.isnan(auc_value):
    st.info(
        "呢個 slice 入面只得一個 class,ROC 冇意義。切去第個 slice 或者"
        "唔同 seed 再試。"
    )
else:
    roc_df = pd.DataFrame(
        {
            "fpr": np.asarray(roc["fpr"]),
            "tpr": np.asarray(roc["tpr"]),
            "threshold": np.asarray(roc["thresholds"]),
        }
    )
    # Replace +inf threshold so Altair tooltip formats nicely.
    roc_df.loc[~np.isfinite(roc_df["threshold"]), "threshold"] = 1.0

    current = point_on_roc(roc, threshold)
    current_df = pd.DataFrame([current])
    baseline_df = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})

    axis_scale_x = alt.Scale(domain=[0.0, 1.0])
    axis_scale_y = alt.Scale(domain=[0.0, 1.005])

    baseline = (
        alt.Chart(baseline_df)
        .mark_line(strokeDash=[4, 4], color="#9aa0a6")
        .encode(
            x=alt.X("fpr:Q", scale=axis_scale_x, title="False Positive Rate"),
            y=alt.Y("tpr:Q", scale=axis_scale_y, title="True Positive Rate"),
        )
    )
    curve = (
        alt.Chart(roc_df)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("fpr:Q", scale=axis_scale_x),
            y=alt.Y("tpr:Q", scale=axis_scale_y),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("fpr:Q", format=".3f"),
                alt.Tooltip("tpr:Q", format=".3f"),
            ],
        )
    )
    marker = (
        alt.Chart(current_df)
        .mark_point(
            color="#d62728", size=120, filled=True, opacity=1.0
        )
        .encode(
            x="fpr:Q",
            y="tpr:Q",
            tooltip=[
                alt.Tooltip("threshold:Q", format=".3f", title="current"),
                alt.Tooltip("fpr:Q", format=".3f"),
                alt.Tooltip("tpr:Q", format=".3f"),
            ],
        )
    )

    roc_chart = (baseline + curve + marker).properties(
        height=360,
        title=(
            f"ROC · AUC = {auc_value:.4f} · "
            f"threshold {threshold:.2f} → "
            f"TPR {current['tpr'] * 100:.1f}%, "
            f"FPR {current['fpr'] * 100:.1f}%"
        ),
    )
    st.altair_chart(roc_chart, use_container_width=True)

    col_auc1, col_auc2, col_auc3 = st.columns(3)
    col_auc1.metric("AUC", f"{auc_value:.4f}")
    col_auc2.metric(
        "TPR @ current threshold",
        f"{current['tpr'] * 100:.2f}%",
        help="= Crack recall。越高越少漏報。",
    )
    col_auc3.metric(
        "FPR @ current threshold",
        f"{current['fpr'] * 100:.2f}%",
        help="= 無裂縫但被標成 Crack 嘅比率,越低越少誤報。",
    )
    st.caption(
        "紅點係而家 sidebar threshold 對應嘅 operating point,拖 threshold "
        "slider 就會沿住條 curve 移動。AUC ≥ 0.9 通常當 excellent,"
        "0.8 - 0.9 good,0.5 即係同亂估無分別。"
    )


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

st.subheader("6. Threshold sweep")
st.caption(
    "掃一次 0.05 → 0.95,睇下應唔應該用其他 threshold。當前嘅 threshold 會"
    "喺圖上面以紅線標出。"
)
thresholds = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds:
    m = metrics_at_threshold(probs, targets, float(t))
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

st.subheader("7. 錯得最離譜嘅樣本 / Worst misclassifications")
crack_probs = probs[:, CRACK_INDEX]
preds = metrics["preds"]
wrong_mask = preds != targets
wrong_indices = np.where(wrong_mask)[0]


def _confidence_distance(i: int) -> float:
    """How far the wrong prediction was past the threshold, larger = worse."""
    return abs(float(crack_probs[i]) - threshold)


if wrong_indices.size == 0:
    st.success(
        f"Threshold {threshold:.2f} 之下,{split_label} 嘅 {num_samples} 張"
        " 全部答啱。唔使睇失敗例子。"
    )
else:
    ranked: List[Tuple[int, float]] = sorted(
        ((int(i), _confidence_distance(int(i))) for i in wrong_indices),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_k = st.slider(
        "顯示幾多張?",
        min_value=4,
        max_value=min(16, len(ranked)),
        value=min(8, len(ranked)),
        step=1,
    )

    cols_per_row = 4
    for row_start in range(0, top_k, cols_per_row):
        cols = st.columns(cols_per_row)
        for offset, col in enumerate(cols):
            i = row_start + offset
            if i >= top_k:
                break
            idx, _dist = ranked[i]
            path = sample_paths[idx]
            name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            try:
                img_bytes = api_client.dataset_image_bytes(path)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except BackendError as exc:
                col.warning(f"讀唔到 {name}: {exc}")
                continue
            true_label = CLASS_NAMES[int(targets[idx])]
            pred_label = CLASS_NAMES[int(preds[idx])]
            p_crack = float(crack_probs[idx])
            col.image(img, width="stretch")
            col.caption(
                f"**{name}**\n\n"
                f"真實: `{true_label}`\n\n"
                f"預測: `{pred_label}` · P(Crack)={p_crack * 100:.1f}%"
            )

    st.caption(
        f"總共有 {wrong_indices.size} 個錯誤 (accuracy = "
        f"{metrics['accuracy'] * 100:.2f}%)。"
        "上面排序係按照「預測信心離 threshold 有幾遠」— 信心越大越錯就越值得睇。"
    )
