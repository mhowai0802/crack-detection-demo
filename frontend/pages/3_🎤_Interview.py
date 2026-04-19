"""Interview prep cheat-sheet page (Cantonese).

A self-contained static page: every question is an ``st.expander`` with
talking points + likely follow-ups, grouped by topic. No backend calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(
    page_title="面試準備 - 混凝土裂縫偵測",
    page_icon=":material/record_voice_over:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

sidebar = st.sidebar
sidebar.title(":material/record_voice_over: Interview Prep")
sidebar.caption("面試官可能會問嘅問題 · 自己 quiz 自己")
sidebar.divider()

with sidebar.expander(":material/list: 內容目錄", expanded=True):
    st.markdown(
        """
        0. 技術名詞速記 (tech glossary)
        1. 項目概覽
        2. ML 基礎 / Transfer learning
        3. 訓練策略
        4. 數據集
        5. 評估 / Metrics
        6. Grad-CAM / 可解釋性
        7. LLM 整合 (HKBU GenAI)
        8. 系統架構 (FastAPI + Streamlit)
        9. 限制 / 改進方向
        10. 行為 / 軟性題
        11. Segmentation + HK 合規 + Sample 策略
        """
    )

sidebar.divider()

with sidebar.expander(":material/tips_and_updates: 答題心法", expanded=True):
    st.markdown(
        """
        - **STAR**: Situation → Task → Action → Result,尤其係行為題
        - 講數字:accuracy / recall / F1 / 數據量,數字令你嘅答案可信
        - 主動講**限制** + **點樣改善**,面試官覺得你有 engineering mindset
        - 唔識就坦白講「我冇做呢部分,但我會咁諗...」,唔好作
        """
    )

sidebar.divider()
sidebar.caption(
    "建議用法:每次揀 2-3 條題目,先自己答一次(出聲),"
    "再 click 開答案對照重點,改進表達。"
)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title(":material/record_voice_over: 面試準備 / Interview Prep")
st.caption(
    "呢頁係為咗幫你準備用呢個 project 去面試而整嘅 cheat sheet。"
    "每條題目都 pair 咗一組「答案重點」+「可能追問」。"
    "先試吓自己答,再展開對照,等你 talking points 更順口。"
)

st.info(
    "📌 **定位一句話:** 「呢個係一個貼近 **HK MBIS** 場景嘅兩段式混凝土裂縫偵測 "
    "demo — ResNet18 classifier 做快速 triage,偵測到 Crack 再 trigger 一個 "
    "~1.9M 參數嘅小型 U-Net 出 **pixel mask + max-width / length**,配埋 "
    "Grad-CAM 解釋,再由 HKBU GenAI 生成符合 **SUC 2013 / PNAP APP-137 / "
    "Buildings Ordinance** 格式嘅中/英文檢查報告;前後端分開,FastAPI 做 "
    "backend、Streamlit 做 frontend。」",
    icon=":material/lightbulb:",
)

st.divider()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def qa(
    question: str,
    points: list[str],
    follow_ups: list[str] | None = None,
    *,
    icon: str = ":material/help:",
) -> None:
    """Render a question as an expander with bullet-point answer."""
    with st.expander(f"{question}", icon=icon):
        st.markdown("**答案重點 / Key talking points**")
        for p in points:
            st.markdown(f"- {p}")
        if follow_ups:
            st.markdown("")
            st.markdown("**可能追問 / Likely follow-ups**")
            for f in follow_ups:
                st.markdown(f"- {f}")


def term(
    name: str,
    analogy: str,
    usage: str,
    one_liner: str,
    *,
    icon: str = ":material/menu_book:",
) -> None:
    """Render a glossary entry: analogy, project usage, interview one-liner."""
    with st.expander(name, icon=icon):
        st.markdown(f"**比喻 / Analogy:** {analogy}")
        st.markdown(f"**呢個 project 點用:** {usage}")
        st.markdown(f"**面試 one-liner (English):** _{one_liner}_")


# ---------------------------------------------------------------------------
# 0. Tech glossary — plain-language explanations
# ---------------------------------------------------------------------------

st.header("0. 技術名詞速記 (Tech glossary)")
st.caption(
    "每個詞 = 一句比喻 + 呢個 project 點用 + 一句英文 interview 答案。"
    "面試前快速刷一次,卡住嗰陣都有 fallback 講法。"
)

st.subheader("ML / Deep learning 基礎")

term(
    "Transfer learning · 轉學式學習",
    "好似請咗一個識睇 1000 類物件嘅學生返嚟,我只教佢分『crack / no crack』,"
    "佢已有嘅視覺能力 (邊、紋理) 直接 reuse。",
    "ResNet18 用 ImageNet (1.2M 張相, 1000 類) 嘅 pretrained weights,"
    "我只換尾端 `fc` layer 再 fine-tune,3,200 張 patch 就攞到 ~97% accuracy。",
    "Reuse a model pretrained on a large dataset so a small target dataset "
    "can reach high accuracy.",
    icon=":material/school:",
)

term(
    "CNN · Convolutional Neural Network",
    "好似好多個放大鏡掃過張相,每個鏡專睇一種 pattern (邊、角、紋理);"
    "一層層疊落去,由低階 edge 變成高階物件 feature。",
    "ResNet18 classifier 同 U-Net segmenter 都係 CNN 家族,"
    "分類做 global decision,U-Net 做 pixel-level。",
    "A neural network that slides learnable filters across the image to "
    "extract local spatial patterns.",
    icon=":material/grid_view:",
)

term(
    "ResNet18 + Residual connection",
    "18 層嘅 CNN,每幾層加一條『短路線』將 input 直接加返去 output;"
    "好處係 deep network 都唔會 gradient vanish,train 得郁。",
    "做分類 backbone,~11M params,CPU forward 都可以 <100ms,適合 demo。",
    "An 18-layer CNN with skip connections so gradients flow easily through "
    "depth.",
    icon=":material/linear_scale:",
)

term(
    "Softmax · 機率分佈化",
    "將 model 吐出嘅 raw score (logit) normalize 到 0-1 + 加埋 = 1,"
    "睇落似機率;但注意:冇 calibrate 過,唔等於真·機率。",
    "Classifier 尾端出 `[p_no_crack, p_crack]`,我用呢個對比 threshold "
    "揀 operating point。",
    "Normalises raw logits into a valid probability distribution; "
    "not necessarily calibrated.",
    icon=":material/functions:",
)

term(
    "Fine-tune + 2-phase training",
    "Phase 1 freeze backbone、只 train 新 `fc` head (等新 head 唔好扯亂舊"
    "backbone);Phase 2 unfreeze 全部,用細 10× 嘅 learning rate 繼續。",
    "Phase 1: lr=1e-3 × 2 epochs;Phase 2: lr=1e-4 × 8 epochs。",
    "Warm up the new head first, then unfreeze and fine-tune everything at "
    "a smaller learning rate.",
    icon=":material/stairs:",
)

term(
    "Learning rate (lr)",
    "每步行幾遠。大 lr 快但易跳過最優點,細 lr 穩但慢;fine-tune 通常要"
    "細 lr 保護已有知識。",
    "Adam optimizer,Phase 1 lr=1e-3,Phase 2 lr=1e-4,配 CosineAnnealing "
    "scheduler slowly decay。",
    "The step size for gradient updates; smaller preserves pretrained "
    "knowledge, larger learns faster but is riskier.",
    icon=":material/speed:",
)

term(
    "Data augmentation",
    "將原相翻轉、旋轉、調光、crop 扭多幾個版本出嚟 train,令 model 見過"
    "多啲變化,唔死記原相。",
    "分類:horizontal/vertical flip + rotate ±15° + ColorJitter。"
    "分割:用 `torchvision.transforms.v2` + `tv_tensors.Mask`,確保 image "
    "同 mask **同步** flip / rotate (唔會 label misalign)。",
    "Synthetically varying the training data so the model generalises "
    "better to unseen inputs.",
    icon=":material/tune:",
)

term(
    "Overfit vs Generalise",
    "Overfit = 死記 training 答案,出新題就唔識;generalise = 學到通用"
    "規律,新題都識。",
    "防範手段:seeded 70/15/15 split + best-checkpoint by val acc + "
    "data augmentation + early stopping。",
    "Overfitting is memorising the training set instead of learning "
    "generalisable patterns.",
    icon=":material/warning:",
)

st.subheader("Metrics / 評估")

term(
    "Confusion matrix · 混淆矩陣",
    "2×2 table:TP / FP / FN / TN。橫軸 predicted、縱軸 actual,一眼睇到 "
    "error 集中喺邊。",
    "Evaluation 頁有 2×2 bar chart + table;FN (漏報真裂縫) 係 safety-"
    "critical,所以我特登標紅。",
    "A 2×2 table counting correct and incorrect predictions per class.",
    icon=":material/grid_on:",
)

term(
    "Precision / Recall / F1",
    "Precision = 『我話係 crack 嗰啲,真係 crack 嘅比例』(準唔準);"
    "Recall = 『真 crack 嗰堆,有幾多被我搵返出嚟』(漏唔漏);"
    "F1 = 兩者 harmonic mean,對 imbalance 較 robust。",
    "Safety-critical,所以 **recall** 優先;F1 用嚟做 single-number "
    "comparison / ablation。",
    "Precision = when I say positive, how often I'm right; "
    "Recall = of all true positives, how many I catch; F1 balances both.",
    icon=":material/speed:",
)

term(
    "Threshold (決定線)",
    "Probability 過咗呢條線先判 positive。0.5 只啱 balanced cost,"
    "真實世界 FN vs FP cost 唔對稱就要調。",
    "Evaluation 頁 slider 實時掃 0.05 → 0.95,"
    "睇 precision / recall / F1 變化,揀 business knee point。",
    "The cut-off that turns a probability into a decision; 0.5 is just "
    "the default.",
    icon=":material/trending_up:",
)

term(
    "ROC curve · AUC",
    "掃晒所有 threshold,畫 TPR (recall) vs FPR (誤報率);曲線愈 hug 左上"
    "愈好,AUC (area under curve) = 1.0 完美、0.5 同擲銀仔冇分別。",
    "Evaluation 頁底部畫 ROC + AUC,俾人一眼睇 classifier quality,"
    "唔受 threshold 選擇影響。",
    "A threshold-free view of classifier quality; AUC 1.0 is perfect, "
    "0.5 is random.",
    icon=":material/show_chart:",
)

term(
    "IoU · Intersection over Union (Jaccard)",
    "兩個 mask 重疊面積 ÷ 合併面積。1 = 完全 match,0 = 完全唔重;"
    "係 segmentation 嘅 standard metric,因為 pixel accuracy 喺 imbalanced "
    "mask 永遠 >95% 係冇意義。",
    "U-Net 訓練嘅 early-stopping metric,典型 val IoU ≈ 0.72。",
    "Overlap over union — the go-to segmentation metric; robust to "
    "foreground-background imbalance.",
    icon=":material/join_inner:",
)

term(
    "Calibration · 機率校正",
    "Softmax 出嘅 probability 睇落似機率,但唔一定係真·機率 (e.g. "
    "model 講 0.9 時,實際只有 70% 係對)。要用 reliability diagram check、"
    "用 temperature scaling 修。",
    "呢個 demo **冇** 做 calibration,係一個 known limitation;"
    "Q6 interview 會講到。",
    "Ensuring predicted probabilities match observed frequencies, "
    "typically via temperature scaling.",
    icon=":material/thermostat:",
)

st.subheader("Loss / 訓練信號")

term(
    "Cross-Entropy · BCE",
    "Model 估啱 class 就 loss 細,估錯就 loss 指數級大 — 係 classification "
    "嘅 default。Binary 版本叫 BCE (Binary Cross-Entropy)。",
    "分類用 `CrossEntropyLoss` (2 logit, 數學上同 BCE 等價);"
    "分割用 `BCEWithLogitsLoss` + Dice 合成。",
    "The default classification loss — heavily penalises confident "
    "wrong predictions.",
    icon=":material/functions:",
)

term(
    "Dice loss",
    "直接 penalise 兩個 mask 嘅 overlap 唔夠。對『crack pixels 好少』嘅 "
    "imbalanced segmentation 特別 work,因為 BCE 單獨會俾 model 偷懶估"
    "『全黑 mask』都攞到低 loss。",
    "U-Net 訓練用 `0.5·BCE + 0.5·Dice`,thin crack 都 train 得起。",
    "An overlap-based loss that resists the 'predict all-zero' shortcut "
    "under heavy class imbalance.",
    icon=":material/join_full:",
)

st.subheader("Explainability / 可解釋性")

term(
    "Grad-CAM · Gradient-weighted Class Activation Map",
    "熱力圖,答『model 睇張相邊個位嚟作決定』;紅 = 對判 crack 貢獻最大、"
    "藍 = 冇乜影響。",
    "Forward 到 `layer4` 攞 activation,backward 攞 gradient,GAP 成 "
    "channel weights,ReLU + 加權和 = CAM,upsample 返 224×224 + JET "
    "colormap overlay。",
    "A heatmap visualising which image regions drove the classifier's "
    "decision.",
    icon=":material/visibility:",
)

term(
    "Dominant quadrant hint",
    "LLM 睇唔到圖,所以我將 Grad-CAM 分成 2×2 格、揀最熱嗰格,變成文字 "
    "『熱力集中喺右下角』餵 prompt;俾 LLM 一個 spatial grounding hint。",
    "`src/gradcam.dominant_quadrant()` 回 `top-left` / `top-right` / "
    "`bottom-left` / `bottom-right` / `centre-heavy` / `uniform` 六選一。",
    "Coarse spatial grounding in text so a text-only LLM can reference "
    "where the model looked.",
    icon=":material/explore:",
)

st.subheader("Segmentation 專用")

term(
    "Semantic segmentation",
    "唔係答『張相有冇 crack』(classification),而係逐粒 pixel 答『呢粒 "
    "係 crack 定 background』;output 係 `[H, W]` mask。",
    "`POST /api/segment` 跑 U-Net 出 binary mask,由 mask 再 derive "
    "width / length。",
    "Pixel-level classification — every pixel is assigned a class.",
    icon=":material/grain:",
)

term(
    "U-Net (encoder-decoder + skip connection)",
    "先壓縮 (encoder) 提取 semantic,再解壓 (decoder) 還原解像度;中間"
    "加 skip connection 將早期嘅細節 (邊界) 帶返去 decoder,所以對細長 "
    "物件 (裂縫) 特別好。",
    "自寫 4-stage U-Net,base channel = 16,~1.9M params,CPU 推論 ~300ms。",
    "An encoder-decoder with skip connections — the workhorse for "
    "pixel-wise segmentation.",
    icon=":material/architecture:",
)

term(
    "Connected components",
    "喺 binary mask 入面數『幾多團獨立嘅 1』;兩粒 pixel 相鄰 (4- 或 "
    "8-connectivity) 就當同一 team。",
    "`cv2.connectedComponents(mask)` 數 crack 有幾條獨立分支,"
    "變成 `num_components` 餵 LLM prompt。",
    "Counting how many separate blobs of foreground exist in a "
    "binary mask.",
    icon=":material/bubble_chart:",
)

term(
    "Distance transform (max width)",
    "對 mask 內每粒 pixel,計佢到最近 background pixel 嘅距離;"
    "最遠嗰點就喺最闊 crack 嘅中央,距離 × 2 ≈ max width (px)。",
    "`cv2.distanceTransform(mask, cv2.DIST_L2, 3).max() * 2`,"
    "比自己寫 loop 快好多。",
    "For each foreground pixel, compute the distance to the nearest "
    "background; the max doubled approximates the thickest width.",
    icon=":material/straighten:",
)

term(
    "Morphological skeleton (crack length)",
    "將闊度唔同嘅 mask 一層層『剝皮』(erosion),直到剩一條 1-pixel 中軸"
    "線;呢條線嘅像素數 ≈ crack length (px)。",
    "OpenCV `erode` + `dilate` 反覆 iterate 實作;"
    "餵 `length_px` 入 LLM prompt。",
    "Iteratively eroding the mask down to a 1-pixel-wide centreline "
    "whose pixel count approximates the crack length.",
    icon=":material/polyline:",
)

st.subheader("LLM / AI report")

term(
    "LLM · Large Language Model",
    "會講人話嘅巨型 autocomplete;會跟 natural-language 指示,整合 context "
    "寫出連貫段落。",
    "HKBU GenAI `gpt-4.1-mini`,中/英文雙語產生 HK MBIS 合規風格嘅"
    "檢查報告 (Observation / Severity / Recommendation / Disclaimer)。",
    "A large autoregressive model that follows natural-language "
    "instructions and returns coherent text.",
    icon=":material/smart_toy:",
)

term(
    "Prompt engineering (System vs User prompt)",
    "**System prompt** = 俾 LLM 個『角色設定』同『規則』(長期生效);"
    "**User prompt** = 今次嘅實際問題 + context (每次 turn 都新)。",
    "System 寫明『你係 HK RSE 報告草擬員,只可以 cite SUC 2013 / PNAP "
    "APP-137 / BO Cap. 123』;user 入面 inject "
    "`[Prediction] [Grad-CAM] [Segmentation]` 三個 block 做 grounding。",
    "Structuring role, rules and context so the LLM answers reliably.",
    icon=":material/edit_note:",
)

term(
    "Hallucination + Grounding",
    "**Hallucination** = LLM 自己作冇根據嘅嘢 (e.g. 作一個唔存在嘅 "
    "clause);**Grounding** = 俾佢可驗證嘅 hard fact,令佢冇得亂作。",
    "每次 call 都 inject 實際 probability + Grad-CAM quadrant + "
    "SegStats,加 system prompt 明確叫佢『信心低就實話實說,唔好誇張』,"
    "最後加 disclaimer。",
    "Hallucination is LLMs inventing facts; grounding counters it by "
    "injecting verified context into the prompt.",
    icon=":material/fact_check:",
)

st.subheader("Backend / 工程")

term(
    "FastAPI",
    "Python 寫 REST API 嘅現代 framework;用 Python type hints 做 "
    "validation,自動生成 OpenAPI (Swagger) docs。",
    "Backend 所有 endpoint (`/predict` / `/gradcam` / `/segment` / "
    "`/ai/report` ...) 都係 FastAPI route,`/docs` 自動攞到 Swagger UI。",
    "A modern Python API framework with type-hint-driven validation and "
    "auto-generated OpenAPI docs.",
    icon=":material/api:",
)

term(
    "Pydantic",
    "用 Python class 定義 data schema,自動 validate type + serialise "
    "JSON;FastAPI 嘅 request/response model 都靠佢。",
    "`backend/schemas.py` 定義 `PredictionResponse` / `SegmentResponse` / "
    "`SegStats` / `ReportRequest`,CI 會 catch breaking change。",
    "Typed, self-validating data models in Python — schema-first design "
    "for APIs.",
    icon=":material/schema:",
)

term(
    "Streamlit",
    "用純 Python 寫 web UI,唔使掂 HTML/CSS/JS;每次用戶郁 widget,"
    "成個 script 重新行一次 (有 `st.cache_*` 幫手 memoize)。",
    "3 個 page (Home / Evaluation / Interview + Architecture) 全部 "
    "Streamlit;`st.cache_data` 緩存 sample list + health ping。",
    "A Python library that turns a script into a live web app — "
    "zero frontend code.",
    icon=":material/web:",
)

term(
    "`@lru_cache` · Memoization",
    "Function decorator,記住同一個 input 嘅 return value,第二次就返 "
    "cache 唔再計。",
    "`get_classifier()` + `get_seg_model()` 用 `lru_cache(maxsize=1)` 包住,"
    "model 整個 process life-time 只 load 一次。",
    "A memoization decorator that caches a function's return per unique "
    "argument.",
    icon=":material/storage:",
)

term(
    "Base64 encoding",
    "將 binary bytes (e.g. PNG 圖片) 轉做 ASCII string,可以塞入 "
    "JSON / URL;代價係體積大咗 ~33%。",
    "`/api/segment` 回 JSON,入面 `overlay_png_b64` + `mask_png_b64` 兩張"
    "圖都 base64 encode,一次 request 攞齊 overlay + mask + stats,"
    "frontend 唔使做 second call。",
    "A text-safe encoding for binary data so it can live inside JSON.",
    icon=":material/data_object:",
)

st.divider()


# ---------------------------------------------------------------------------
# 1. Project overview
# ---------------------------------------------------------------------------

st.header("1. 項目概覽 (Project overview)")

qa(
    "Q1. 可唔可以一分鐘內介紹吓呢個 project?",
    [
        "**問題:** HK MBIS 每 10 年一次強制樓宇檢驗,RSE / AP 要逐張相判 "
        "「有冇 crack、喺邊、幾闊」,手動又慢又主觀。",
        "**方案(兩段式):**\n"
        "    (1) ResNet18 transfer learning 做 **Crack vs No Crack triage**;\n"
        "    (2) 一旦 flagged 做 Crack,trigger 一個 ~1.9M 參數嘅小型 "
        "**U-Net** 出 **pixel mask + max width / length (px)**,作為 "
        "HK SUC 2013 / PNAP APP-137 合規討論嘅 pixel-level 證據。",
        "**解釋性:** Grad-CAM (classifier 嘅注意力) 同 U-Net mask "
        "(真係 crack pixels) 並排顯示 — 係 debug classifier 係咪 "
        "look-at-the-right-thing 嘅快速工具。",
        "**數字:** 分類用 Özgenel CCIC (3,200 patch,70/15/15 split),"
        "test accuracy ~97%、Crack F1 ≈ 0.96;分割用 DeepCrack "
        "(537 + 237),val IoU ≈ 0.72。",
        "**HK 味道:** AI 報告用中/英文雙語,prompt 明確引 SUC 2013 "
        "(0.2 / 0.3 mm 裂縫限值)、PNAP APP-137、Buildings Ordinance "
        "Cap. 123;`scripts/fetch_hk_samples.py` 可以 local download "
        "BD Common Building Defects gallery(`.gitignore` 保護,"
        "唔 push)做 interview demo。",
        "**架構:** FastAPI backend 獨佔所有 torch / LLM / OpenCV,"
        "Streamlit frontend 純 HTTP client;兩者獨立 Dockerize、"
        "獨立 scale、獨立換掉。",
    ],
    [
        "點解揀 concrete crack,唔揀 spalling 或者 rebar rust?",
        "點解唔淨係 classification、又再加 segmentation?(Q3 會深入)",
        "你用過邊啲 stakeholder feedback 去 refine?",
    ],
    icon=":material/rocket_launch:",
)

qa(
    "Q2. 呢個 project 解決咩 business pain point?ROI 點計?",
    [
        "**人力成本:** 傳統人手巡查,1 個工程師走一個 site 可以用半日;"
        "AI 可以 30 秒處理一張相,配合相機或者 drone 做 batch screening。",
        "**安全風險:** 人眼有 fatigue / inconsistency,漏咗一條 crack "
        "可能等於一個結構問題;AI 可以先做 triage,"
        "將「明顯有裂」同「明顯冇裂」分開,工程師只 focus 模棱兩可個堆。",
        "**可追溯:** 每次預測都有時間、位置、confidence、Grad-CAM,"
        "係 auditable 嘅紀錄,對 QA / insurance / legal 有用。",
        "**ROI:** 就算 precision 未達 100%,做 triage 都可以令工程師吞吐量"
        "翻幾倍;FP 成本係「多睇一張相」,FN 成本先係大鑊,"
        "所以 threshold 要調細少少 (high recall)。",
    ],
    [
        "FN 成本如果量化,你會點估?",
        "點同工地既有嘅 safety workflow integrate?",
    ],
    icon=":material/query_stats:",
)

qa(
    "Q3. 點解同時做 classification + segmentation,唔淨係揀其中一樣?",
    [
        "**兩者目的唔同:** Classifier 答「張相有冇 crack」用嚟做 triage;"
        "U-Net 答「crack pixels 具體喺邊、幾闊、幾長」用嚟支援合規討論。"
        "而家個 pipeline 係 **gated hybrid** — Classifier 先跑,"
        "flagged 做 Crack 先至 call U-Net。",
        "**慳 compute:** Segmentation 慢過分類 ~5×;大部分 HK MBIS 照相"
        "其實係 No Crack,gate 住之後只有 ~20-30% request 會跑 U-Net。",
        "**減 false segmentation:** 如果無分類,U-Net 會喺完全無 crack "
        "嘅相上亂畫 pixel(純 noise);先用 classifier 攔住,mask 會"
        "乾淨好多,對 HK 合規 report 個可信度好重要。",
        "**唔揀純 detection (YOLO):** Ultralytics YOLO 係 **AGPL-3.0**,"
        "同呢個 repo 嘅 **MIT** license 會撞;加上 bounding box 度唔到 "
        "SUC 2013 要求嘅 crack width，所以我明寫「detection 唔係 goal」。",
        "**Data 成本係可以接受嘅:** DeepCrack 有 ~537 pairs 公開 + "
        "non-commercial license,唔需要自己 label 就 train 到 val "
        "IoU ≈ 0.72。",
    ],
    [
        "Gate 住 U-Net 會唔會令 rare true-crack 俾 classifier 漏咗?",
        "Dataset 唔平衡 (cracked pixels << non-cracked) 你會點處理?",
        "點解 classifier + segmenter 用不同 dataset(Özgenel vs DeepCrack)?",
    ],
    icon=":material/filter_center_focus:",
)

st.divider()


# ---------------------------------------------------------------------------
# 2. ML fundamentals / transfer learning
# ---------------------------------------------------------------------------

st.header("2. ML 基礎 / Transfer learning")

qa(
    "Q4. 咩係 transfer learning?點解 ImageNet pretrained 對混凝土相有用?",
    [
        "**定義:** 將一個喺大 dataset (ImageNet, 1.2M 張 1000 類) 訓練好嘅"
        "模型,reuse 去一個相關但細啲嘅新任務。",
        "**點解 work:** 前幾 layer 學到嘅係通用 visual feature —— edges, "
        "textures, shapes —— 裂縫本質上就係一啲「窄、長、低亮度」嘅 edge,"
        "呢啲 feature 同 ImageNet 學到嘅 overlap 高。",
        "**幾大得益:** 我個 demo 只有 3,200 張,如果 from scratch 訓練"
        "ResNet18 基本上 overfit,但 fine-tune pretrained model 就做到 "
        "~97% accuracy。",
        "**Fallback:** 就算冇 checkpoint,demo 會 fallback 落 pretrained "
        "ResNet18,UI 唔會壞,只係 prediction 冇意義(隨機)。",
    ],
    [
        "如果 target domain 同 ImageNet 差好遠 (e.g. 醫療 X-ray),"
        "transfer learning 仲 work 咪?",
        "點解唔直接凍結 backbone,只訓練 classifier?",
    ],
    icon=":material/school:",
)

qa(
    "Q5. 點解揀 ResNet18?唔可以用 ResNet50 / EfficientNet / ViT?",
    [
        "**Trade-off:** ResNet18 係 ~11M 參數,CPU 推論都好快 (<100ms),"
        "適合 demo + edge deployment。ResNet50 大 2-3 倍、"
        "EfficientNet-B0 參數差唔多但 architecture 複雜啲、"
        "ViT 喺細 dataset 通常 underperform。",
        "**Baseline 心態:** Transfer learning 項目,先用細 model 攞 baseline,"
        "確認 pipeline (data → train → eval → UI) 成立,再換大 model 對比。",
        "**Grad-CAM friendly:** ResNet 尾個 `layer4` 好 natural 做 Grad-CAM "
        "target layer,ViT 就要另一套 attention rollout。",
        "**結果驅動:** ~97% accuracy 已經 ≥ published baselines,"
        "再 scale up model 不如去 cover 真實 site 嘅 domain gap。",
    ],
    [
        "如果 accuracy ceiling 係由 model size 限制,你點 verify?",
        "你會點 benchmark ResNet18 vs ResNet50 公平啲?",
    ],
    icon=":material/memory:",
)

qa(
    "Q6. Softmax 出嘅 probability 可唔可以當成「真實裂縫機率」?",
    [
        "**唔可以。** Softmax 只係將 logit normalize 到 0-1,唔等於 "
        "calibrated probability。",
        "**點知冇 calibrate:** 無做過 temperature scaling / Platt scaling / "
        "isotonic regression,reliability diagram 應該唔係 diagonal。",
        "**我點 handle:** 我將個 softmax 當成 **operational knob**,"
        "用 sidebar slider 調 threshold;threshold sweep section 會俾你睇吓"
        "每個 threshold 底下 precision / recall / F1 嘅變化,選一個 match "
        "業務 cost 嘅 operating point。",
        "**真正要 probability:** 上線之前可以加 temperature scaling(一個 "
        "scalar,唔改 accuracy 但改 calibration),再用 reliability diagram "
        "confirm。",
    ],
    [
        "Platt scaling vs temperature scaling 分別?",
        "點樣喺 imbalanced dataset 做 calibration?",
    ],
    icon=":material/thermostat:",
)

st.divider()


# ---------------------------------------------------------------------------
# 3. Training strategy
# ---------------------------------------------------------------------------

st.header("3. 訓練策略 (Training)")

qa(
    "Q7. 點解兩階段訓練?有冇 ablation?",
    [
        "**Phase 1 (warm-up head):** 凍結 backbone,只訓練新嘅 `fc` layer,"
        "Adam lr=1e-3,預設 2 epochs。因為新 head 嘅 weight 係隨機,"
        "直接一齊 train 會將個已 pretrained 嘅 backbone 扯亂。",
        "**Phase 2 (full fine-tune):** Unfreeze 全部,用細好多嘅 "
        "lr=1e-4 繼續 train。細 lr 保護 pretrained feature 唔俾 overwrite。",
        "**Ablation (老實講):** 我冇正式跑 1-phase vs 2-phase ablation,"
        "呢個係我諗嘅 next step;不過好多 transfer learning paper "
        "(e.g. fastai) 都有類似做法,所以我揀咗 safer 嘅 default。",
        "**結果:** 最後 val accuracy ~97%,冇明顯 overfit。",
    ],
    [
        "點解 Adam 唔用 SGD + momentum?",
        "`--freeze-epochs 0` 同 `--freeze-epochs 5` 你估會點?",
    ],
    icon=":material/stairs:",
)

qa(
    "Q8. 你用咗咩 augmentation?點解?",
    [
        "**Training transforms:** Horizontal + vertical flip(裂縫方向唔重要)、"
        "Random rotation ±15°(相機角度可變)、"
        "ColorJitter brightness/contrast 0.2(光線變化)。",
        "**冇用嘅:** Crop(驚 crop 走唯一條裂縫)、"
        "CutMix/MixUp(binary + 好細 dataset 估值有限)、"
        "heavy color distortion (對混凝土紋理唔好)。",
        "**Val / test:** deterministic,只 resize + ImageNet normalise,"
        "保證每次數字可 reproduce,三個 slice 可以直接比較。",
        "**ImageNet mean/std:** 用返同 pretrained backbone 一樣嘅 stats,"
        "維持 input distribution。",
    ],
    [
        "如果加 CutOut / CoarseDropout 你估會點?",
        "Domain-specific augmentation(例如加模擬陰影)你點 design?",
    ],
    icon=":material/tune:",
)

qa(
    "Q9. 點解 loss 用 CrossEntropyLoss?唔用 BCE 或者 Focal?",
    [
        "**CE for 2-class:** 數學上等同 BCE(out 2 個 logit vs 1 個),"
        "揀 CE 係因為 framework conventions + 容易擴展到 multi-class。",
        "**Class balance:** Mendeley dataset 係 50/50,"
        "我自己切嘅 patch dataset 大約都平衡,所以唔 need class weights。",
        "**Focal loss:** 針對 imbalanced data (e.g. 1:100),"
        "我呢邊 balance 嘅話 gain 有限,而且引入 2 個 hyperparameter。",
    ],
    [
        "如果 dataset 去到 95/5 imbalance,你會點改 loss?",
        "Label smoothing 有冇試過?",
    ],
    icon=":material/functions:",
)

qa(
    "Q10. 你點樣保證模型冇 overfit?",
    [
        "**Seeded 3-way split (70 / 15 / 15, seed=42):** `src/splits.py` "
        "係 single source of truth,`train.py` 同 `evaluate.py` 共用"
        "同一組 index,test slice 完全冇掂過(冇做 gradient、冇做 "
        "checkpoint selection)。",
        "**Best-checkpoint by val accuracy:** Training script 每個 epoch "
        "睇 val,save 最好嗰個 checkpoint,避免 last epoch 已經 overfit。",
        "**Unbiased 終值喺 test:** 訓練完一次性跑 test slice 報 "
        "generalisation,Evaluation 頁預設就係顯示呢個數字。",
        "**Gap monitoring:** Train loss 一路跌,val loss 喺尾階段開始 flatten,"
        "冇繼續上升即冇明顯 overfit;如果 test 數字同 val 差好遠,"
        "就係 selection bias / leakage 嘅警號。",
    ],
    [
        "K-fold cross-validation 你有冇做?",
        "Early stopping 你會點 set patience?",
        "點解揀 70 / 15 / 15 唔係 80 / 10 / 10?",
    ],
    icon=":material/monitoring:",
)

st.divider()


# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------

st.header("4. 數據集 (Dataset)")

qa(
    "Q11. 用咗邊啲 dataset?License?點 clean?",
    [
        "**Classifier — Özgenel CCIC** (Mendeley `5y9wdsg2zt`, v2,"
        "CC-BY-4.0): 原 40k 張 227×227 METU facade close-up;"
        "demo 用 HuggingFace mirror `Vizuara/concrete-crack-dataset` "
        "(800 張相 + mask),由 `scripts/prepare_data.py` 切 2×2 patch、"
        "用 mask 判 label,出到 3,200 張 balanced set。",
        "**Segmenter — DeepCrack (Zou 2018)** (`github.com/yhlleo/"
        "DeepCrack`, non-commercial research/edu): 537 train + 237 test "
        "張 RGB + binary mask,多 scale crack、大部分係路面同一般混凝土。"
        "`scripts/prepare_seg_data.py` 自動拉 zip(有嵌套 "
        "`DeepCrack.zip` 要二次解)、80/20 carve val split、寫入 "
        "`data_seg/{train,val,test}/{images,masks}`。",
        "**Transform:** 分類 resize 224×224 + ImageNet normalise;"
        "分割用 `torchvision.transforms.v2` + `tv_tensors.Mask`,"
        "flip / rotate / crop 對 image 同 mask **同步** apply,"
        "保證 label alignment。",
        "**全套 provenance:** `data/DATASETS.md` 列晒兩個 dataset 嘅 "
        "DOI、citation、license、file layout、split seed — single "
        "source of truth,Evaluation 頁頂部個 banner + Architecture "
        "頁 Section 4 都 point 返去。",
        "**Label 次序:** `src/constants.py` 寫死 `[\"No Crack\", "
        "\"Crack\"]`,前後端 shared。",
    ],
    [
        "800 → 3,200 個 patch,group-level leakage 會唔會有問題?(Q17)",
        "DeepCrack non-commercial,如果要商用你點 handle?",
        "點解唔用 CRACK500 / CFD?",
    ],
    icon=":material/dataset:",
)

qa(
    "Q12. Dataset 有冇 bias?你點 handle?",
    [
        "**已知 bias:** 所有相都係 close-up、乾淨、均勻光線、中心 crop。"
        "真實 site 有雜物 / 陰影 / 人 / 工具 / blur。",
        "**後果:** Domain gap 會令 real-world accuracy 跌明顯,"
        "一啲 naive augmentation (flip/rotate) 掩蓋唔到。",
        "**我點減輕:** (1) 對 stakeholder 講清楚 limitation;"
        "(2) 用 threshold slider 俾人 tune recall/precision;"
        "(3) future work 提出 active learning + domain adaptation。",
        "**唔 cover:** 磚、柏油、木、老化鋼筋都超出 scope,"
        "唔會亂 deploy 去嗰啲地方。",
    ],
    [
        "點 measure domain gap?",
        "你會點整一個 validation set for production use?",
    ],
    icon=":material/warning:",
)

qa(
    "Q13. 如果俾你再拎多啲資源(錢 / 人 / 時間),你第一步會點改 dataset?",
    [
        "**Data diversity:** 喺真實工地影 500-1,000 張 photo(鋼筋旁、"
        "柱、牆、不同光、不同相機角度),配合專業工程師 label,"
        "砌一個 held-out **site test set**,打返個真 accuracy。",
        "**Severity levels:** 而家只得 binary,我會加 3-4 級 severity "
        "(hairline / moderate / structural),俾下游決策更有用。",
        "**Negative hard case:** 特登收一啲容易 confuse 嘅 sample,"
        "例如接縫、陰影線、表面污漬,做 contrastive fine-tuning。",
        "**Active learning loop:** 將 model 喺 field 嘅 low-confidence "
        "prediction push 去 labelling queue,將 real distribution "
        "一步步併入 training set。",
    ],
    [
        "Severity 4 classes,點樣定義先唔會 inter-annotator disagree?",
        "你會點樣 prioritise labelling budget?",
    ],
    icon=":material/auto_awesome:",
)

st.divider()


# ---------------------------------------------------------------------------
# 5. Evaluation & metrics
# ---------------------------------------------------------------------------

st.header("5. 評估 / Metrics")

qa(
    "Q14. 你報 accuracy / precision / recall / F1,"
    "對呢個 project 邊個最重要?",
    [
        "**Recall for Crack 最重要。** 漏咗一條真裂縫(false negative)"
        "嘅 downstream cost 好大 —— 結構風險、insurance、人命。"
        "誤報(false positive)只係「多派個工程師去 double check」。",
        "**所以 threshold 會調細啲** (e.g. 0.3-0.4),換多啲 FP 返少啲 FN。",
        "**Accuracy 會誤導:** 如果 class imbalance 出現 (e.g. 99% no-crack),"
        "永遠答 no-crack 都 99% accurate,所以我 always 睇埋 per-class。",
        "**F1 for Crack** 用嚟做 single-number comparison / ablation。",
    ],
    [
        "你會用 PR-AUC 定 ROC-AUC?點解?",
        "如果 downstream 係 alert system,你會睇咩 operational metric?",
    ],
    icon=":material/speed:",
)

qa(
    "Q15. Confusion matrix 4 個格,邊個你最 care?",
    [
        "**FN (漏報):** 真裂縫但 model 話冇。最危險、影響安全,"
        "Evaluation 頁特登寫咗「FN = 漏報裂縫(最危險)」。",
        "**FP (誤報):** 冇裂但 model 話有。Cost 係人力浪費,可接受。",
        "**TP / TN:** 正常 case,用嚟計 precision / recall,"
        "冇特別 operational 意義。",
        "**Evaluation 頁 UI:** 2×2 table + bar chart 並排,"
        "一眼睇到 error pattern 係集中喺 FN 定 FP。",
    ],
    [
        "如果 business 話 FN cost = 50×FP cost,你點揀 threshold?",
        "Cost-sensitive learning(加 class weight / threshold moving)"
        "你揀邊種?",
    ],
    icon=":material/grid_on:",
)

qa(
    "Q16. Threshold sweep 做嚟做咩?點解唔 hardcode 0.5?",
    [
        "**0.5 係 argmax 決策,只啱 balanced cost**。真實場景 FN vs FP "
        "cost 唔對稱。",
        "**Sweep 0.05 → 0.95:** 一張圖睇到 accuracy / crack recall / "
        "crack precision / crack F1 點隨 threshold 變化,"
        "揀到「business knee point」而唔使 retrain。",
        "**運作方式:** Backend 只 forward 一次、"
        "store `probs[N,2]`;前端 `metrics_at_threshold(probs, t)` "
        "純 numpy 即時計,slider 拉邊邊就算到。",
        "**好處:** Stakeholder 可以自己 tune,唔一定成日搵 ML team。",
    ],
    [
        "點樣自動揀 threshold?你會用 Youden J / cost curve?",
        "如果 model retrain,threshold 係咪都要重選?",
    ],
    icon=":material/trending_up:",
)

qa(
    "Q17. 你嗰 3,200 張 patch,train/val/test split 安全咪?"
    "會唔會有 data leakage?",
    [
        "**已做:** 用 3-way split (70 / 15 / 15, seed=42),"
        "`src/splits.py` 係 SSOT,train / val / test 三個 slice 完全"
        "disjoint,test 從未影響過 gradient 或 checkpoint selection。",
        "**仲存在嘅 risk:** 一張原相切 4 個 patch,如果 4 個 patch 分散入 "
        "train / val / test,model 就可能「見過隔壁 patch」,inflate 成個 "
        "metric。",
        "**理想做法:** 按**原相 id** 做 `GroupShuffleSplit`,確保同一張相 "
        "嘅所有 patch 落喺同一 slice。",
        "**Demo 實際:** 而家 split 係 sample-level random,group-level "
        "leakage 未 fix;係 known limitation,下一步可以 extend "
        "`three_way_split_indices` 接受 `group_key` 參數。",
    ],
    [
        "Leakage 嘅 symptom 通常係咩?",
        "產業上 data leakage 仲有咩 common mistake?",
        "GroupShuffleSplit vs StratifiedGroupKFold 你會揀邊個?",
    ],
    icon=":material/security:",
)

st.divider()


# ---------------------------------------------------------------------------
# 6. Grad-CAM / explainability
# ---------------------------------------------------------------------------

st.header("6. Grad-CAM / 可解釋性")

qa(
    "Q18. Grad-CAM 底層原理係咩?",
    [
        "**Goal:** 答 "
        "「model 睇張相邊個位嚟判斷 class c?」",
        "**Step 1:** Forward pass,hook 住 target layer 嘅 activation "
        "`A ∈ R^(C×H×W)` (ResNet18 用 `layer4`,shape `[512,7,7]`)。",
        "**Step 2:** Backward pass `dY_c/dA`,攞到每個 channel 嘅 gradient。",
        "**Step 3:** `w_c = mean(dY_c/dA_c)` — spatial GAP,"
        "每 channel 對 class c 嘅 importance。",
        "**Step 4:** `CAM = ReLU(Σ w_c · A_c)`,normalize 到 [0,1],"
        "bilinear upsample 返 224×224,加 JET colormap blend 落原相。",
    ],
    [
        "ReLU 呢步點解唔可以唔做?",
        "Grad-CAM++ 同 Grad-CAM 有咩分別?",
    ],
    icon=":material/visibility:",
)

qa(
    "Q19. Grad-CAM 有啲咩 limitation?你點同用戶溝通?",
    [
        "**粒度粗:** 7×7 upsample 返 224,本質係 coarse,唔係 pixel-accurate,"
        "唔可以用嚟度裂縫寬度。",
        "**Correlation ≠ causation:** 熱嘅位只係話 gradient 強,"
        "唔一定等於「人眼意義上嘅 crack」。有時 model 喺陰影、接縫學到 cheat。",
        "**Class-specific:** 對不同 class 跑會得到唔同熱區,"
        "要小心解讀講緊邊個 class。",
        "**Production 用法:** Grad-CAM 我當 **visual sanity check** 而唔係 "
        "ground truth,係俾工程師快速判斷「model 係咪睇啱位」,"
        "catch obvious failure mode,例如 model 居然望住天空或者鏡頭 flare。",
    ],
    [
        "你會唔會比較 Integrated Gradients / SHAP / attention rollout?",
        "如果要 pixel-level 解釋,你會點 switch approach?",
    ],
    icon=":material/warning:",
)

qa(
    "Q20. `dominant_quadrant()` 做乜?點解要佢?",
    [
        "**功能:** 將 Grad-CAM 分成 2×2,揀最熱嗰格回 "
        "`\"top-left\"` / `\"top-right\"` / `\"bottom-left\"` / "
        "`\"bottom-right\"` / `\"centre-heavy\"` / `\"uniform\"`。",
        "**點解要:** LLM 睇唔到圖,所以我將「熱點集中喺邊」抽成文字 hint "
        "俾佢,令 AI 檢查備註可以講「熱力集中喺右下角」而唔係亂作。",
        "**實作:** 比較每 quadrant mean 同 overall mean,"
        "細過 `(1+centre_margin)` 就 fallback `centre-heavy`,"
        "避免細差異亂分 quadrant。",
        "**Trade-off:** 好粗略(只得 5 個 bucket),"
        "但係足夠俾 LLM 有 grounding 而唔 hallucinate。",
    ],
    [
        "點解唔直接將 heatmap 以 base64 image 傳俾 multimodal LLM?",
        "你會唔會諗 9-cell grid 定 polar description?",
    ],
    icon=":material/explore:",
)

st.divider()


# ---------------------------------------------------------------------------
# 7. LLM integration
# ---------------------------------------------------------------------------

st.header("7. LLM 整合 (HKBU GenAI)")

qa(
    "Q21. 點解要加 LLM 落一個 CV project 度?",
    [
        "**降低 adoption 門檻:** 工程師習慣睇文字報告多過 JSON,"
        "AI 一段檢查備註可以直接 paste 落佢哋嘅 site report。",
        "**Grounded 而唔係 chat toy:** `build_prediction_context()` "
        "會 pass 實際 label / probabilities / Grad-CAM focus quadrant 俾 LLM,"
        "system prompt 要佢基於呢啲 fact 作答,唔可以亂講。",
        "**雙語:** ZH / EN 兩個 system prompt,工地用廣東話,國際團隊用 English。",
        "**Fail-safe:** LLM error (config error / network error) "
        "只會出 error banner,主 CV pipeline 繼續 work。",
    ],
    [
        "LLM hallucination 嘅 risk 你點管?",
        "如果 offline / no internet,你個 LLM fallback 會點設計?",
    ],
    icon=":material/smart_toy:",
)

qa(
    "Q22. Prompt engineering 你用咗咩技巧?",
    [
        "**Role + Task:** System prompt 明確講「你係一位有經驗嘅混凝土"
        "結構工程師,寫一段 3-5 句嘅現場檢查備註」。",
        "**Structured output:** 要 LLM 按三點出 — 觀察 / 成因 / 建議,"
        "令 output format 穩定。",
        "**Grounding context:** User turn 入面 inject "
        "`[Prediction] [Threshold] [Grad-CAM focus]` 三行,"
        "model 有 fact 可以 anchor,唔易亂作。",
        "**Negative guard:** Prompt 明確叫佢 "
        "「信心低就實話實說,唔好誇張」、"
        "「唔好重複列百分比數字」,呢啲係對常見 LLM failure mode 嘅"
        "pre-empt。",
        "**Chat vs report 用唔同 system prompt:** Chat 短(1-3 句),"
        "Report 長啲(3-5 句),根據 UX 調整。",
    ],
    [
        "你會唔會加 few-shot example 入 system prompt?",
        "點 evaluate prompt quality?",
    ],
    icon=":material/edit_note:",
)

qa(
    "Q23. HKBU GenAI call 嘅 latency、cost、error handling?",
    [
        "**Model:** `gpt-4.1-mini` (via HKBU Azure-style endpoint),"
        "典型 350 tokens,latency 通常 2-5s。",
        "**Cost control:** 每次只喺用戶 click button / 發問 先 call,"
        "唔 pre-generate、唔 stream (demo 夠用),chat history capped 8 turn "
        "防止 payload 膨脹。",
        "**Error classes:** `LLMConfigError` 代表 `.env` 無 `HKBU_API_KEY`,"
        "backend 回 400;`LLMRequestError` 代表 upstream 出錯,"
        "backend 回 502;frontend banner 顯示 detail,不會 crash。",
        "**Safety net:** `.env` 唔 commit (`.gitignore`),"
        "`.env.example` 俾 collaborator 參考。",
    ],
    [
        "生產環境點管 API key rotation?",
        "Streaming response 你會點 implement?",
    ],
    icon=":material/bolt:",
)

st.divider()


# ---------------------------------------------------------------------------
# 8. System architecture
# ---------------------------------------------------------------------------

st.header("8. 系統架構 (FastAPI + Streamlit)")

qa(
    "Q24. 點解要將 frontend / backend 拆開?",
    [
        "**Separation of concerns:** Backend 只管 ML + LLM,"
        "Frontend 只管 UI;future 可以換 UI (e.g. React / mobile) "
        "唔洗郁 ML。",
        "**Dependency weight:** Frontend 唔 need torch / opencv / fastapi,"
        "install footprint 細好多,加快 CI + Docker build。",
        "**Scale out:** Backend 可以 horizontally scale (stateless "
        "forward pass) 獨立於 frontend。",
        "**Testability:** Backend 有 OpenAPI docs + Pydantic schema,"
        "可以用 `pytest` + `TestClient` 單獨 test,唔需要起 Streamlit。",
        "**Trade-off:** 多咗一層 network hop,demo latency 稍微高,"
        "但可接受。",
    ],
    [
        "如果你要加 mobile app,你會 reuse 呢個 API 定 re-design?",
        "Backend 如果要 scale 到 1000 QPS,瓶頸喺邊?",
    ],
    icon=":material/lan:",
)

qa(
    "Q25. API design 你點諗?有咩 trade-off?",
    [
        "**Resource-oriented:** `/api/predict`, `/api/gradcam`, "
        "`/api/segment`, `/api/samples`, `/api/ai/report`, "
        "`/api/ai/chat`, `/api/evaluate`,每個 endpoint 一個職責。",
        "**`/api/segment` 返乜:** 一個 JSON,入面 base64 PNG "
        "(overlay + raw mask) 加 `SegStats` (`crack_pixel_ratio`, "
        "`num_components`, `length_px`, `max_width_px`, image "
        "dimensions)。用 base64 pack 落 JSON 係為咗 frontend 可以"
        "一次 request 就攞齊 overlay + 數字,唔使做 second call。",
        "**Flexible inputs:** `/predict` 同 `/gradcam` 支援兩種 input "
        "— `?sample=<name>`(用 backend bundled sample)或者 "
        "`multipart file`(upload)— demo 場景用 sample 唔需要 round-trip。",
        "**Return type 合適:** JSON for predictions,`image/png` for "
        "Grad-CAM overlay,用 HTTP 本身嘅 content-type 語意。",
        "**Sandboxing:** `/dataset-image` 用 `resolve().relative_to()` "
        "驗 path 必然喺 `DATA_DIR` 入面,避免 path traversal。",
        "**Validation:** Pydantic schema 強制 `threshold ∈ [0,1]` 、"
        "`alpha ∈ [0.05, 0.95]`,出錯會自動 422。",
    ],
    [
        "點解唔用 gRPC?",
        "版本化 / breaking change 你點 handle?(e.g. /api/v2/predict)",
    ],
    icon=":material/api:",
)

qa(
    "Q26. Caching 策略?邊啲要 cache、邊啲唔可以?",
    [
        "**Backend side:**\n"
        "    - `@lru_cache` for model + device(load 一次 serve 無限 request)\n"
        "    - `@lru_cache` for `/api/evaluate` by "
        "(checkpoint mtime + split config),避免重複 forward。",
        "**Frontend side:**\n"
        "    - `@st.cache_data(ttl=60)` for health ping\n"
        "    - `@st.cache_data(ttl=300)` for sample list\n"
        "    - `@st.cache_data` for sample bytes(永 cache,names 係 static)\n"
        "    - 唔 cache prediction / Grad-CAM(設計上用戶可能試唔同 sample "
        "或者 override,每次都想 fresh)",
        "**Cache invalidation:** Evaluation page 有 "
        "「Re-run evaluation」button 可以 `_run_eval_cached.clear()`,"
        "mtime key 令 retrain 後數字自動 refresh。",
    ],
    [
        "如果 backend deploy 上 serverless (e.g. Lambda),"
        "`lru_cache` 仲 work 咪?",
        "Multi-worker (gunicorn) 點保持 cache 共享?",
    ],
    icon=":material/storage:",
)

qa(
    "Q27. 如果要 productionize,你會加咩?",
    [
        "**Auth:** API key / OAuth,最少 backend 要有 per-tenant rate limit。",
        "**Observability:** Structured logging + Prometheus metrics "
        "(`prediction_count`, `predict_latency_ms`, `llm_error_rate`)、"
        "Sentry for exception tracking。",
        "**Model versioning:** MLflow / DVC 管 checkpoint,"
        "每個 prediction response 夾埋 `model_version` 俾 audit。",
        "**Container + CI:** Dockerfile for backend + frontend 分開,"
        "GitHub Actions 跑 `pytest` + `ruff` + `mypy` + build image。",
        "**Deployment:** Backend 上 Kubernetes,HPA 用 request-rate scale,"
        "Frontend 上一個 CDN friendly 嘅 Streamlit replica(或者換 "
        "Next.js / FastAPI 的 HTML)。",
        "**Data pipeline:** Kafka / queue 接工地相機 → S3 + "
        "ingest service → model batch scoring + active learning loop。",
    ],
    [
        "你點 design canary / A/B deployment for a new model?",
        "ML monitoring (data drift / concept drift) 你點捕捉?",
    ],
    icon=":material/deployed_code:",
)

qa(
    "Q28. 點樣保證 backend 同 frontend 嘅 class 次序冇對錯?",
    [
        "**Single source of truth:** `src/constants.py` 有 "
        "`CLASS_NAMES = [\"No Crack\", \"Crack\"]`,backend 同 frontend "
        "都 import 呢個(frontend 唔會 pull 到 torch 因為 `constants.py` 純 "
        "Python)。",
        "**Response 重複標示:** `/api/predict` response 含 "
        "`class_names: [...]`,就算 frontend version 舊,"
        "都可以按 response order 讀,唔會靠 assumption。",
        "**Schema enforced:** Pydantic `PredictionResponse` 固定 field,"
        "breaking change 會 CI fail。",
    ],
    [
        "如果 future 加第三類 (e.g. 'spalling'),你個 API 點 migrate?",
        "Schema evolution 你用咩 tool?(protobuf / JSON schema / OpenAPI)",
    ],
    icon=":material/sync_alt:",
)

st.divider()


# ---------------------------------------------------------------------------
# 9. Limitations & future work
# ---------------------------------------------------------------------------

st.header("9. 限制 / 改進方向")

qa(
    "Q29. 呢個 project 最大嘅 limitation 係咩?",
    [
        "**Domain gap (最大):** Classifier train 喺 METU Ankara facade "
        "close-up,Segmenter train 喺 DeepCrack 馬路 / 一般混凝土;"
        "兩個 training distribution 都唔係 HK MBIS tile-clad 外牆,"
        "真 site accuracy / IoU 一定會跌。",
        "**Defect coverage 窄:** 只 cover **hairline-to-mid cracks**;"
        "**spalling / efflorescence / rust staining** 完全唔 handle,"
        "因為冇公開、redistributable 嘅 HK-labelled 數據。",
        "**Pixel → millimetre 要 on-site calibration:** U-Net 出嘅 "
        "`max_width_px` / `length_px` 係 pixel 值;轉做 mm 要知相機距離 + "
        "sensor 或者 fiducial marker。Demo 唔知,所以 AI report 明確寫 "
        "「pixel-level estimate, requires on-site calibration」,唔直接"
        "答「0.2 mm 合規 / 0.3 mm 超標」。",
        "**In-distribution test only:** val / test 仲係由同一個 upstream "
        "dataset carve 出嚟,未代表真 site distribution。",
        "**Possible group-level leakage** (Q17),同一張相切出嚟嘅 patch "
        "有機會喺 train / val / test 三邊都出現。",
        "**Grad-CAM ≠ ground truth:** Grad-CAM 熱區只講「classifier 邊個 "
        "位 gradient 最強」,U-Net mask 先係 crack pixels;兩者 disagree "
        "好常見,production 要同工程師講清楚。",
        "**LLM hallucination risk:** 就算 grounded,model 都可能出 "
        "plausible but wrong 嘅因果推斷;production 要有免責聲明 + "
        "人工 review loop,AI 報告永遠係 **draft**,要 RSE / AP 蓋章。",
    ],
    [
        "呢啲 limitation 邊個最影響 business?",
        "點樣 measure HK domain gap?",
        "px → mm 你覺得用 ML 定 hardware (laser range finder) 解好?",
    ],
    icon=":material/warning_amber:",
)

qa(
    "Q30. 如果你有多 3 個月,你會做咩?",
    [
        "**Month 1 — HK site data + calibration:** 同 HK 某個 PMSA / "
        "RSE firm 合作,收 500-1000 張真·MBIS 外牆相 + 工程師 label "
        "(crack mask + severity),建立 held-out **HK site test set**,"
        "同時喺每張相放一個 fiducial marker(例如一張 50mm × 50mm "
        "tape)解決 px → mm 問題。",
        "**Month 2 — Multi-defect coverage + domain adaptation:** "
        "將 U-Net extend 做 multi-class (crack / spalling / "
        "efflorescence / rust stain),用 HK data 做 fine-tune;"
        "試 domain-adversarial training (DANN) 或者 simple "
        "photometric augmentation 對抗 HK lighting / tile texture。",
        "**Month 3 — Production scaffold:** Dockerize backend + "
        "frontend、加 Prometheus / Sentry、寫 CI (ruff + mypy + pytest + "
        "TestClient 測 all endpoints)、做一個 minimal React 替代 UI "
        "prove API reusability、加 authentication + model registry "
        "(MLflow / DVC),每個 prediction response 夾埋 `model_version`。",
    ],
    [
        "點解 multi-defect 排第二而唔係第一?",
        "Fiducial marker 實際工地可行咪?有冇其他 px→mm 方案?",
        "React 換 Streamlit 你估會跳啲咩坑?",
    ],
    icon=":material/schedule:",
)

st.divider()


# ---------------------------------------------------------------------------
# 10. Behavioural
# ---------------------------------------------------------------------------

st.header("10. 行為 / 軟性題")

qa(
    "Q31. 整呢個 project 途中,你遇過最辣手嘅問題係咩?點 solve?",
    [
        "**STAR 範例(可按你真實經驗調整):**",
        "**S (Situation):** Frontend Streamlit 同 ML pipeline 一開始 "
        "塞喺一齊,install 成隻 torch 先做到 UI 改動,iteration loop 慢。",
        "**T (Task):** 想將兩邊拆開,令前端 iteration 唔 depend on "
        "ML runtime,亦都方便未來換 UI。",
        "**A (Action):** 抽 `CLASS_NAMES` 去 torch-free `src/constants.py`、"
        "抽 `metrics_at_threshold` 去 pure-numpy `src/metrics.py`;"
        "起 FastAPI backend 包住所有 torch code;frontend 變成 "
        "thin HTTP client;跑 `TestClient` + live curl 做 smoke test。",
        "**R (Result):** 10/10 smoke test pass;frontend install "
        "少咗 2GB dependency;API docs 自動生成;而家加第二個 UI "
        "(e.g. React)完全可行。",
    ],
    [
        "Trade-off 有咩?你點同 team communicate?",
        "如果重做一次,你邊樣會做得唔同?",
    ],
    icon=":material/psychology:",
)

qa(
    "Q32. 你做 design decision 嘅時候,點 balance 「速度交貨」同 「code 品質」?",
    [
        "**Demo 性質的 trade-off:** 呢個 project 先要 showcase end-to-end "
        "流程,所以第一版本我允許 Streamlit monolith;"
        "確定 feature set 穩定之後,先至 refactor 成前後端分離,"
        "避免 over-engineering early。",
        "**守住非 negotiable 嘅 quality:** 即使係 demo,"
        "我都確保(1)seed 可 reproduce、(2)`.env` 唔 commit、"
        "(3)key paths 有 test(smoke test 10 項)、"
        "(4)function 有 docstring。",
        "**Refactor 觸發點:** 當我要加第二 page (Evaluation) + 第二個 "
        "consumer 潛在需求 (API) 嘅時候,就係拆開嘅好時機。",
    ],
    [
        "你點決定呢個 project 夠唔夠 ready 去 demo?",
        "Tech debt 你會記喺邊度?",
    ],
    icon=":material/balance:",
)

qa(
    "Q33. 你覺得自己喺呢個 project 學到最多嘅係咩?",
    [
        "**技術:** Grad-CAM 內部 gradient flow 點揸 hook;"
        "Pydantic + FastAPI 嘅 schema-first design 點樣令前後端解耦。",
        "**設計思維:** 每一個 design decision 都要對得返 "
        "「呢個係 demo,下一步可能 productionize」嘅雙時間軸,"
        "所以要留 extension point(例如 class names 抽出嚟),"
        "但唔可以 over-engineer。",
        "**Communication:** 自己寫 Architecture page 嘅時候,"
        "先 realise「每個 design 嘅 why」比「code 點寫」重要,"
        "Grad-CAM 嘅 Graphviz 圖係 reviewer 理解最快嘅途徑。",
        "**Limitation honesty:** 講清楚 domain gap / in-distribution "
        "test 嘅局限 / group-level leakage 風險,展示 engineering maturity。",
    ],
    [
        "如果重頭嚟,你最想 avoid 嘅 mistake 係咩?",
        "你會點將經驗 apply 落一個 production CV project?",
    ],
    icon=":material/favorite:",
)

qa(
    "Q34. 我(面試官)如果話 accuracy 要 99% 先 ship,你點應對?",
    [
        "**先 clarify:** 「99% 係 accuracy、recall、定 F1?對邊個 class?"
        "Training distribution 定 real-site distribution?」"
        "— 展示 metric 嚴謹性。",
        "**擺事實:** 而家 in-distribution test accuracy ~97%,"
        "但 test set 同 train set 嚟自同一批相,仲未代表真·工地 "
        "distribution;要 build 一個 real-site held-out set 先知真數字。",
        "**攞 roadmap:** 到達 99% 嘅可能 path — "
        "(a) 數據 diversity + active learning、"
        "(b) Ensemble(ResNet18 + ResNet50)、"
        "(c) Segmentation-based voting、"
        "(d) Rule-based post-processing (e.g. min crack length)。",
        "**Re-frame business:** 99% 可能係一個 overspec,"
        "我可以俾 cost curve 俾佢睇 — 97% + Grad-CAM + 人手 review "
        "嘅 TPR 可能已經等同 99% 純 model。",
    ],
    [
        "如果佢堅持 99%,你會 commit 定 push back?",
        "如果 6 個月後 accuracy drift 到 92%,你點 detect?",
    ],
    icon=":material/campaign:",
)

qa(
    "Q35. 俾你一個類似 task (e.g. 外牆剝落偵測),你會點起手?",
    [
        "**Step 1 (1 day) — Clarify scope:** Binary 定 multi-class?"
        "輸入係手機相定 drone?有冇 target precision/recall?"
        "有冇 labelled data?",
        "**Step 2 (1 week) — Baseline first:** 揾 1 個 public dataset + "
        "Resnet18 transfer learning,快速起一個 demo pipeline,"
        "證實 data → train → eval → UI 通。",
        "**Step 3 — Real data + iterate:** 配合 stakeholder 收真數據,"
        "加 augmentation / ensemble,每週做一次 evaluation report。",
        "**Step 4 — Explainability + trust:** Grad-CAM / attention "
        "+ UI 俾 domain expert review,"
        "唔係黑盒 throw over the wall。",
        "**Step 5 — Productionize:** Backend API + monitoring + "
        "model registry + drift alerts,先 ship 到真 user 手。",
    ],
    [
        "如果一開始 labelled data = 0,你會點 bootstrap?",
        "同呢個 crack project 比,邊啲野可以 reuse、邊啲要 rebuild?",
    ],
    icon=":material/trending_flat:",
)

st.divider()


# ---------------------------------------------------------------------------
# 11. Segmentation + HK compliance + Sample strategy
# ---------------------------------------------------------------------------

st.header("11. Segmentation + HK 合規 + Sample 策略")

qa(
    "Q36. 點解喺 classifier 之上再加 segmentation?直接 scale up "
    "classifier 唔得咩?",
    [
        "**Classifier 嘅天花板:** 佢只答 yes/no,report 寫到一句「有 crack」"
        "就冇 evidence;HK MBIS / RSE 要討論 SUC 2013 嘅 **0.2 mm / 0.3 mm "
        "限值**,必須要有 width 數字。",
        "**Segmentation 解到嘅:** Pixel mask 可以直接度 max width、length、"
        "連通分量數目,轉成 `SegStats` 餵 LLM,AI 報告就有 grounded 證據。",
        "**Debug classifier:** 將 Grad-CAM (classifier 注意力) 同 U-Net "
        "mask (真 crack pixels) 並排,如果 Grad-CAM 紅喺天空而 mask 紅"
        "喺牆,就知 classifier 喺 cheat,係一個 cheap 嘅 model auditing "
        "tool。",
        "**Scale up classifier 唔解決問題:** 換 ResNet50 / ViT 只會令 "
        "binary accuracy 略升,but 係 wrong task — you can't measure "
        "width with a classifier。",
    ],
    [
        "如果 classifier 99% 準,segmentation 仲有需要咪?",
        "Grad-CAM vs segmentation mask disagree 點 resolve?",
    ],
    icon=":material/grain:",
)

qa(
    "Q37. 點解自己寫 ~1.9M params 嘅小 U-Net,唔直接用 "
    "`segmentation_models_pytorch` 或者 YOLOv8-seg?",
    [
        "**License constraint:** Ultralytics YOLO (所有 v5-v8 variant) 係 "
        "**AGPL-3.0**,任何 network service 用咗就要 open-source 成個 "
        "stack;呢個 repo 係 MIT,直接撞 license。`smp` 本身 MIT 冇問題,"
        "但佢 default 帶 ImageNet-pretrained backbone,backbone 權重嘅 "
        "license 要逐個睇。",
        "**Size vs task complexity:** Crack 係**二分類、細 object、"
        "低 semantic complexity**,唔需要 ResNet50 或者 EfficientNet "
        "backbone。自家寫嘅 4-stage U-Net (base channel = 16) 只得 "
        "1.9M params,**CPU inference ~300ms**,demo 場景夠用。",
        "**Explainability:** 自己寫嘅 model 冇 black-box dependency,"
        "interview 可以由 forward pass 一直講到 up-sampling,"
        "係 showcase 嘅一部分。",
        "**Joint augmentation:** 用 `torchvision.transforms.v2` + "
        "`tv_tensors.Image/Mask` 做 image-mask 同步 transform;唔會出現 "
        "「image 翻咗但 mask 冇翻」嘅 label leakage。",
    ],
    [
        "Param 1.9M 夠唔夠 capture HK 複雜表面嘅 texture?",
        "如果可以換 license,你會揀 smp 定繼續自家 U-Net?",
    ],
    icon=":material/construction:",
)

qa(
    "Q38. Loss 用 BCE + Dice,metric 用 IoU,點解咁揀?",
    [
        "**BCE 單獨唔夠:** Crack pixels 通常 < 5% 全相,BCE 會俾 model "
        "猜「全黑 mask」就攞到低 loss — 嚴重 class imbalance。",
        "**Dice loss:** `1 - 2|P∩T| / (|P|+|T|)` 係 set-overlap-based,"
        "即使 positive pixels 好少都會 penalise 錯 mask,互補 BCE。"
        "我用 `0.5·BCE + 0.5·Dice`,對 thin structure 比 pure BCE 穩定"
        "好多。",
        "**Metric 用 IoU 而唔用 accuracy:** Pixel-level accuracy 喺 "
        "imbalanced mask 任何時候都 >95%,完全無意義;IoU(Jaccard)"
        "對 small-object segmentation 係 industry standard。",
        "**Early stopping by val IoU:** 每個 epoch 跑 val IoU,"
        "patience = 10,保住最好嗰個 checkpoint,避免 last epoch 過度 "
        "fit 噪音 mask。典型 val IoU ≈ 0.72。",
    ],
    [
        "Tversky / Focal-Tversky loss 你試過咪?邊個會更 work?",
        "IoU @ threshold=0.5,點解唔 sweep threshold?",
    ],
    icon=":material/percent:",
)

qa(
    "Q39. Mask 出咗 max_width_px = 8,你點同工程師講「超唔超 0.2 mm」?",
    [
        "**唔可以直接答。** Pixel 同 mm 之間要**相機幾何參數** — "
        "working distance、focal length、sensor size,或者張相入面有 "
        "scale reference (e.g. 尺、tape)。Demo 冇呢啲,所以 AI report "
        "prompt 明確寫 「`max_width_px` only; mm conversion requires "
        "on-site calibration」。",
        "**LLM 要 guard 住:** `SYSTEM_REPORT_ZH / EN` prompt 寫咗如果 "
        "冇 `px_per_mm`,model 只可以講 「pixel-level evidence suggests "
        "a **wider / thinner / comparable** crack relative to SUC 2013 "
        "thresholds, pending on-site calibration」,唔可以 firm commit "
        "「合規」或「超標」。",
        "**Production 點解:** (a) 工地放 fiducial marker(最直接),"
        "(b) 用 LiDAR / ToF distance,(c) 相機內建 AR anchor,"
        "(d) 拍埋一張 ruler 相做 per-scene calibration。",
        "**實作:** `src.seg_infer.mask_stats(binary_mask, px_per_mm=None)`,"
        "如果 caller 傳 `px_per_mm`,就 augment `max_width_mm` / "
        "`length_mm` 入 stats,否則只回 pixel。",
    ],
    [
        "如果我畀你一張 reference ruler,你點 auto-calibrate?",
        "LLM 永遠要 hedge 咁講,workflow 會唔會 annoy 工程師?",
    ],
    icon=":material/straighten:",
)

qa(
    "Q40. 點解 HK BD defect 相唔直接 commit 入 repo,而要用 "
    "`scripts/fetch_hk_samples.py`?",
    [
        "**Crown Copyright:** HK Buildings Department 網頁上啲圖片(包括 "
        "「Common Building Defects」gallery)係 **Crown Copyright**,"
        "default 唔 grant redistribution license;commit 入 public "
        "GitHub 會違反 copyright。",
        "**Demo 需要 vs 公開 redistribute 嘅矛盾:** 我想 interview 時"
        "show HK-flavoured sample,但又唔想 repo 本身帶 infringing "
        "content。**折衷:** 寫一個 fetcher script,用戶要行 "
        "`python scripts/fetch_hk_samples.py --accept-license` 先拉相"
        "落 `sample_images/hk_bd_*.jpg`;script 打晒 license notice,"
        "要 explicit flag 先 run。",
        "**`.gitignore` 保護:** `sample_images/hk_bd_*.jpg|jpeg|png` 三個 "
        "pattern 全部 ignore,即使 fetch 完都唔會意外 commit。",
        "**UI 對應:** `frontend/🏠_Home.py` 嘅 sample dropdown 會"
        "自動 sort HK BD 相排頭,揀到嗰陣 show `st.info` attribution "
        "banner,URL link 返去 BD website,清楚俾 interviewer 睇到"
        "圖片出處同 copyright。",
        "**Official 寫法:** `sample_images/README.md` + `README.md` + "
        "`data/DATASETS.md` 三個地方都 duplicate 呢個 policy,"
        "避免用戶只讀其中一個就 miss。",
    ],
    [
        "如果 HK BD 攞到 permission,你會直接 commit 咪?",
        "Crown Copyright 同 CC / MIT 點 mix?",
        "有冇諗過 Wikimedia Commons 搵 HK 建築相 (CC-BY / CC-0)?",
    ],
    icon=":material/gavel:",
)

qa(
    "Q41. AI 報告 prompt 點樣將 HK SUC 2013 / PNAP APP-137 / "
    "Buildings Ordinance 寫得對?",
    [
        "**System prompt:** `SYSTEM_REPORT_ZH / EN` 入面 role 係「HK RSE / "
        "AP 合規報告草擬員」,輸出 **structured report**(Report ID / "
        "Date / Location / Element / Inspector / Observation / "
        "Preliminary Severity / Recommended Action / Compliance Note / "
        "Disclaimer),跟 HK 工程師日常嘅報告 format。",
        "**引文準則:** Prompt 明確列出可以 cite 嘅文件 — **SUC 2013** "
        "(Structural Use of Concrete, 裂縫限值)、**PNAP APP-137** "
        "(Practice Note for Authorized Persons 137, MBIS 指引)、"
        "**Buildings Ordinance Cap. 123**;禁止 cite 其他 jurisdiction "
        "或者未 verify 嘅 standard。",
        "**Grounding 而非 RAG:** 我冇 ingest 真 SUC 2013 PDF(license "
        "問題 + demo scope),只係 prompt 提 standard 名同 clause"
        "概念;LLM 靠 training knowledge answer,所以 report 尾會有 "
        "disclaimer「final determination by RSE / AP on site」。",
        "**Segmentation context block:** `build_prediction_context()` 會"
        "append 一個 `[Segmentation]` block — coverage / components / "
        "max width / length / image size — 令 LLM 可以引 pixel evidence "
        "寫 Observation 同 Preliminary Severity,而唔係純粹 probability。",
        "**雙語:** ZH 畀工地用、EN 畀國際團隊或者 insurance 用。"
        "Home page 預設 Cantonese,用 sidebar 改唔到(預設生效)。",
    ],
    [
        "如果要真正做 compliance,應該 RAG SUC 2013 full text?",
        "LLM 引錯 clause 點 detect?",
        "點同 `HKIE / HKIS` 嘅 professional conduct 對齊?",
    ],
    icon=":material/policy:",
)

qa(
    "Q42. Smoke test / CI 點樣 cover 新加嘅 segmentation endpoint?",
    [
        "**`scripts/smoke_test.py` 新 test:** \n"
        "    - `test_seg_module`: 起 synthetic mask (已知 components / "
        "length / max-width),驗 `predict_mask` shape + dtype + resize、"
        "`mask_stats` 數值、`px_per_mm` mm conversion、空 mask edge case、"
        "`overlay_mask` / `draw_mask_contours` 出圖無 crash。\n"
        "    - `test_segment_endpoint`: 用 `fastapi.testclient.TestClient` "
        "in-process call `POST /api/segment`,驗 200 + base64 PNGs + "
        "`stats` keys + `GET /api/health` 報 `has_seg_weights: true`;"
        "如果 checkpoint 唔喺度,就驗 503 回得啱。",
        "**Fallback logic:** Backend 起動 `lifespan` 會 preload 兩個 "
        "model;segmenter missing 唔 crash,只係 log warning + "
        "`has_seg_weights=false`;frontend sidebar 會 show 「Pixel "
        "segmentation: ⚠️ missing weights」,UI 唔會 break。",
        "**CI 可以跑:** `TestClient` 唔需要真起 server,pure Python in-"
        "process call,跑 <10s,啱 GitHub Actions。",
    ],
    [
        "如果 CI agent 冇 `crack_segmenter.pt`,你點 test?(訓練太慢)",
        "Dataset fixture 你會點 pin? (DeepCrack ~700MB)",
    ],
    icon=":material/verified:",
)

st.divider()

st.success(
    "祝你面試順利!🍀 記住最重要嘅係 **講 why,唔係 what** —— "
    "面試官睇 code 睇得到 what,但 why 係你同 resume 嘅分別。",
    icon=":material/emoji_events:",
)
