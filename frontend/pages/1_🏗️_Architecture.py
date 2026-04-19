"""Architecture overview page for the Streamlit demo (Cantonese)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="架構說明 - 混凝土裂縫偵測",
    page_icon=":construction:",
    layout="wide",
)

sidebar = st.sidebar
sidebar.title(":material/architecture: Architecture")
sidebar.caption("系統架構講解 · how the demo is built")
sidebar.divider()
with sidebar.expander(":material/list: 內容目錄", expanded=True):
    st.markdown(
        """
        1. 整體流程 (Data flow)
        2. 模型 (Model)
        3. 訓練策略 (Training)
        4. 數據集 (Datasets)
        5. 推論流程 (Inference)
        6. Grad-CAM 解釋
        7. Segmentation (U-Net) 分支
        8. 模型評估 (Evaluation 頁)
        9. HKBU GenAI 整合 (AI 功能)
        10. 專案目錄結構
        11. 已知限制 (Limitations)
        """
    )
sidebar.divider()
sidebar.caption(
    "呢頁純文字 + Graphviz,無需要揀相。想試模型就返 Home,"
    "想睇數字就去 Evaluation。"
)

st.title("系統架構")
st.caption(
    "由頭到尾講解個裂縫分類器係點樣砌出嚟:"
    "數據、模型、訓練、推論、解釋、評估,同 HKBU GenAI 整合。"
)

st.markdown(
    """
    呢個 demo 雖然細細個,但係一個幾完整嘅 transfer learning + LLM
    pipeline。下面嘅部份會一步步講清楚,由你揀咗張相到主頁面彈出
    「Crack / No Crack」結果、AI 檢查備註同對話之間,中間究竟發生咗
    啲乜嘢事;另外亦都會講埋 Evaluation 頁點樣獨立 re-score 個 checkpoint。

    > 🧱 **架構:** 前後端已經分開。FastAPI **backend** (`backend/main.py`)
    > 獨家負責 PyTorch 模型、Grad-CAM 同 HKBU GenAI 呼叫;
    > Streamlit **frontend** (`frontend/🏠_Home.py`) 淨係 HTTP client,
    > 透過 `frontend/api_client.py` 打 backend 嘅 `/api/*` endpoint。
    """
)

st.divider()

# -------------------- High-level pipeline --------------------
st.header("1. 整體流程 (Data flow)")

st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#f0f4ff",
              fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        upload   [label="揀張相\\n(sample image)"];
        preprocess [label="預處理\\nResize 224x224\\nImageNet normalise"];
        model    [label="ResNet18 backbone\\n+ 2 類 classifier head"];
        softmax  [label="Softmax\\n[P(No Crack), P(Crack)]"];
        verdict  [label="同 threshold 比較\\n判斷 Crack / No Crack"];
        gradcam  [label="Grad-CAM\\n(喺 layer4 做)", fillcolor="#fff4e6"];
        overlay  [label="熱力圖疊上去\\n(視覺解釋)",
                  fillcolor="#fff4e6"];
        quadrant [label="Dominant quadrant\\n(top-left / ... / centre-heavy)",
                  fillcolor="#fff4e6"];
        seg      [label="U-Net segmenter\\n(只喺 Crack 時 trigger)",
                  fillcolor="#e6f0ff"];
        mask     [label="Binary mask + overlay\\n+ stats (coverage / width_px /\\nlength_px / components)",
                  fillcolor="#e6f0ff"];
        prompt   [label="build_prediction_context\\n(label + probs + focus +\\n[Segmentation] block)",
                  fillcolor="#f3e6ff"];
        llm      [label="HKBU GenAI\\nchat / chat_messages",
                  fillcolor="#f3e6ff"];
        ai       [label="AI 檢查報告 (HK compliance)",
                  fillcolor="#f3e6ff"];

        upload -> preprocess -> model -> softmax -> verdict;
        model -> gradcam -> overlay;
        gradcam -> quadrant -> prompt;
        verdict -> prompt;
        verdict -> seg [label="positive"];
        seg -> mask -> prompt;
        prompt -> llm -> ai;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
    - **藍色分支 (分類):** ResNet18 出 label 同 confidence。
    - **橙色分支 (Grad-CAM):** 用同一次 forward pass 計出嚟嘅 gradient,
      做 Grad-CAM 熱力圖,再提取「邊個象限最熱」嘅 hint。
    - **淺藍分支 (Segmentation):** 預測係 Crack 時,call 細 U-Net
      出 pixel-level mask,再衍生 coverage / max width / length /
      component count 等 metric — 直接對應 HK MBIS / SUC 2013
      「where + how bad」問題。詳情睇第 7 節。
    - **紫色分支 (AI):** 將預測結果 + Grad-CAM focus + segmentation
      stats 打包成一段 context,交俾 HKBU GenAI 生成 HK 合規報告格式
      嘅檢查備註。詳情睇第 9 節。
    """
)

st.divider()

# -------------------- Model --------------------
st.header("2. 模型 (Model)")

col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown(
        """
        **ResNet18 (ImageNet 預訓練) + 自製 classifier head。**

        - Backbone: `torchvision.models.resnet18`,用 `ResNet18_Weights.DEFAULT`
          (即係 ImageNet pretrained 權重)。
        - Feature extractor 出嚟嘅係一個 512 維 vector (global average
          pooling 之後)。
        - Head: 原本用嚟分 1000 類嘅 `fc` layer 會被換成
          `nn.Linear(512, 2)`,出兩個 logit,分別對應
          `[No Crack, Crack]`。
        - 最後用 `softmax` 將兩個 logit 變成機率。

        呢個係典型 transfer learning:backbone 已經識得提取一啲
        common 嘅 visual feature (例如邊緣、紋理、形狀),
        所以就算淨係用咗 3,200 張 patch (HuggingFace mirror),
        fine-tune 出嚟喺 seeded **70 / 15 / 15 train/val/test split**
        都可以做到 **~97% accuracy、Crack F1 ≈ 0.96** on held-out
        **test** slice。想睇實時數字就去 Evaluation 頁;
        section 6 再講個 pipeline 點行。
        """
    )

with col_b:
    st.markdown("**Tensor 喺網絡入面嘅 shape 變化**")
    st.graphviz_chart(
        """
        digraph Model {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fillcolor="#eef5ff",
                  fontname="Helvetica", fontsize=10];
            edge [fontname="Helvetica", fontsize=9];

            input   [label="input\\n[B, 3, 224, 224]"];
            conv1   [label="conv1 + bn + relu\\n[B, 64, 112, 112]"];
            l1      [label="layer1\\n[B, 64, 56, 56]"];
            l2      [label="layer2\\n[B, 128, 28, 28]"];
            l3      [label="layer3\\n[B, 256, 14, 14]"];
            l4      [label="layer4\\n[B, 512, 7, 7]",
                     fillcolor="#fff4e6"];
            pool    [label="avgpool\\n[B, 512, 1, 1]"];
            flat    [label="flatten\\n[B, 512]"];
            fc      [label="fc (new)\\n[B, 2]", fillcolor="#e6ffed"];
            sm      [label="softmax\\n[B, 2]", fillcolor="#e6ffed"];
            cam     [label="Grad-CAM hook",
                     shape=note, fillcolor="#fff0f0"];

            input -> conv1 -> l1 -> l2 -> l3 -> l4 -> pool
                  -> flat -> fc -> sm;
            l4 -> cam [style=dashed, arrowhead=none];
        }
        """,
        use_container_width=True,
    )

st.divider()

# -------------------- Training --------------------
st.header("3. 訓練策略 (Training)")

st.markdown(
    """
    訓練分兩個階段嚟做 (睇 `src/train.py`)。呢個係 small-scale
    transfer learning 其中一個最常見又好用嘅方法。
    """
)

st.graphviz_chart(
    """
    digraph Train {
        rankdir=LR;
        node [shape=box, style="rounded,filled",
              fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        subgraph cluster_p1 {
            label="Phase 1: Warm-up head";
            style="rounded,filled";
            fillcolor="#eef5ff";
            fontname="Helvetica";
            freeze [label="Freeze backbone\\n(conv1 ... layer4)",
                    fillcolor="#dbe7ff"];
            trainfc [label="Train fc only\\nAdam, lr=1e-3",
                     fillcolor="#ffffff"];
            epoch1 [label="freeze-epochs\\n(default 2)",
                    fillcolor="#ffffff"];
            freeze -> trainfc -> epoch1;
        }

        subgraph cluster_p2 {
            label="Phase 2: Full fine-tune";
            style="rounded,filled";
            fillcolor="#fff4e6";
            fontname="Helvetica";
            unfreeze [label="Unfreeze all layers",
                      fillcolor="#ffe0b3"];
            trainall [label="Train whole network\\nAdam, lr=1e-4",
                      fillcolor="#ffffff"];
            epoch2 [label="remaining epochs",
                    fillcolor="#ffffff"];
            unfreeze -> trainall -> epoch2;
        }

        data [label="Train loader\\n(flip / rotate / jitter)",
              shape=cylinder, fillcolor="#e6ffed", style=filled];
        val  [label="Val loader\\n(resize + normalise)",
              shape=cylinder, fillcolor="#e6ffed", style=filled];
        best [label="Save best.pt\\n(highest val acc)",
              shape=note, fillcolor="#fff0f0", style=filled];

        data -> freeze;
        data -> unfreeze;
        epoch1 -> unfreeze;
        epoch2 -> val -> best;
    }
    """,
    use_container_width=True,
)

phase_a, phase_b = st.columns(2)

with phase_a:
    st.subheader("第 1 階段 - 淨係訓練個 head")
    st.markdown(
        """
        - 除咗 `model.fc`,其他所有 parameter 全部 freeze 住。
        - Optimizer: Adam,`lr = 1e-3`。
        - 時長: `--freeze-epochs` (預設係 2)。
        - 目的: 俾個新嘅 2 類 head 先追上已經訓練好嘅 backbone,
          唔好一開始就扯到 backbone 嘅 weight。
        """
    )

with phase_b:
    st.subheader("第 2 階段 - 全網絡 fine-tune")
    st.markdown(
        """
        - 將成個網絡 unfreeze。
        - Optimizer: Adam,`lr = 1e-4` (特登調細啲,保護 pretrained feature)。
        - 時長: 剩落嘅 epoch。
        - 目的: 慢慢將個 backbone 適應返混凝土紋理嘅 statistics。
        """
    )

st.markdown(
    """
    **Augmentation** (淨係 training split 先用): random 水平/垂直翻轉、
    最多 15 度 rotation、輕微 color jitter。
    Val / test 嗰邊就 deterministic,只有 resize + ImageNet normalise,
    確保 3 個 slice 嘅數字可以直接比較。

    **Split**: `src/splits.py` 係 single source of truth — `train.py`
    同 `evaluate.py` 都從呢度攞 `(train_idx, val_idx, test_idx)`。
    預設 70 / 15 / 15,seed = 42。Train 用嚟做 gradient update,
    **Val** 逐個 epoch 監察、揀 best checkpoint,**Test** 完全冇掂過,
    訓練結束先跑一次,俾一個冇 selection bias 嘅 generalisation 數字。

    **Loss**: 用 standard `CrossEntropyLoss`。Mendeley dataset 兩個 class
    係 50/50,所以唔洗做 class weighting。

    ---

    **U-Net segmenter training** (見 `src/seg_train.py`) 獨立另一個
    pipeline,同 classifier 完全分開:

    - **Loss**: BCE-with-logits + Soft Dice (50/50 blend,對 thin-crack
      class imbalance 特別友善)。
    - **Optimiser**: Adam `lr=1e-3` + `CosineAnnealingLR`。
    - **Metric**: pixel-level IoU (crack class)。
    - **早停**: validation IoU 冇進步 `patience` epoch 就停,
      save best 到 `models/crack_segmenter.pt`。
    - **Augmentation**: `torchvision.transforms.v2` + `tv_tensors.Mask`,
      確保 flip / rotate / crop 對 image 同 mask 同步 apply。
    - Typical IoU: M1 CPU / MPS 跑 15-20 epoch 可以 hit
      ≈ 0.72 IoU on DeepCrack val split。
    """
)

st.divider()

# -------------------- Dataset --------------------
st.header("4. 數據集 (Datasets)")

st.markdown(
    """
    Demo 用兩個獨立公開 dataset — 一個訓練 classifier,一個訓練
    segmenter。兩個都 **唔會 redistribute**,只用 script 拉。

    **4.1 Classifier — Özgenel CCIC (Mendeley `5y9wdsg2zt`, v2)**

    - 40,000 張 227×227 RGB 相 (20k Positive / 20k Negative),
      METU 校園混凝土 facade。
    - License: CC-BY-4.0。
    - 預設用 `src.splits.three_way_split_indices(seed=42)`
      切成 70 / 15 / 15 train / val / test。
    - Label 次序: `0 = No Crack`、`1 = Crack`
      (`src.model.CLASS_NAMES`)。

    > ℹ️ **Demo 實際跑緊嘅**: 2.3 GB Mendeley zip 唔易直接落到,
    > 所以 `scripts/prepare_data.py` 會由 HuggingFace
    > `Vizuara/concrete-crack-dataset` (800 張相 + 分割 mask) 抽,
    > 每張切 2×2 patch、用 mask 判斷有冇 crack,整返一個
    > 3,200 張嘅細 dataset。

    **4.2 Segmenter — DeepCrack (Zou 2018)**

    - 537 張 train + 237 張 test 嘅 RGB 相,每張有 pixel-level
      binary crack mask。
    - License: 限制 non-commercial research / 教育用途。
    - `scripts/prepare_seg_data.py` 會下載 GitHub 嗰個 repo,
      解埋入面嵌套嘅 `DeepCrack.zip`,再 split 返
      `data_seg/{train,val,test}/{images,masks}/` (train 80/20 再
      cut val,seed = 42)。

    **4.3 做 HK 樣本**

    - `sample_images/` 裡面淨係 Özgenel CCIC 嘅 CC-BY 相,
      並 **唔會** 夾帶 HK Buildings Department 嘅圖片 (版權不許
      redistribute)。
    - 全部 dataset 出處同 license 見
      [`data/DATASETS.md`](../../data/DATASETS.md)。
    """
)

st.divider()

# -------------------- Inference --------------------
st.header("5. 推論流程 (Inference)")

st.markdown(
    """
    當你喺主頁揀一張相,背後會做以下嘢:
    """
)

st.graphviz_chart(
    """
    digraph Infer {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#f0f4ff",
              fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        pick  [label="User 揀相\\n(sample / upload)"];
        load  [label="PIL.Image.open\\n.convert('RGB')"];
        tfm   [label="get_eval_transform()\\nresize 224 + normalise"];
        bt    [label="unsqueeze(0) -> GPU/CPU"];
        fwd   [label="model(tensor)\\ntorch.no_grad()"];
        sm    [label="F.softmax\\n[p_no, p_crack]"];
        thr   [label="p_crack >= threshold?",
               shape=diamond, fillcolor="#fff4e6"];
        yes   [label="Crack (red banner)",
               fillcolor="#ffe0e0"];
        no    [label="No Crack (green banner)",
               fillcolor="#e0ffe0"];

        pick -> load -> tfm -> bt -> fwd -> sm -> thr;
        thr -> yes [label="yes"];
        thr -> no  [label="no"];
    }
    """,
    use_container_width=True,
)

st.code(
    """image = Image.open(selected_sample).convert("RGB")

# src/predict.py
tensor = get_eval_transform()(image).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0].tolist()

# app.py
prob_crack = probs[CLASS_NAMES.index("Crack")]
flagged    = prob_crack >= sidebar_threshold""",
    language="python",
)

st.markdown(
    """
    Sidebar 入面嘅 **「Crack alert threshold」** slider
    可以喺唔洗再訓練模型嘅情況下,調整最終嘅 decision boundary。
    實際用嚟調 precision / recall trade-off 都好方便 ——
    例如要做安全檢查嘅話,可以將 threshold 調細啲,令佢更容易 trigger 警報。
    """
)

st.divider()

# -------------------- Grad-CAM --------------------
st.header("6. Grad-CAM 解釋")

st.markdown(
    """
    Grad-CAM 答嘅問題係:**「個 model 究竟望緊張相邊個位嚟判斷 Crack?」**
    實作喺 `src/gradcam.py`。

    **流程係咁樣** (以 「Crack」 呢個 class 嘅 score 為例):

    1. 喺 `model.layer4` 掛一個 **forward hook**,截低
       activation `A`,shape 係 `[512, 7, 7]`。
    2. 喺同一個 layer 掛一個 **backward hook**,截低
       gradient `dY/dA`,其中 `Y` 係 Crack 嘅 logit。
    3. 將 gradient 喺 spatial 維度做 global average pooling,
       得到每個 channel 一個權重: `w_c = mean(dY/dA_c)`。
    4. 用呢啲權重線性組合 channel:
       `cam = ReLU(Σ w_c * A_c)`,shape 係 `[7, 7]`。
    5. Normalise 去 `[0, 1]`,再 bilinear interpolate 放大返原本 resolution,
       加 JET colormap,最後 blend 去原相上面。

    紅色位置代表 「Crack」 嘅 evidence 最強;藍色就代表對 prediction
    幫助好細。
    """
)

st.graphviz_chart(
    """
    digraph GradCAM {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#f0f4ff",
              fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", fontsize=9];

        img   [label="Input image\\n[1, 3, 224, 224]"];
        fwd   [label="Forward pass\\n(register hooks on layer4)",
               fillcolor="#fff4e6"];
        act   [label="Activations A\\n[512, 7, 7]"];
        logit [label="Crack logit Y"];
        back  [label="Y.backward()"];
        grad  [label="Gradients dY/dA\\n[512, 7, 7]"];
        w     [label="w_c = mean(dY/dA_c)\\n(channel weights)"];
        comb  [label="cam = ReLU(Sum w_c * A_c)\\n[7, 7]"];
        norm  [label="Normalise [0, 1]"];
        up    [label="Bilinear upsample\\n224 x 224"];
        cmap  [label="JET colormap + blend",
               fillcolor="#ffe0e0"];
        out   [label="Overlay image",
               shape=note, fillcolor="#e6ffed"];

        img -> fwd;
        fwd -> act;
        fwd -> logit -> back -> grad;
        act -> comb;
        grad -> w -> comb;
        comb -> norm -> up -> cmap -> out;
    }
    """,
    use_container_width=True,
)

st.info(
    "Grad-CAM 係一種視覺 sanity check,唔係 pixel-level 嘅裂縫 mask。"
    "用佢嚟建立對模型嘅信心、捉到明顯嘅出錯情況 (例如 model "
    "居然望住陰影嚟判斷) 就好好,但唔好攞嚟做精確量度。",
    icon=":material/info:",
)

st.divider()

# -------------------- Segmentation --------------------
st.header("7. Segmentation (U-Net) 分支")

st.markdown(
    """
    為咗答 HK MBIS 真正關心嘅「**邊度爆 + 有幾嚴重**」問題,
    classifier 出 Crack 之後會 trigger 一個獨立嘅 **pixel-level
    segmenter**。相關檔案:

    - `src/seg_model.py` — 自家整嘅小型 U-Net (~1.9M params),
      4 層 encoder / decoder + bottleneck,`base_channels=16`,
      專登細啲方便 CPU inference。
    - `src/seg_dataset.py` — 用 `torchvision.transforms.v2` +
      `tv_tensors.Mask` 做 **image / mask 同步** augmentation
      (flip / rotate / crop 保證對齊)。
    - `src/seg_train.py` — BCE + Dice loss、Adam + cosine LR、
      早停 (validation IoU)。
    - `src/seg_infer.py` — `predict_mask()` 出 binary mask,
      `mask_stats()` 用 `cv2.connectedComponents` +
      `cv2.distanceTransform` + 形態學 skeleton 計出 coverage、
      component count、area、**max width (px)**、
      **length (px)**。
    - `src/seg_viz.py` — 用半透明紅色疊罩嚟 render overlay。
    - `backend/routers/segment.py` — `POST /api/segment`
      (base64 overlay + base64 mask + stats dict)。
    """
)

st.graphviz_chart(
    """
    digraph Seg {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#e6f0ff",
              fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", fontsize=9];

        img    [label="Input image\\n(PIL RGB)"];
        resize [label="Resize 384x384\\n+ normalise"];
        unet   [label="U-Net forward\\n[1, 1, 384, 384] logits"];
        sig    [label="sigmoid + threshold 0.5"];
        upsz   [label="Resize back to\\noriginal H x W"];
        stats  [label="mask_stats()\\ncoverage / max_width_px /\\nlength_px / components",
                fillcolor="#fff4e6"];
        overlay[label="overlay_mask()\\n(red tint, alpha=0.5)",
                fillcolor="#fff4e6"];
        resp   [label="SegmentResponse\\n(overlay_png_b64 +\\n mask_png_b64 + stats)",
                shape=note, fillcolor="#e6ffed"];

        img -> resize -> unet -> sig -> upsz;
        upsz -> stats;
        upsz -> overlay;
        stats -> resp;
        overlay -> resp;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
    **點解揀 segmentation,唔揀 YOLO?**

    - **License**: Ultralytics YOLO 係 **AGPL-3.0**,同 demo repo 嘅
      **MIT** 有衝突;DeepCrack / CRACK500 等 segmentation dataset
      license 相對 friendly。
    - **合規對應**: Pixel mask 可以直接由 `max_width_px` 估 crack
      width (配合現場 calibration),對應 SUC 2013 約 0.2 / 0.3 mm
      嘅限值;YOLO bounding box 做唔到 pixel-accurate 嘅量度。
    - **Side-by-side**: Grad-CAM 答「Classifier 望緊邊?」,
      Segmentation 答「裂縫 pixel 喺邊度?」 — 兩者並排一齊睇,
      就可以肉眼 debug classifier 係咪 look at the right thing。
    """
)

st.info(
    "px → mm 換算要現場校準 (相機焦距、拍攝距離、或者入鏡擺條尺/"
    "crack gauge)。所以 AI 報告同 UI 兩邊都 **只** 顯示 px，"
    "再加警示 caption 提醒用家要自己做 calibration — 唔會假扮出 mm。",
    icon=":material/info:",
)

st.divider()

# -------------------- Evaluation --------------------
st.header("8. 模型評估 (Evaluation 頁)")

st.markdown(
    """
    `frontend/pages/2_📊_Evaluation.py` 係一個獨立 dashboard,call
    backend 嘅 `POST /api/evaluate` 再 re-score 現時
    `models/crack_classifier.pt`。邏輯抽咗去 `src/evaluate.py`
    (重用 `src/splits.py` 嘅 split 函數),所以 training、
    evaluation、notebook、smoke test 睇嘅係同一份 index list。

    **Pipeline 嘅重點:**

    1. 用同 training 時 **一樣 seed 嘅 70 / 15 / 15 split**
       (`src.splits.three_way_split_indices(seed=42)`),確保
       度量到嘅係真 held-out set,唔係睇過嘅訓練相。
    2. 用 sidebar 揀要 score 邊個 slice:
       - **Test** (預設) — 完全冇掂過,俾一個冇 selection bias 嘅
         generalisation 數字。
       - **Val** — 揀 best checkpoint 嗰個 slice,數字會樂觀少少。
    3. 跑一次 forward pass 攞到選中 slice 嘅 softmax 機率
       `probs[N, 2]`,連埋每張相嘅 file path 儲起。
    4. 任何同 threshold 相關嘅 metric (accuracy、confusion matrix、
       per-class precision / recall / F1) 之後都可以
       `metrics_at_threshold(probs, targets, t)` 即時計出嚟,
       唔使 forward 多次 —— 所以 sidebar 嘅 threshold slider 拉到
       邊,數字就跳到邊。
    """
)

st.graphviz_chart(
    """
    digraph Eval {
        rankdir=LR;
        node [shape=box, style="rounded,filled",
              fontname="Helvetica", fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        ckpt [label="models/crack_classifier.pt",
              shape=cylinder, fillcolor="#e6ffed"];
        data [label="data/Positive + Negative\\n(3,200 patches)",
              shape=cylinder, fillcolor="#e6ffed"];
        split [label="seeded 70/15/15 split\\n(three_way_split_indices, seed=42)",
               fillcolor="#eef5ff"];
        pick  [label="pick slice\\n(val | test)",
               fillcolor="#eef5ff"];
        fwd   [label="forward pass (no_grad)\\n-> probs[N, 2]",
               fillcolor="#eef5ff"];
        store [label="EvaluationResult\\n(probs, targets, paths, split)",
               shape=note, fillcolor="#fff0f0"];
        thr   [label="metrics_at_threshold(t)",
               fillcolor="#fff4e6"];
        ui    [label="Accuracy / P / R / F1\\nConfusion matrix\\nThreshold sweep\\nWorst errors",
               fillcolor="#fff4e6"];

        data -> split -> pick -> fwd;
        ckpt -> fwd -> store -> thr -> ui;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
    頁面入面你會見到:

    - **Dataset 同 split** — total / train / val / test + 你 score
      緊邊個 slice 嘅 class balance。
    - **Headline metrics** — Accuracy、Crack recall (最重要,漏報就
      大鑊)、Crack precision。
    - **Classification report** — 兩個 class 齊齊有 precision / recall /
      F1 / support。
    - **Confusion matrix** — 2×2 table 配埋 TP / FP / FN / TN
      caption,同旁邊個 bar chart 對照。
    - **Threshold sweep** — 掃 0.05 → 0.95 睇 accuracy / crack P / R /
      F1 點樣換,幫手決定個 sidebar threshold 調幾多最合理。
    - **錯得最離譜嘅樣本** — 按「信心離 threshold 有幾遠」排序,
      click 到每張真係有 predictions wrong 嘅 patch。
    """
)

st.caption(
    "Cache: `@st.cache_data` 嘅 key 包括 checkpoint 個 mtime + split 設定,"
    "所以 retrain 完拎返新 .pt,refresh 一次頁就自動 invalidate,"
    "唔會俾你睇到舊 model 嘅數字。"
)

st.divider()

# -------------------- AI features --------------------
st.header("9. HKBU GenAI 整合 (AI 功能)")

st.markdown(
    """
    主頁面除咗 Crack / No Crack 之外,仲會將預測結果交俾 HKBU GenAI
    (預設 `gpt-4.1-mini`,經 Azure-style REST endpoint)。
    嚟源碼分三層:

    - `src/llm.py` — thin wrapper,load `.env`、check `HKBU_API_KEY`、
      共用一個 `_post_chat()` helper,expose 兩個 public function:
      `chat(system, user, ...)` (single-turn) 同
      `chat_messages(messages, ...)` (multi-turn history)。
    - `src/ai_prompts.py` — 中 / 英 system prompt 同
      `build_prediction_context(prediction, threshold, grad_cam_hint)`,
      將預測打包成一段簡潔 context,避免個 model 亂估數字。
    - `app.py` — button-gated 「生成 AI 檢查報告」section 同
      `st.chat_input` 對答框,history 最多保留 8 回合,換 sample
      自動清 chat + report,避免上一張相嘅 context 捲入新題。
    """
)

st.graphviz_chart(
    """
    digraph LLM {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="Helvetica",
              fontsize=11];
        edge [fontname="Helvetica", fontsize=10];

        pred [label="Prediction\\nlabel + probs"];
        cam  [label="Grad-CAM focus\\n(dominant_quadrant)"];
        ctx  [label="build_prediction_context()\\n[Prediction] / [Threshold] /\\n[Grad-CAM focus]",
              fillcolor="#fff4e6"];
        sys  [label="SYSTEM_REPORT_* /\\nSYSTEM_CHAT_*",
              fillcolor="#fff4e6"];
        llm  [label="HKBU GenAI\\nchat() / chat_messages()",
              fillcolor="#f3e6ff"];
        rep  [label="AI 檢查備註\\n(session_state['last_report'])",
              fillcolor="#e6ffed"];
        chat [label="問答歷史\\n(session_state['chat_history'])",
              fillcolor="#e6ffed"];

        pred -> ctx;
        cam  -> ctx;
        sys  -> llm;
        ctx  -> llm;
        llm  -> rep;
        llm  -> chat;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
    **Cost / latency note.** Model 由 `.env` 嘅 `HKBU_MODEL` 控制
    (目前 `gpt-4.1-mini`,快 + 平)。報告每撳一次 button 先叫一次
    API,大概 350 tokens;chat 每次用戶出 message 先會 call。
    兩邊都用 `try/except LLMConfigError / LLMRequestError`,
    如果 API key 唔見或者 endpoint 出錯,主頁會係紅色 banner
    提示,完全唔會 crash 成個 app。
    """
)

st.divider()

# -------------------- Repo layout --------------------
st.header("10. 專案目錄結構")

st.code(
    """.
backend/               # FastAPI 服務 (PyTorch + Grad-CAM + U-Net + HKBU GenAI)
  main.py              # FastAPI app + CORS + routers + lifespan preload
  config.py            # 環境變數 / 路徑 (含 SEG_MODEL_PATH)
  deps.py              # get_model / get_seg_model (lru_cache) + helpers
  schemas.py           # Pydantic: SegStats / SegmentResponse / ...
  routers/
    health.py          # GET  /api/health (含 has_seg_weights)
    samples.py         # GET  /api/samples  /  /api/samples/{name}
    predict.py         # POST /api/predict  /  /api/gradcam
    segment.py         # POST /api/segment  (U-Net overlay + mask + stats)
    ai.py              # POST /api/ai/report  /  /api/ai/chat
    evaluate.py        # POST /api/evaluate  +  /api/dataset-image
frontend/                # Streamlit 前端 (HTTP client,唔用 torch)
  🏠_Home.py             # 揀相 + 預測 + Grad-CAM + Segmentation + AI 報告
  api_client.py          # requests wrapper (含 segment_sample / segment_upload)
  pages/
    1_🏗️_Architecture.py  # <- 你而家睇緊
    2_📊_Evaluation.py    # 用 /api/evaluate 重新評估 classifier
src/                   # 共享 ML code (backend import 嚟用)
  constants.py         # CLASS_NAMES 等無 torch 依賴嘅 constant
  model.py             # ResNet18 + classifier head + load/save
  dataset.py           # CrackDataset + train/eval transform
  train.py             # 兩階段 classifier training CLI
  predict.py           # 單張相 classifier 推論 helper
  gradcam.py           # Grad-CAM overlay / raw map / dominant_quadrant
  seg_model.py         # 小型 U-Net + load_seg_model
  seg_dataset.py       # CrackSegDataset + v2 transform (image+mask)
  seg_train.py         # BCE+Dice + IoU 早停嘅 segmentation training CLI
  seg_infer.py         # predict_mask + mask_stats (cv2-based)
  seg_viz.py           # overlay_mask / draw_mask_contours
  evaluate.py          # classifier re-scoring (val / test split)
  splits.py            # SSOT 70/15/15 train/val/test
  metrics.py           # metrics_at_threshold + ROC / AUC
  llm.py               # HKBU GenAI REST wrapper
  ai_prompts.py        # ZH/EN prompt + build_prediction_context
                       #   (含 [Segmentation] block)
scripts/
  prepare_data.py      # 拉 HuggingFace classification dataset
  prepare_seg_data.py  # 拉 DeepCrack + 整返 data_seg/ split
  smoke_test.py        # 跑一次 classifier + segmenter sanity
notebooks/
  01_explore_and_train.ipynb
sample_images/         # 主頁 dropdown 用嘅示範相 (Özgenel CC-BY only)
models/                # 訓練好嘅 .pt weight (gitignored)
  crack_classifier.pt  #   ResNet18 分類 head
  crack_segmenter.pt   #   U-Net pixel mask
data/                  # 分類原始 dataset (gitignored, 有 DATASETS.md)
data_seg/              # DeepCrack split 出嚟嘅 segmentation data (gitignored)
.env                   # HKBU_API_KEY / BACKEND_URL 等 secret (gitignored)
.env.example           # 俾其他人參考嘅 template
README.md""",
    language="text",
)

st.divider()

# -------------------- Limitations --------------------
st.header("11. 已知限制 (Limitations)")

st.markdown(
    """
    - **Domain gap (HK-specific)。** Classifier 嚟自 METU 校園 (Özgenel),
      segmenter 嚟自 DeepCrack (路面 + 一般混凝土),兩個都 **唔係 HK**。
      真正 MBIS target 係 70s-80s 瓷磚或批盪 facade,仲會有滲水痕、
      efflorescence、algae、冷氣機水漬 — 呢啲 demo 係見唔到嘅。
    - **Defect 類型覆蓋好窄。** Model 只分「crack vs 冇 crack」,
      唔處理 spalling、rebar rust staining、efflorescence、
      honeycombing、tile drummy、algae。真實 HK MBIS 報告好多時
      cracks 其實係小眾,呢啲先係主角。
    - **Px,唔係 mm。** Segmenter 只出 `max_width_px` / `length_px`,
      冇 camera intrinsics 或者現場擺條尺/crack gauge,單張相決定
      唔到佢係 SUC 2013 0.2 mm / 0.3 mm 入面定邊度。**AI 報告有
      warning caveat 提醒要 on-site calibration。**
    - **單張相決策。** 冇 video、冇多角度 aggregation,所以方向、
      深度、活動性 (是否仍在擴張) 完全判斷唔到。
    - **Confidence 未 calibrate。** softmax probability 同 mask
      threshold 都係 uncalibrated;當佢係 ranking knob 就岩,
      當真實概率就有 bias。
    - **Grad-CAM ≠ segmentation。** Grad-CAM 答「Classifier 望緊邊?」,
      U-Net 答「Crack pixel 喺邊?」 — 兩者並排畀 reviewer 肉眼對
      得返個意思,唔好當做同一件事。
    - **Val vs Test 係 in-distribution。** 70/15/15 split + seed
      42,training 時揀 best-val、最尾先跑 test,所以 Evaluation
      頁報嘅係冇 selection bias 嘅 generalisation 估計 — 但 test
      同 train 同部相機、同個 dataset,真正 OOD (地盤實拍) 就要
      自己整個 site test set 先算得準。
    - **LLM 會錯。** HKBU GenAI 只睇到 prediction + seg stats 字串,
      冇掂過原圖。報告入面嘅 HK 條文引用係 prompt engineering 出嚟,
      唔代替 RSE / AP judgement。`.env` 無 `HKBU_API_KEY` 就會 fallback
      到 error banner,唔會 crash。
    - **Code 同 Dataset license 分開。** 整個 repo code 係 MIT,
      但 Özgenel CC-BY 同 DeepCrack non-commercial 限制仍然
      apply 喺 dataset 上面 — 詳情睇 `data/DATASETS.md`。
    """
)
