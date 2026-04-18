"""Architecture overview page for the Streamlit demo (Cantonese)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="架構說明 - 混凝土裂縫偵測",
    page_icon=":construction:",
    layout="wide",
)

st.title("系統架構")
st.caption(
    "由頭到尾講解個裂縫分類器係點樣砌出嚟:"
    "數據、模型、訓練、推論同解釋。"
)

st.markdown(
    """
    呢個 demo 雖然細細個,但係一個幾實用嘅 transfer learning pipeline。
    下面嘅部份會一步步講清楚,由你揀咗張相到主頁面彈出「Crack / No Crack」
    結果之間,中間究竟發生咗啲乜嘢事。
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

        upload   [label="揀張相\\n(JPG / PNG)"];
        preprocess [label="預處理\\nResize 224x224\\nImageNet normalise"];
        model    [label="ResNet18 backbone\\n+ 2 類 classifier head"];
        softmax  [label="Softmax\\n[P(No Crack), P(Crack)]"];
        verdict  [label="同 threshold 比較\\n判斷 Crack / No Crack"];
        gradcam  [label="Grad-CAM\\n(喺 layer4 做)", fillcolor="#fff4e6"];
        overlay  [label="熱力圖疊上去\\n(視覺解釋)",
                  fillcolor="#fff4e6"];

        upload -> preprocess -> model -> softmax -> verdict;
        model -> gradcam -> overlay;
    }
    """,
    use_container_width=True,
)

st.markdown(
    """
    - **藍色分支 (分類):** 出 label 同 confidence。
    - **橙色分支 (解釋):** 用同一次 forward pass 計出嚟嘅 gradient,
      做 Grad-CAM 熱力圖。
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
        所以唔洗好多 crack 相,都可以 fine-tune 到成 98% 以上
        val accuracy (用返 Mendeley 全個 dataset 嘅話)。
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
    Validation 嗰邊就 deterministic,只有 resize + ImageNet normalise。

    **Loss**: 用 standard `CrossEntropyLoss`。Mendeley dataset 兩個 class
    係 50/50,所以唔洗做 class weighting。
    """
)

st.divider()

# -------------------- Dataset --------------------
st.header("4. 數據集 (Dataset)")

st.markdown(
    """
    **Concrete Crack Images for Classification** (作者: Çağlar Fırat
    Özgenel,Mendeley Data id `5y9wdsg2zt`,version 2)。

    - 40,000 張 227x227 RGB 相
    - 20,000 張 **Positive** (有裂縫) / 20,000 張 **Negative** (冇裂縫)
    - License: CC BY 4.0

    `src.dataset.CrackDataset` 會將 `Positive/` 同 `Negative/`
    兩個 folder 打平成一個 index,再 apply `torchvision.transforms`
    嘅 train 或者 eval transform。

    Label 跟 `src.model.CLASS_NAMES` 嘅次序:
    `0 = No Crack`、`1 = Crack`。

    > ℹ️ **個 demo 實際用緊嘅係乜?** 2.3 GB 嘅 Mendeley zip 而家唔易直接落到,
    > 所以呢個 demo 用咗 HuggingFace 上面 `Vizuara/concrete-crack-dataset`
    > (800 張有裂縫相 + segmentation mask),跟住用個 script
    > `scripts/prepare_data.py` 將每張相切成 2x2 patch,用對應 mask
    > 判斷每個 patch 係咪 crack,再整返一個 3,200 張細 dataset。
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

# -------------------- Repo layout --------------------
st.header("7. 專案目錄結構")

st.code(
    """.
app.py                 # Streamlit 主頁 (揀相 + 預測 + Grad-CAM)
pages/
  1_Architecture.py    # <- 你而家睇緊呢一頁
src/
  model.py             # ResNet18 + 新 classifier head,load/save helper
  dataset.py           # CrackDataset 同 train/eval transform
  train.py             # 兩階段 CLI training script
  predict.py           # 單張相嘅推論 helper
  gradcam.py           # Grad-CAM 同 overlay 實作
scripts/
  prepare_data.py      # 由 HuggingFace parquet 整返個分類 dataset
notebooks/
  01_explore_and_train.ipynb
sample_images/         # 主頁 dropdown 用嘅示範相
models/                # 訓練好嘅 .pt weight (gitignored)
data/                  # 原始 / 生成嘅 dataset (gitignored)
requirements.txt
README.md""",
    language="text",
)

st.divider()

# -------------------- Limitations --------------------
st.header("8. 已知限制 (Limitations)")

st.markdown(
    """
    - **Domain gap (分佈差異)。** 訓練相都係乾淨、近距離、光線均勻嘅
      crop;真實地盤相有雜物 (鋼筋、陰影、人、工具)、
      反光同 motion blur,預期 accuracy 會跌明顯一截。
    - **淨係出 binary label。** 個 model 唔會 localise、量度、
      或者判斷裂縫嚴重程度;表面裂痕同結構性裂縫睇落好似。
    - **Confidence 未 calibrate。** softmax 個 probability *唔係*
      真實世界 damage 嘅機率,threshold 要當成 operational knob 去調,
      唔好當佢係機率解讀。
    - **Grad-CAM 係粗粒度嘅。** Upsample 之前先至 7x7,
      佢係一種「解釋」,唔係 segmentation mask。
    - **淨係針對混凝土。** 磚、柏油、木,統統超出範圍。
    """
)
