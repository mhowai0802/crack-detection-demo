"""Streamlit frontend for the concrete crack classifier.

This page is now a thin HTTP client — all model / LLM work happens in
the FastAPI backend (``backend/main.py``). Start the backend first::

    uvicorn backend.main:app --port 8000

Then launch this app::

    streamlit run frontend/🏠_Home.py

Override the backend URL with ``BACKEND_URL`` (env or ``.env``).
"""

from __future__ import annotations

import io
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from PIL import Image

from frontend import api_client
from frontend.api_client import BackendError

st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon=":construction:",
    layout="wide",
)


@st.cache_data(ttl=60)
def _health_cached() -> Optional[api_client.HealthInfo]:
    try:
        return api_client.health()
    except BackendError:
        return None


@st.cache_data(ttl=300)
def _list_samples_cached() -> list[str]:
    return api_client.list_samples()


@st.cache_data(show_spinner=False)
def _sample_bytes_cached(name: str) -> bytes:
    return api_client.sample_image_bytes(name)


DEFAULT_SHOW_CAM: bool = True
DEFAULT_CAM_ALPHA: float = 0.45
DEFAULT_SEG_ALPHA: float = 0.5
DEFAULT_THRESHOLD: float = 0.5
DEFAULT_LANG: str = "zh"


HK_BD_PAGE_URL = (
    "https://www.bd.gov.hk/en/safety-inspection/building-safety/"
    "index_bsi_defects.html"
)

HK_BD_META: dict[str, str] = {
    "hk_bd_overview.jpg": "HK 樓宇缺陷概覽 (Overview)",
    "hk_bd_ns_crack.jpg": "非結構性裂縫 (Non-structural cracks)",
    "hk_bd_spalling.jpg": "混凝土剝落 (Spalling of concrete)",
    "hk_bd_struct_crack.jpg": "結構性裂縫 (Structural cracks)",
    "hk_bd_wall_finish.jpg": "外牆飾面缺陷 (Defective wall finish)",
}


def _sample_display_label(name: str) -> str:
    """Return a friendlier dropdown label for known HK BD samples."""
    if name in HK_BD_META:
        return f"🇭🇰 {name} — {HK_BD_META[name]}"
    return name


def _sort_samples(samples: list[str]) -> list[str]:
    """Put HK BD samples first (demo-worthy), then the rest alphabetically."""
    hk = sorted(s for s in samples if s.startswith("hk_bd_"))
    rest = sorted(s for s in samples if not s.startswith("hk_bd_"))
    return hk + rest


def render_hk_attribution(sample_name: str) -> None:
    """Show a compact attribution banner for HK Buildings Department samples.

    The BD thumbnails are Crown Copyright and NOT redistributed with this
    repo; they are fetched locally via ``scripts/fetch_hk_samples.py``.
    Surfacing the source + license inline makes it obvious during an
    interview demo that we are copyright-aware, and that the HK context
    is a deliberate addition (rather than cherry-picked METU crops).
    """
    caption = HK_BD_META.get(sample_name)
    if not caption:
        return
    st.info(
        f"**Sample source · {caption}**  \n"
        f"Photo © HKSAR Government (Crown Copyright). Fetched locally via "
        f"`scripts/fetch_hk_samples.py` from the HK Buildings Department "
        f"[*Building Defects* reference page]({HK_BD_PAGE_URL}) for this "
        f"interview demo only — **not redistributed by this repo**.",
        icon=":material/public:",
    )


def render_sidebar(
    health: Optional[api_client.HealthInfo],
    samples: Optional[list[str]] = None,
) -> dict:
    """Render the zero-config Home sidebar.

    All inference / display / report knobs are pinned to sensible defaults
    (see ``DEFAULT_*`` constants above) so the sidebar only advertises the
    app's identity, shows runtime health, and explains what the page does.
    If an inspector needs to tweak threshold / language, that belongs on a
    dedicated page (e.g. Evaluation) rather than cluttering the demo Home.

    Takes ``samples`` so the sidebar can surface the HK-BD sample count +
    attribution hint — useful context during an interview demo so the
    reviewer immediately sees which corpus they are looking at.
    """
    sidebar = st.sidebar
    sidebar.title(":construction: Crack Detection")
    sidebar.caption(
        "ResNet18 classifier · Grad-CAM · small U-Net segmenter · HKBU GenAI"
    )
    sidebar.divider()

    seg_on = bool(health and health.has_seg_weights)
    sidebar.markdown(
        "**預設設定 / Fixed settings**\n"
        f"- Grad-CAM heatmap: **on** (opacity {DEFAULT_CAM_ALPHA:.2f})\n"
        "- Pixel segmentation (U-Net): "
        f"**{'on' if seg_on else 'off — checkpoint missing'}**\n"
        f"- Alert threshold: **P(Crack) ≥ {DEFAULT_THRESHOLD:.2f}**\n"
        "- Report language: **廣東話 (Cantonese)**"
    )
    sidebar.caption(
        "想試唔同 threshold 嘅 precision / recall trade-off,開 "
        "**Evaluation** page 睇。"
    )

    sidebar.divider()
    sidebar.markdown("**Sample corpus**")
    if samples is None:
        sidebar.caption(":material/hourglass_empty: loading sample list…")
    else:
        hk_samples = [s for s in samples if s in HK_BD_META]
        metu_samples = [s for s in samples if s not in HK_BD_META]
        sidebar.markdown(
            f"- 🇭🇰 HK BD: **{len(hk_samples)}** / 5\n"
            f"- METU (Özgenel CCIC): **{len(metu_samples)}**"
        )
        if hk_samples:
            sidebar.caption(
                ":material/public: HK BD photos are Crown Copyright · fetched "
                "locally via `scripts/fetch_hk_samples.py` · **not** committed."
            )
        else:
            sidebar.caption(
                ":material/download: Run `python -m scripts.fetch_hk_samples "
                "--accept-license` to add 5 HK Buildings Department demo "
                "photos (local only)."
            )

    sidebar.divider()
    with sidebar.expander(":material/info: About", expanded=False):
        st.markdown(
            "**Architecture:** FastAPI backend (`backend/main.py`) owns the "
            "ResNet18 classifier, Grad-CAM, small U-Net segmenter, and HKBU "
            "GenAI calls. This Streamlit app is a thin HTTP client.\n\n"
            f"**Backend:** `{api_client.BASE_URL}`\n\n"
            "**Pages:**\n"
            "- 🏠 **Home** — classify + Grad-CAM + segmentation + AI HK "
            "  compliance report (Cantonese)\n"
            "- 🏗️ **Architecture** — full pipeline walkthrough with "
            "  Graphviz diagrams\n"
            "- 📊 **Evaluation** — re-score the ResNet18 checkpoint on "
            "  val / test slice\n"
            "- 🎤 **Interview** — Cantonese Q&A prep notes"
        )

    if health is None:
        sidebar.error(
            f"Cannot reach backend at `{api_client.BASE_URL}`. "
            "Start it with `uvicorn backend.main:app --port 8000`.",
            icon=":material/error:",
        )
    elif not health.has_trained_weights:
        sidebar.warning(
            "Backend reports no trained weights at `models/crack_classifier.pt`. "
            "Predictions will be random until you train with "
            "`python -m src.train --data-dir data/`.",
            icon=":material/warning:",
        )
    elif not seg_on:
        sidebar.warning(
            "Segmenter checkpoint missing at `models/crack_segmenter.pt`. "
            "Pixel mask + HK-compliance evidence will be skipped until you "
            "train with `python -m src.seg_train --data-dir data_seg/`.",
            icon=":material/warning:",
        )

    return {
        "show_cam": DEFAULT_SHOW_CAM,
        "cam_alpha": DEFAULT_CAM_ALPHA,
        "threshold": DEFAULT_THRESHOLD,
        "lang": DEFAULT_LANG,
        "has_seg": seg_on,
    }


def render_result(
    prediction: dict, threshold: float, class_names: list[str]
) -> None:
    crack_idx = class_names.index("Crack")
    prob_crack = prediction["probs"][crack_idx]
    flagged = prob_crack >= threshold
    label = "CRACK DETECTED" if flagged else "NO CRACK"

    if flagged:
        st.error(f"### {label}")
    else:
        st.success(f"### {label}")

    col_conf, col_thr = st.columns(2)
    col_conf.metric("Crack probability", f"{prob_crack * 100:.1f}%")
    col_thr.metric("Alert threshold", f"{threshold * 100:.0f}%")
    st.progress(min(max(prob_crack, 0.0), 1.0))

    with st.expander("Raw prediction details"):
        for name, p in zip(class_names, prediction["probs"]):
            st.write(f"- **{name}**: {p * 100:.2f}%")


def run_inference(sample_name: str, image: Image.Image, settings: dict) -> None:
    render_hk_attribution(sample_name)

    try:
        with st.spinner("Analysing image..."):
            prediction = api_client.predict_sample(sample_name)
    except BackendError as exc:
        st.error(f"Predict failed: {exc}")
        return

    col_img, col_cam = st.columns(2)
    with col_img:
        st.subheader("Uploaded image")
        st.image(image, width="stretch")

    with col_cam:
        st.subheader("Model focus (Grad-CAM)")
        if settings["show_cam"]:
            try:
                overlay_bytes = api_client.gradcam_sample(
                    sample_name, alpha=settings["cam_alpha"]
                )
                st.image(
                    Image.open(io.BytesIO(overlay_bytes)),
                    width="stretch",
                )
            except BackendError as exc:
                st.warning(f"Could not render Grad-CAM: {exc}")
        else:
            st.info("Grad-CAM disabled in sidebar.")

    st.divider()
    render_result(
        prediction,
        threshold=settings["threshold"],
        class_names=prediction["class_names"],
    )

    crack_idx = prediction["class_names"].index("Crack")
    flagged = prediction["probs"][crack_idx] >= settings["threshold"]
    seg_stats: Optional[dict] = None
    if flagged and settings.get("has_seg"):
        seg_stats = render_segmentation(sample_name, settings["lang"])

    render_ai_sections(
        prediction=prediction,
        threshold=settings["threshold"],
        lang=settings["lang"],
        cam_hint=prediction.get("gradcam_focus"),
        sample_name=sample_name,
        seg_stats=seg_stats,
    )


def render_segmentation(sample_name: str, lang: str) -> Optional[dict]:
    """Call ``/api/segment`` and render mask overlay + pixel metrics.

    Returned as ``seg_stats`` dict (or ``None`` on failure) so the AI
    report call can inject a ``[Segmentation]`` block into the prompt.
    """
    st.divider()
    st.subheader(
        "裂縫像素分割 (U-Net)" if lang == "zh" else "Crack pixel segmentation (U-Net)"
    )
    st.caption(
        "紅色覆蓋 = U-Net 分割模型預測嘅裂縫像素。"
        "所有數值以像素為單位,須配合現場比例尺先可以換算做 mm。"
        if lang == "zh"
        else
        "Red overlay = pixels predicted as crack by the U-Net. All figures "
        "are in pixel units; an on-site reference scale is required before "
        "converting to millimetres."
    )

    try:
        with st.spinner(
            "Segmenting..." if lang == "en" else "分割緊裂縫像素..."
        ):
            overlay_png, _mask_png, stats = api_client.segment_sample(sample_name)
    except BackendError as exc:
        st.warning(
            ("分割失敗,將唔會有 pixel 證據寫入 AI 報告。" if lang == "zh"
             else "Segmentation failed; pixel evidence will be omitted from "
                  "the AI report.")
            + f"  ({exc})"
        )
        return None

    col_overlay, col_metrics = st.columns([3, 2])
    with col_overlay:
        st.image(Image.open(io.BytesIO(overlay_png)), width="stretch")

    with col_metrics:
        ratio = float(stats.get("crack_pixel_ratio", 0.0)) * 100.0
        max_w = float(stats.get("max_width_px", 0.0))
        length = int(stats.get("length_px", 0))
        components = int(stats.get("num_components", 0))

        st.metric(
            "Crack coverage" if lang == "en" else "裂縫覆蓋率",
            f"{ratio:.2f}%",
        )
        st.metric(
            "Max width (px)" if lang == "en" else "最大闊度 (px)",
            f"{max_w:.1f}",
        )
        st.metric(
            "Length (px, skeleton)" if lang == "en" else "長度 (px,骨架)",
            f"{length}",
        )
        st.metric(
            "Components" if lang == "en" else "分離裂縫數",
            f"{components}",
        )
        st.caption(
            "px → mm 必須現場量度校準 (裂縫尺 / 比例尺)。"
            if lang == "zh"
            else "px → mm conversion requires on-site calibration "
                 "(crack gauge / ruler)."
        )

    return stats


ELEMENT_CHOICES_ZH = [
    "— 未指定 —",
    "樓板 (Slab)",
    "樑 (Beam)",
    "柱 (Column)",
    "牆身 (Wall)",
    "樓梯 (Staircase)",
    "其他 (Other)",
]
ELEMENT_CHOICES_EN = [
    "— Unspecified —",
    "Slab",
    "Beam",
    "Column",
    "Wall",
    "Staircase",
    "Other",
]


def _default_report_id(sample_name: str, today: date) -> str:
    """Derive a stable suggested Report ID from the sample name + date.

    E.g. ``00001.jpg`` + 2026-04-19 -> ``CRK-2026-0419-00001``. The
    inspector can overwrite it freely in the UI.
    """
    stem = Path(sample_name).stem if sample_name else "sample"
    slug = re.sub(r"[^A-Za-z0-9]+", "", stem) or "sample"
    return f"CRK-{today:%Y-%m%d}-{slug[:12]}"


def _element_value(choice: str) -> Optional[str]:
    """Strip the leading placeholder option; return ``None`` if unspecified."""
    if not choice or choice.startswith("—"):
        return None
    return choice


def render_ai_sections(
    prediction: dict,
    threshold: float,
    lang: str,
    cam_hint: Optional[str],
    sample_name: str,
    seg_stats: Optional[dict] = None,
) -> None:
    """Render the HK construction compliance inspection report."""

    st.divider()
    st.subheader(
        "AI 合規檢查報告" if lang == "zh" else "AI Compliance Report"
    )
    st.caption(
        "格式參考香港建造業慣用體例 (SUC 2013 · PNAP APP-137 · "
        "Buildings Ordinance Cap. 123)。僅供初步視覺篩查參考,"
        "正式評估須由註冊結構工程師 (RSE) / 認可人士 (AP) 執行。"
        if lang == "zh"
        else
        "Format follows HK construction compliance conventions (SUC 2013 · "
        "PNAP APP-137 · Buildings Ordinance Cap. 123). Preliminary visual "
        "screening only — formal assessment must be performed by an HK "
        "Registered Structural Engineer (RSE) or Authorised Person (AP)."
    )

    today = date.today()
    element_choices = ELEMENT_CHOICES_ZH if lang == "zh" else ELEMENT_CHOICES_EN
    details_label = "報告資料 (選填)" if lang == "zh" else "Report details (optional)"
    with st.expander(details_label, expanded=False):
        st.caption(
            "以下欄位會加入 AI 報告頂部嘅「報告資料」表格。全部留空亦可,"
            "AI 會直接跳過呢個 section。"
            if lang == "zh"
            else
            "These fields populate the \"Report Details\" table at the top "
            "of the AI report. Leave all blank to skip the section."
        )
        col_id, col_date = st.columns(2)
        report_id = col_id.text_input(
            "Report ID",
            value=_default_report_id(sample_name, today),
            key="hdr_report_id",
        )
        inspection_date = col_date.date_input(
            "Inspection date" if lang == "en" else "檢查日期",
            value=today,
            key="hdr_date",
        )
        location = st.text_input(
            "Location / 位置",
            value="",
            placeholder=(
                "例:香港仔某大廈 A 座 3/F 停車場天花"
                if lang == "zh"
                else "e.g. Block A, 3/F Carpark Soffit, Aberdeen"
            ),
            key="hdr_location",
        )
        col_elem, col_insp = st.columns(2)
        element_choice = col_elem.selectbox(
            "Structural element" if lang == "en" else "構件類別",
            element_choices,
            index=0,
            key="hdr_element",
        )
        inspector = col_insp.text_input(
            "Inspector" if lang == "en" else "檢查員",
            value="",
            placeholder="e.g. W. Lee / RSE No. …",
            key="hdr_inspector",
        )

    header = {
        "report_id": report_id,
        "inspection_date": (
            inspection_date.isoformat() if inspection_date else None
        ),
        "location": location,
        "element": _element_value(element_choice),
        "inspector": inspector,
    }

    report_btn_label = (
        "生成合規檢查報告" if lang == "zh" else "Generate compliance report"
    )
    if st.button(report_btn_label, use_container_width=True, key="ai_report_btn"):
        spinner_msg = (
            "正在呼叫 HKBU GenAI 撰寫報告..."
            if lang == "zh"
            else "Drafting report via HKBU GenAI..."
        )
        with st.spinner(spinner_msg):
            try:
                report = api_client.ai_report(
                    prediction,
                    threshold=threshold,
                    lang=lang,
                    grad_cam_hint=cam_hint,
                    header=header,
                    seg_stats=seg_stats,
                )
                st.session_state["last_report"] = report
            except BackendError as exc:
                st.error(str(exc))

    if "last_report" in st.session_state:
        st.markdown(st.session_state["last_report"])


def main() -> None:
    st.title("Concrete Crack Detection")
    st.caption(
        "A lightweight computer vision demo for construction inspection: pick "
        "a sample concrete surface image and the model will classify it as "
        "*Crack* or *No Crack* and show where it was looking."
    )

    health = _health_cached()

    samples: Optional[list[str]] = None
    samples_error: Optional[str] = None
    if health is not None:
        try:
            samples = _list_samples_cached()
        except BackendError as exc:
            samples_error = str(exc)

    settings = render_sidebar(health, samples)

    if health is None:
        st.error(
            f"Backend unreachable at `{api_client.BASE_URL}`. "
            "Start it first: `uvicorn backend.main:app --port 8000`."
        )
        return

    if samples_error is not None:
        st.error(f"Could not list samples: {samples_error}")
        return

    if not samples:
        st.warning(
            "No sample images returned by the backend. Add JPG/PNG files to "
            "`sample_images/` and refresh."
        )
        return

    ordered = _sort_samples(samples)
    sample_options = ["-- none --"] + ordered
    choice = st.selectbox(
        "Pick a sample image",
        sample_options,
        index=0,
        format_func=lambda n: n if n == "-- none --" else _sample_display_label(n),
        help=(
            "🇭🇰 prefixed entries are real HK Buildings Department defect "
            "photos (fetched locally via scripts/fetch_hk_samples.py, not "
            "redistributed). Others are METU close-ups from the Özgenel "
            "CCIC dataset used to train the classifier."
        ),
    )
    if choice == "-- none --":
        st.info("Choose a sample from the dropdown to get started.")
        return

    if st.session_state.get("current_sample") != choice:
        st.session_state["current_sample"] = choice
        st.session_state.pop("last_report", None)

    try:
        image_bytes = _sample_bytes_cached(choice)
    except BackendError as exc:
        st.error(f"Could not load sample image: {exc}")
        return

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    run_inference(choice, image, settings)


if __name__ == "__main__":
    main()
