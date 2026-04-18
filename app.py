"""Streamlit frontend for the concrete crack classifier.

Run with::

    streamlit run app.py

The app lets a user pick a sample image of a concrete surface and returns a
Crack / No Crack label, a confidence score and an optional Grad-CAM heatmap
that highlights the region the model relied on.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from PIL import Image

from src.ai_prompts import (
    SYSTEM_CHAT_EN,
    SYSTEM_CHAT_ZH,
    SYSTEM_REPORT_EN,
    SYSTEM_REPORT_ZH,
    build_prediction_context,
)
from src.gradcam import (
    compute_gradcam_map,
    compute_gradcam_overlay,
    dominant_quadrant,
)
from src.llm import LLMConfigError, LLMRequestError, chat, chat_messages
from src.model import CLASS_NAMES, build_model, load_model
from src.predict import predict

CHAT_HISTORY_LIMIT = 8

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crack_classifier.pt"
SAMPLE_DIR = ROOT / "sample_images"

st.set_page_config(
    page_title="Concrete Crack Detection",
    page_icon=":construction:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def get_model(model_path: Path) -> Optional[torch.nn.Module]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_path.exists():
        model = load_model(model_path, device=device)
    else:
        # Fall back to an untrained model so the UI is still functional for a dry run.
        model = build_model(pretrained=True)
        model.to(device)
        model.eval()
    return model


def list_sample_images() -> List[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(
        p
        for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def render_sidebar() -> dict:
    st.sidebar.header("Settings")
    st.sidebar.markdown(
        "**Model:** ResNet18 fine-tuned on the Mendeley concrete crack dataset."
    )

    show_cam = st.sidebar.toggle("Show Grad-CAM heatmap", value=True)
    cam_alpha = st.sidebar.slider(
        "Heatmap opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )
    threshold = st.sidebar.slider(
        "Crack alert threshold",
        min_value=0.1,
        max_value=0.99,
        value=0.5,
        step=0.01,
        help="If the crack probability is above this, the site is flagged.",
    )

    lang_label = st.sidebar.radio(
        "AI reply language / AI 回覆語言",
        ["廣東話", "English"],
        index=0,
        horizontal=True,
    )
    lang = "zh" if lang_label == "廣東話" else "en"

    if not MODEL_PATH.exists():
        st.sidebar.warning(
            "No trained weights found at `models/crack_classifier.pt`. "
            "The app is running on an untrained model; predictions will be random. "
            "Train with `python -m src.train --data-dir data/`."
        )

    return {
        "show_cam": show_cam,
        "cam_alpha": cam_alpha,
        "threshold": threshold,
        "lang": lang,
    }


def render_result(prediction: dict, threshold: float) -> None:
    prob_crack = prediction["probs"][CLASS_NAMES.index("Crack")]
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
        for name, p in zip(CLASS_NAMES, prediction["probs"]):
            st.write(f"- **{name}**: {p * 100:.2f}%")


def run_inference(image: Image.Image, settings: dict) -> None:
    model = get_model(MODEL_PATH)
    if model is None:
        st.error("Failed to load model.")
        return

    device = next(model.parameters()).device
    with st.spinner("Analysing image..."):
        prediction = predict(image, model=model, device=device)

    cam_hint: Optional[str] = None
    col_img, col_cam = st.columns(2)
    with col_img:
        st.subheader("Uploaded image")
        st.image(image, use_column_width=True)

    with col_cam:
        st.subheader("Model focus (Grad-CAM)")
        if settings["show_cam"]:
            try:
                overlay = compute_gradcam_overlay(
                    image,
                    model=model,
                    class_index=CLASS_NAMES.index("Crack"),
                    device=device,
                    alpha=settings["cam_alpha"],
                )
                st.image(overlay, use_column_width=True)
            except Exception as exc:  # pragma: no cover - defensive UI path
                st.warning(f"Could not render Grad-CAM: {exc}")
        else:
            st.info("Grad-CAM disabled in sidebar.")

    try:
        heatmap = compute_gradcam_map(
            image,
            model=model,
            class_index=CLASS_NAMES.index("Crack"),
            device=device,
        )
        cam_hint = dominant_quadrant(heatmap)
    except Exception:  # pragma: no cover - hint is best-effort only
        cam_hint = None

    st.divider()
    render_result(prediction, threshold=settings["threshold"])

    render_ai_sections(
        prediction=prediction,
        threshold=settings["threshold"],
        lang=settings["lang"],
        cam_hint=cam_hint,
    )


def render_ai_sections(
    prediction: dict,
    threshold: float,
    lang: str,
    cam_hint: Optional[str],
) -> None:
    """Render the AI inspection report + prediction chat box."""

    st.divider()
    st.subheader("AI 檢查備註" if lang == "zh" else "AI Inspection Note")
    report_btn_label = (
        "生成 AI 檢查報告" if lang == "zh" else "Generate AI report"
    )
    if st.button(report_btn_label, use_container_width=True, key="ai_report_btn"):
        spinner_msg = (
            "正在呼叫 HKBU GenAI..." if lang == "zh" else "Calling HKBU GenAI..."
        )
        with st.spinner(spinner_msg):
            try:
                report = chat(
                    system=SYSTEM_REPORT_ZH if lang == "zh" else SYSTEM_REPORT_EN,
                    user=build_prediction_context(
                        prediction, threshold, cam_hint
                    ),
                    max_tokens=350,
                    temperature=0.4,
                )
                st.session_state["last_report"] = report
            except LLMConfigError as exc:
                st.error(str(exc))
            except LLMRequestError as exc:
                st.error(f"LLM request failed: {exc}")

    if "last_report" in st.session_state:
        st.markdown(st.session_state["last_report"])

    st.divider()
    st.subheader("問下個 model / Ask about this prediction")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    placeholder = (
        "例如：呢條裂縫嚴重嗎？ / e.g. Is this crack serious?"
        if lang == "zh"
        else "e.g. Is this crack serious? / 呢條裂縫嚴重嗎？"
    )
    user_msg = st.chat_input(placeholder)
    if not user_msg:
        return

    st.session_state["chat_history"].append(
        {"role": "user", "content": user_msg}
    )

    recent = st.session_state["chat_history"][-CHAT_HISTORY_LIMIT:]
    system_prompt = (SYSTEM_CHAT_ZH if lang == "zh" else SYSTEM_CHAT_EN) + (
        "\n\n" + build_prediction_context(prediction, threshold, cam_hint)
    )
    messages = [{"role": "system", "content": system_prompt}, *recent]

    try:
        reply = chat_messages(messages, max_tokens=350, temperature=0.5)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": reply}
        )
    except LLMConfigError as exc:
        st.session_state["chat_history"].pop()
        st.error(str(exc))
        return
    except LLMRequestError as exc:
        st.session_state["chat_history"].pop()
        st.error(f"LLM request failed: {exc}")
        return

    st.rerun()


def main() -> None:
    st.title("Concrete Crack Detection")
    st.caption(
        "A lightweight computer vision demo for construction inspection: pick "
        "a sample concrete surface image and the model will classify it as "
        "*Crack* or *No Crack* and show where it was looking."
    )

    settings = render_sidebar()

    samples = list_sample_images()
    if not samples:
        st.warning(
            "No sample images found in `sample_images/`. Add some JPG or PNG "
            "concrete surface images to that folder and rerun the app."
        )
        return

    sample_options = ["-- none --"] + [p.name for p in samples]
    choice = st.selectbox("Pick a sample image", sample_options, index=0)
    if choice == "-- none --":
        st.info("Choose a sample from the dropdown to get started.")
        return

    if st.session_state.get("current_sample") != choice:
        st.session_state["current_sample"] = choice
        st.session_state.pop("chat_history", None)
        st.session_state.pop("last_report", None)

    image = Image.open(SAMPLE_DIR / choice).convert("RGB")
    run_inference(image, settings)


if __name__ == "__main__":
    main()
