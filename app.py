"""Streamlit frontend for the concrete crack classifier.

Run with::

    streamlit run app.py

The app lets a user upload (or pick a sample) image of a concrete surface and
returns a Crack / No Crack label, a confidence score and an optional Grad-CAM
heatmap that highlights the region the model relied on.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from PIL import Image

from src.gradcam import compute_gradcam_overlay
from src.model import CLASS_NAMES, build_model, load_model
from src.predict import predict

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

    if not MODEL_PATH.exists():
        st.sidebar.warning(
            "No trained weights found at `models/crack_classifier.pt`. "
            "The app is running on an untrained model; predictions will be random. "
            "Train with `python -m src.train --data-dir data/`."
        )

    return {"show_cam": show_cam, "cam_alpha": cam_alpha, "threshold": threshold}


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

    st.divider()
    render_result(prediction, threshold=settings["threshold"])


def main() -> None:
    st.title("Concrete Crack Detection")
    st.caption(
        "A lightweight computer vision demo for construction inspection: upload "
        "a photo of a concrete surface and the model will classify it as *Crack* "
        "or *No Crack* and show where it was looking."
    )

    settings = render_sidebar()

    uploaded = st.file_uploader(
        "Upload a concrete surface image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

    samples = list_sample_images()
    selected_sample: Optional[Path] = None
    if samples:
        sample_options = ["-- none --"] + [p.name for p in samples]
        choice = st.selectbox("...or pick a sample image", sample_options, index=0)
        if choice != "-- none --":
            selected_sample = SAMPLE_DIR / choice

    image: Optional[Image.Image] = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif selected_sample is not None:
        image = Image.open(selected_sample).convert("RGB")

    if image is None:
        st.info("Upload an image or choose a sample to get started.")
        return

    run_inference(image, settings)


if __name__ == "__main__":
    main()
