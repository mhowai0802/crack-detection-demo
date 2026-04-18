# Concrete Crack Detection

A lightweight computer vision demo built for a construction industry use case:
given a photo of a concrete surface, classify it as **Crack** or **No Crack**
and visualise *where* the model is looking using Grad-CAM.

- **Model:** ResNet18, ImageNet-pretrained, fine-tuned on the Mendeley concrete
  crack dataset (40,000 images).
- **Frontend:** Streamlit single-page app with file upload, sample picker,
  confidence threshold slider and toggleable Grad-CAM overlay.
- **Why it matters:** Automatic crack inspection from photos or drone footage
  is one of the highest-ROI CV applications in construction - it reduces
  manual inspection cost, improves safety, and produces auditable records.

![Demo placeholder - add assets/demo_screenshot.png](assets/demo_screenshot.png)

---

## Quick start

```bash
git clone https://github.com/<your-username>/crack-detection-demo.git
cd crack-detection-demo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501).

> The first run without trained weights will fall back to an un-fine-tuned
> ResNet18 so you can validate the UI. Train the model (below) for real
> predictions.

## Dataset

**Concrete Crack Images for Classification** by Çağlar Fırat Özgenel (Mendeley
Data, id `5y9wdsg2zt`, version 2).

- 40,000 RGB images at 227x227
- 20,000 **Positive** (crack) / 20,000 **Negative** (no crack)
- License: CC BY 4.0

Download the archive, unzip it and place the two class folders under `data/`
so the layout is:

```
data/
  Positive/
    00001.jpg
    ...
  Negative/
    00001.jpg
    ...
```

Direct link: <https://data.mendeley.com/datasets/5y9wdsg2zt/2>

## Train your own weights

```bash
python -m src.train \
  --data-dir data/ \
  --epochs 5 \
  --batch-size 64 \
  --output models/crack_classifier.pt
```

Fine-tuning strategy:

1. Freeze the ResNet18 backbone for the first `--freeze-epochs` (default 2)
   epochs and only train the new 2-way classifier head with `lr=1e-3`.
2. Unfreeze the whole network and continue training at `lr=1e-4` for the
   remaining epochs.

Training time (5 epochs, batch size 64):

| Hardware        | Approx. time |
| --------------- | ------------ |
| CPU (M1/M2)     | 30-45 min    |
| Colab T4 GPU    | 3-5 min      |
| RTX 3060/4090   | 1-3 min      |

Expected validation accuracy on this dataset: **>98%**. The task is visually
easy - images are close-up, centred and well-lit - which is a limitation we
acknowledge below.

## Results

After training, the script prints a classification report and confusion
matrix. A typical run looks like:

```
              precision    recall  f1-score   support

    No Crack     0.993     0.995     0.994      4012
       Crack     0.995     0.993     0.994      3988

    accuracy                         0.994      8000
```

A full walkthrough (EDA, training, evaluation, Grad-CAM) is in
[`notebooks/01_explore_and_train.ipynb`](notebooks/01_explore_and_train.ipynb).

## Project layout

```
.
├── app.py                 # Streamlit frontend
├── src/
│   ├── model.py           # ResNet18 + classifier head
│   ├── dataset.py         # Mendeley folder loader + transforms
│   ├── train.py           # CLI training script
│   ├── predict.py         # Single-image inference helper
│   └── gradcam.py         # Grad-CAM visual explanation
├── notebooks/
│   └── 01_explore_and_train.ipynb
├── sample_images/         # Demo images the app shows in a picker
├── models/                # Trained weights (gitignored)
├── data/                  # Raw dataset (gitignored)
├── requirements.txt
└── README.md
```

## Limitations

Important context before anyone uses this in the real world:

- **Domain gap.** Training images are clean, cropped close-ups on uniform
  concrete. Real site photos have clutter (rebar, shadows, people, tools),
  lower resolution, glare and motion blur. Expect a noticeable accuracy drop.
- **Binary output only.** The model does not localise, measure or classify
  crack severity. A visible crack and a structurally significant one look
  similar to it.
- **No material coverage beyond concrete.** Masonry, asphalt and timber
  cracking are out of scope.
- **Confidence is not calibrated.** The softmax probability should not be
  treated as a calibrated likelihood without post-hoc calibration.
- **Grad-CAM is a visual sanity check**, not a ground-truth crack mask. Use
  it to build trust and catch obvious failure modes, not as a precise
  localisation tool.

## Future work

- **Semantic segmentation** (U-Net / DeepLabV3) to output a pixel-level crack
  mask and derive width / length metrics for severity scoring.
- **Object detection** (YOLOv8) for broader defects: spalling, exposed rebar,
  honeycombing.
- **Video / RTSP** ingestion from existing site CCTV, with per-frame alerts
  and deduplication.
- **Edge deployment** on a Jetson Nano / Orin at site entrances or on
  inspection drones.
- **Integration** with a site management system: log every inspection with
  timestamp, GPS, image, Grad-CAM, and operator acknowledgement.
- **Active learning loop**: push low-confidence or disagreed-with predictions
  to a labelling queue.

## License

MIT. See [LICENSE](LICENSE).

Dataset credit: Özgenel, Ç.F., Gönenç Sorguç, A., "Performance Comparison of
Pretrained Convolutional Neural Networks on Crack Detection in Buildings",
*ISARC 2018*, Berlin.
