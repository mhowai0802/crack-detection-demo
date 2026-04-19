# Concrete Crack Detection

A lightweight computer vision demo built for a Hong Kong construction
industry use case: given a photo of a concrete surface, **classify** it as
*Crack* or *No Crack*, **segment** the crack pixels, and generate a
compliance-style inspection report referencing SUC 2013 / PNAP APP-137 / the
Buildings Ordinance (Cap. 123).

- **Classifier:** ResNet18, ImageNet-pretrained, fine-tuned on the Özgenel
  CCIC concrete crack dataset (40,000 images).
- **Segmenter:** custom small U-Net (~1.94 M params) trained on DeepCrack
  (Zou 2018). Produces a pixel-level binary mask plus coverage / length /
  max-width statistics.
- **Backend:** FastAPI (`backend/main.py`) — owns both PyTorch models,
  Grad-CAM, segmentation, evaluation, and HKBU GenAI integration. Exposes
  JSON / image endpoints under `/api/*`.
- **Frontend:** Streamlit (`frontend/🏠_Home.py`) — a thin HTTP client. Sample
  picker, Grad-CAM overlay, **pixel-level mask overlay with metrics**, AI
  compliance report, plus an Evaluation dashboard.
- **Why it matters:** Automatic crack inspection from photos or drone footage
  is one of the highest-ROI CV applications in construction — it reduces
  manual inspection cost, improves safety, and produces auditable records.
  Segmentation turns "is there a crack?" into "**where** is it and **how
  bad**?", which is the form HK MBIS inspectors actually report.

![Demo placeholder - add assets/demo_screenshot.png](assets/demo_screenshot.png)

---

## Quick start

```bash
git clone https://github.com/<your-username>/crack-detection-demo.git
cd crack-detection-demo

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install both sides (or use the per-side files separately).
pip install -r requirements.txt

cp .env.example .env             # fill in HKBU_API_KEY if you want AI features
```

Run the two processes in separate terminals:

```bash
# terminal 1 — backend on :8000
uvicorn backend.main:app --reload --port 8000

# terminal 2 — frontend on :8501
streamlit run frontend/🏠_Home.py
```

Open the URL Streamlit prints (default http://localhost:8501). The FastAPI
interactive docs are at http://localhost:8000/docs.

> The first run without trained weights falls back to an un-fine-tuned
> ResNet18 so you can validate the UI. Train the model (below) for real
> predictions.

### Running only the backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --port 8000
# or: python -m backend
```

### Running only the frontend

```bash
pip install -r frontend/requirements.txt
BACKEND_URL=http://my-backend-host:8000 streamlit run frontend/🏠_Home.py
```

## API surface

| Method | Path                       | Purpose                                              |
|--------|----------------------------|------------------------------------------------------|
| GET    | `/api/health`              | Status + whether trained weights are loaded         |
| GET    | `/api/samples`             | List bundled demo images                             |
| GET    | `/api/samples/{name}`      | Serve a demo image                                   |
| POST   | `/api/predict`             | Upload `file` **or** `?sample=<name>` → prediction   |
| POST   | `/api/gradcam`             | Same inputs → PNG Grad-CAM overlay (`?alpha=`)       |
| POST   | `/api/segment`             | Same inputs → overlay + binary mask + mask stats     |
| POST   | `/api/ai/report`           | Generate an inspection note (HKBU GenAI)             |
| POST   | `/api/ai/chat`             | Multi-turn chat grounded in the current prediction   |
| POST   | `/api/evaluate`            | Re-score checkpoint on seeded val or test slice      |
| GET    | `/api/dataset-image`       | Serve a dataset image by absolute path (sandboxed)   |

`/api/segment` returns JSON of the form `{overlay_png_b64, mask_png_b64,
stats: {crack_pixel_ratio, num_components, area_px, length_px,
max_width_px, image_height_px, image_width_px}}`. The overlay is the
original image with a red tint over predicted crack pixels; the raw
mask is provided separately for downstream processing. When the
segmentation checkpoint is missing, the endpoint responds `503` and the
Home page quietly hides the segmentation section.

## Datasets

The full inventory (including licenses, citations and download
instructions) lives in [`data/DATASETS.md`](data/DATASETS.md). Summary:

| Purpose          | Dataset                                    | Size          | License     |
| ---------------- | ------------------------------------------ | ------------- | ----------- |
| Classification   | Özgenel *Concrete Crack Images for Class.* | 40,000 × 227² | CC-BY-4.0   |
| Segmentation     | DeepCrack (Zou 2018)                       | 537 images    | Research-only |

### Classification

**Concrete Crack Images for Classification** by Çağlar Fırat Özgenel
(Mendeley Data, id `5y9wdsg2zt`, version 2). 40,000 RGB images at
227×227 (20,000 Positive / 20,000 Negative). License: CC-BY-4.0.
Direct link: <https://data.mendeley.com/datasets/5y9wdsg2zt/2>. Place
under `data/Positive/` and `data/Negative/`.

### Segmentation

**DeepCrack** (Zou et al., *Neurocomputing* 2019). 537 RGB images with
pixel-level crack masks. Use the bundled prep script to download and
carve a train/val/test split:

```bash
python -m scripts.prepare_seg_data
# → data_seg/{train,val,test}/{images,masks}/
```

The script pulls the upstream archive from `yhlleo/DeepCrack` on
GitHub on demand. We redistribute nothing — the ZIP lands only on your
local machine.

## Train your own weights

### Classifier

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

The data is partitioned into a seeded **train / val / test** split (default
70 / 15 / 15, `seed=42`) via `src/splits.py` — a single source of truth that
both `src/train.py` and `src/evaluate.py` import, so the three slices line up
exactly across training and evaluation. The best checkpoint is selected on
**val**; the held-out **test** slice is scored once at the end of training
(and by the Evaluation page by default) to give an unbiased generalisation
number.

Expected test accuracy on this dataset: **>97%**. The task is visually
easy — images are close-up, centred and well-lit — which is a limitation we
acknowledge below.

### Segmenter

```bash
python -m scripts.prepare_seg_data          # one-off dataset download
python -m src.seg_train \
  --epochs 30 \
  --image-size 256 \
  --batch-size 4 \
  --device auto \
  --output models/crack_segmenter.pt
```

Loss is BCE-with-logits + Dice, both weighted 0.5. Best val-IoU
checkpoint is saved with early stopping after `--patience` epochs
(default 5) without improvement. The default ~1.94 M-parameter U-Net
(`base_channels=16`, input 256×256) fits CPU memory.

Training time (30 epochs, DeepCrack 240-image train split):

| Hardware          | Approx. time | Typical test IoU |
| ----------------- | ------------ | ---------------- |
| CPU               | 2-3 hr       | ~0.55-0.65       |
| Apple M-series MPS | 4-6 min     | ~0.60-0.67       |
| Colab T4 GPU      | 2-3 min      | ~0.65-0.72       |

Reference run on Apple M2 @ image_size 256: **val IoU 0.626**, **test
IoU 0.664**. The segmenter is deliberately small so it stays
explainable in an interview setting and does not introduce an
additional heavyweight dependency — swapping in `torchvision.models.`
`segmentation.deeplabv3_mobilenet_v3_large` is a one-line change if you
want a stronger baseline.

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
├── backend/                # FastAPI service
│   ├── main.py             # app + CORS + routers
│   ├── config.py           # env vars + paths + device resolution
│   ├── deps.py             # cached model loader + image helpers
│   ├── schemas.py          # Pydantic request/response models
│   ├── routers/            # health, samples, predict, ai, evaluate
│   └── requirements.txt
├── frontend/                 # Streamlit HTTP client
│   ├── 🏠_Home.py            # Home page (sample picker + predict + AI)
│   ├── api_client.py         # requests wrapper around /api/*
│   ├── pages/
│   │   ├── 1_🏗️_Architecture.py
│   │   └── 2_📊_Evaluation.py
│   └── requirements.txt
├── src/                    # Shared ML code (backend imports)
│   ├── constants.py        # CLASS_NAMES (torch-free, shared)
│   ├── model.py            # ResNet18 + classifier head + load/save
│   ├── dataset.py          # Mendeley folder loader + transforms
│   ├── train.py            # CLI training script (classifier)
│   ├── predict.py          # Single-image inference helper
│   ├── gradcam.py          # Grad-CAM overlay / raw map
│   ├── seg_model.py        # Small U-Net architecture + load helper
│   ├── seg_dataset.py      # Paired image/mask dataset + joint transforms
│   ├── seg_train.py        # CLI training script (segmenter, BCE+Dice)
│   ├── seg_infer.py        # predict_mask + mask_stats (area, length, max_width)
│   ├── seg_viz.py          # Mask overlay + contour helpers
│   ├── evaluate.py         # evaluate_checkpoint
│   ├── metrics.py          # metrics_at_threshold + ROC (pure numpy)
│   ├── llm.py              # HKBU GenAI REST wrapper
│   └── ai_prompts.py       # ZH/EN system prompts + context builder
├── scripts/
│   ├── prepare_data.py         # Build the classification dataset
│   ├── prepare_seg_data.py     # Pull DeepCrack into data_seg/
│   └── smoke_test.py           # Import / pipeline sanity check
├── notebooks/
│   └── 01_explore_and_train.ipynb
├── sample_images/          # Demo images for the Home page picker
├── models/                 # Trained weights (gitignored)
├── data/                   # Raw classification dataset (gitignored)
│   └── DATASETS.md         # Dataset provenance + license table
├── data_seg/               # Segmentation dataset (gitignored)
├── requirements.txt        # dev-all (includes both sides)
└── README.md
```

## Limitations

Important context before anyone uses this in the real world — especially for
HK MBIS-style deployments:

- **Domain gap.** The classifier is trained on clean 227×227 close-ups from
  METU (Turkey); the segmenter is trained on DeepCrack (mostly pavement /
  general concrete). Neither dataset is Hong Kong. Real HK MBIS targets are
  tile-clad pre-1980s façades with cladding, water staining and
  efflorescence. Expect accuracy to drop on wide-angle HK site photos.
- **Defect type coverage.** The models only handle "crack vs background" on
  concrete. They **do not** cover concrete spalling, rebar corrosion,
  efflorescence, honeycombing, mosaic / tile drummy areas, or algae
  growth — each of which typically out-numbers visible cracks in actual HK
  MBIS reports.
- **Pixels, not millimetres.** The segmenter reports `max_width_px` and
  `length_px`. Converting to millimetres requires an on-site reference scale
  (ruler / crack gauge in frame) — SUC 2013's 0.2 / 0.3 mm compliance limits
  cannot be judged from a single uncalibrated photo.
- **Single-frame prediction.** No video ingestion, no temporal smoothing,
  no multi-angle aggregation.
- **Confidence is not calibrated.** Softmax probabilities and mask
  thresholds are both uncalibrated — use them as ranking signals, not
  calibrated likelihoods.
- **Grad-CAM ≠ crack mask.** Grad-CAM answers "where did the classifier
  look?". The segmenter answers "where are the crack pixels?". They are
  deliberately rendered side-by-side so the interviewer / user can see the
  difference.
- **AI report is advisory.** Every generated report ends with a disclaimer
  pointing back to RSE / AP under SUC 2013 and the Buildings Ordinance; the
  model's job is to save an inspector 10 minutes of typing, not to replace
  engineering judgement.

## Future work

- **Fine-tune on HK data.** Even a few hundred hand-labelled HK MBIS photos
  (crack + spalling + staining masks) would close a lot of the domain gap.
- **Multi-class segmentation.** Add spalling / rust / efflorescence classes
  so the report can talk about more than just cracks.
- **Automatic pixel→mm calibration** via reference-object detection (tape
  measure, crack gauge, A4 paper) or stereo.
- **Stronger backbone.** Swap the small U-Net for DeepLabV3 +
  MobileNet-V3-Large (one-line change in `src/seg_model.py`) for accuracy
  over latency.
- **Video / RTSP** ingestion from existing site CCTV.
- **Edge deployment** on a Jetson Orin at site entrances or inspection
  drones.
- **Active learning loop**: push low-confidence / disagreed-with predictions
  to a labelling queue for inspectors to correct.

## License

Code: MIT. See [LICENSE](LICENSE).

Datasets are **not** relicensed by this project:

- Classification: Özgenel, Ç.F., Gönenç Sorguç, A., "Performance Comparison
  of Pretrained Convolutional Neural Networks on Crack Detection in
  Buildings", *ISARC 2018*, Berlin. Dataset: CC-BY-4.0.
- Segmentation: Zou, Q., Zhang, Z., Li, Q., Qi, X., Wang, Q., Wang, S.,
  "DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack
  Segmentation", *Neurocomputing* 338 (2019) 139-153. Dataset:
  non-commercial research/education only.

See [`data/DATASETS.md`](data/DATASETS.md) for full attribution.
