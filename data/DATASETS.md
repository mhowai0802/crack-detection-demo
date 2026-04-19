# Datasets

This project uses two public datasets — one for **classification**
(the ResNet18 model in `models/crack_classifier.pt`) and one for
**segmentation** (the small U-Net in `models/crack_segmenter.pt`). We
redistribute neither; both are pulled on demand when you run the prep
scripts.

## 1. Classification — Özgenel *Concrete Crack Images for Classification (CCIC)*

- **What:** 40,000 RGB images, 227×227, split 20,000 **Positive** /
  20,000 **Negative**. Captured on the facades of METU (Middle East
  Technical University, Ankara).
- **Version used here:** v2 on Mendeley Data.
- **License:** Creative Commons Attribution 4.0 International
  (CC-BY-4.0).
- **Source:** <https://data.mendeley.com/datasets/5y9wdsg2zt/2>
  (`doi:10.17632/5y9wdsg2zt.2`)
- **Bundled helper:** `scripts/prepare_data.py` (expects the ZIP at
  `data/`).
- **Citation:**

  > Özgenel, Ç.F. (2019). *Concrete Crack Images for Classification*
  > [Data set]. Mendeley Data, v2.
  > https://doi.org/10.17632/5y9wdsg2zt.2
  >
  > Özgenel, Ç.F., Gönenç Sorguç, A. (2018). "Performance Comparison
  > of Pretrained Convolutional Neural Networks on Crack Detection in
  > Buildings." *ISARC 2018*, Berlin.

- **Local layout expected by `src.dataset.CrackDataset`:**

  ```
  data/
    Positive/   # crack images
    Negative/   # no-crack images
  ```

## 2. Segmentation — DeepCrack (Zou 2018)

- **What:** 537 RGB images with pixel-level binary crack annotations
  (+ 237 held-out test images). Multi-scale cracks across pavement
  and general concrete scenes.
- **License:** "Usage is restricted to non-commercial research and
  educational purposes" (see upstream README).
- **Source:** <https://github.com/yhlleo/DeepCrack> (inner
  `dataset/DeepCrack.zip`)
- **Bundled helper:** `scripts/prepare_seg_data.py` downloads the
  upstream archive, unpacks the inner ZIP, and writes
  `data_seg/{train,val,test}/{images,masks}/` with an 80/20 val split
  carved from the upstream train set (seed 42).
- **Citation:**

  > Zou, Q., Zhang, Z., Li, Q., Qi, X., Wang, Q., Wang, S. (2019).
  > "DeepCrack: A deep hierarchical feature learning architecture for
  > crack segmentation." *Neurocomputing*, 338, 139-153.
  > https://doi.org/10.1016/j.neucom.2019.01.036

- **Local layout produced by the prep script:**

  ```
  data_seg/
    train/images/  + train/masks/   (240 pairs)
    val/images/    + val/masks/      (60 pairs)
    test/images/   + test/masks/    (237 pairs)
  ```

## 3. Out-of-scope datasets (optional fallbacks)

None shipped by default. The following are license-compatible and can
be added by copying their images + masks into `data_seg/train/…` if
you want to broaden the segmenter's training distribution:

- **CRACK500** (Yang et al., 2019) — pavement-focused, ~500 images.
  Source: <https://github.com/fyangneil/pavement-crack-detection>.
- **CrackForest (CFD)** — 118 road pavement images.

## Provenance discipline

- No image from any dataset above is committed to git. All dataset
  files are covered by `.gitignore`.
- Bundled `sample_images/*.jpg` are derived from the Mendeley CCIC
  release under its CC-BY-4.0 license; attribution is in
  `sample_images/README.md`.
- **Hong Kong** imagery is intentionally not bundled. The HK
  Buildings Department "Common Building Defects" photo gallery is
  referenced by URL only; see `sample_images/README.md` for the full
  rationale.
