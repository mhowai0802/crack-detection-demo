# Sample Images

Bundled demo images shown in the Streamlit "Pick a sample" dropdown.
Naming convention:

- `crack_01.jpg`, `crack_02.jpg`, … — positive (cracked) examples
- `no_crack_01.jpg`, `no_crack_02.jpg`, … — negative examples

Supported formats: `.jpg`, `.jpeg`, `.png`.

## Provenance of bundled samples

All eight currently bundled images are cropped from the [**Özgenel
Concrete Crack Images for Classification (CCIC)**](https://data.mendeley.com/datasets/5y9wdsg2zt)
dataset (Özgenel, Ç.F., 2019, Mendeley Data, V2). Source material was
captured at METU (Middle East Technical University, Ankara) facades.

- **License:** CC-BY-4.0
- **Attribution:** Özgenel, Çağlar Fırat (2019). *Concrete Crack Images
  for Classification* [Data set]. Mendeley Data, v2.
  [`doi:10.17632/5y9wdsg2zt.2`](https://doi.org/10.17632/5y9wdsg2zt.2)

The segmentation model (`models/crack_segmenter.pt`) was not trained on
these images — it was trained on DeepCrack (Zou 2018). See
`data/DATASETS.md` for the full data provenance table.

## Why no HK-specific photos are bundled

The demo's stated goal is to stay close to Hong Kong MBIS inspection
practice, but we deliberately do **not** redistribute any HK building
defect photo in this repository:

- HK Buildings Department's "Common Building Defects" gallery is under
  Crown copyright and explicit terms of use that disallow
  redistribution. We reference it by URL only (see below).
- Photos taken from private management offices, engineering firms, or
  Mandatory Building Inspection Scheme (MBIS) reports are proprietary.
- Wikimedia Commons has only a handful of HK building facade photos
  that happen to show visible cracking, and the subject / angle /
  surface finish rarely match the typical tile-clad pre-1980s HK
  building stock that MBIS mostly inspects.

For an honest discussion of why this matters for demo realism, see the
**Limitations** section in the top-level `README.md`.

## Adding HK-flavoured samples yourself

Two supported ways, roughly in order of legal cleanliness:

### Option 1 — your own photos (recommended)

If you have photographs you took on-site (with consent), simply drop
them into this folder and the Streamlit app will pick them up. Zero
licensing friction because you own the copyright. A good test set
includes:

- Close-up of a hairline crack on a concrete beam or slab soffit
  (target ≤ 0.3 mm per SUC 2013)
- A wider crack ≥ 1 mm suspected to be structural
- Concrete spalling with exposed / rusting rebar (PNAP APP-137
  scenarios)
- A clean "control" concrete surface with no cracking

### Option 2 — `scripts/fetch_hk_samples.py` (LOCAL DEMO ONLY)

The HK Buildings Department publishes a small set of defect reference
thumbnails on its public "Building Defects" page. These are
**Crown Copyright of the HKSAR Government** and we do **not**
redistribute them in this repo. A helper script is provided that:

- Prints the BD copyright notice and asks you to confirm with
  `--accept-license`.
- Downloads 5 BD thumbnails into `sample_images/hk_bd_*.jpg`
  (non-structural crack, spalling, structural crack, wall finish,
  overview).
- Relies on the `.gitignore` rule `sample_images/hk_bd_*.jpg` so the
  downloaded copies never reach git.

Usage (from the repo root):

```bash
python -m scripts.fetch_hk_samples --list                 # dry-run preview
python -m scripts.fetch_hk_samples --accept-license        # actually download
python -m scripts.fetch_hk_samples --accept-license --force  # overwrite
```

**You MUST NOT commit or push the downloaded `hk_bd_*.jpg` files** —
they exist on your laptop for private demo / interview use only. If
that constraint is a problem for your use case, delete them after
the demo.

## External references (URL only — not redistributed)

- **HK Buildings Department — Common Building Defects**: photographs
  of typical defects found during MBIS inspections.
  <https://www.bd.gov.hk/en/safety-inspection/mbis/mandatory-building-inspection-scheme/common-building-defects/index.html>
- **HK Buildings Department — Code of Practice for Structural Use of
  Concrete 2013 (SUC 2013)**: reference document for crack-width
  limits quoted in the AI report.
- **Wikimedia Commons** search links (license-clean if you pick an
  image with a compatible tag):
  - [`Hong Kong concrete`](https://commons.wikimedia.org/w/index.php?search=Hong+Kong+concrete&title=Special:MediaSearch&go=Go)
  - [`pavement crack`](https://commons.wikimedia.org/w/index.php?search=pavement+crack&title=Special:MediaSearch&go=Go)
  - [`concrete spalling`](https://commons.wikimedia.org/w/index.php?search=concrete+spalling&title=Special:MediaSearch&go=Go)

If you download anything from Commons, keep the attribution in this
README alongside the file name.
