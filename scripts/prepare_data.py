"""Build a classification dataset from the Vizuara HuggingFace parquet.

The source parquet contains 800 crack images paired with 800 binary
segmentation masks. We split each 256x256 image into a 2x2 grid of
128x128 patches and label each patch according to its matching mask
region:

    mask_white_fraction(patch) >= THRESHOLD  -> Positive (Crack)
    mask_white_fraction(patch) <  THRESHOLD  -> Negative (No Crack)

This mirrors the construction of the original Mendeley dataset (high
resolution photos -> 227x227 patches, labeled by presence of a crack)
while needing only a 7 MB parquet download instead of 2.3 GB.

Output layout::

    data/
      Positive/   crack_<id>_q<quadrant>.jpg
      Negative/   nocrack_<id>_q<quadrant>.jpg
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PARQUET_PATH = Path("/tmp/crack.parquet")
OUT_ROOT = Path(__file__).resolve().parents[1] / "data"
POSITIVE_DIR = OUT_ROOT / "Positive"
NEGATIVE_DIR = OUT_ROOT / "Negative"

PATCH_GRID = 2
CRACK_PIXEL_VALUE = 0
CRACK_MIN_PIXELS = 5


def _load_pairs(path: Path) -> list[Tuple[str, bytes, bytes]]:
    """Return a list of (stem, image_bytes, mask_bytes) triples."""
    table = pq.read_table(path)
    image_col = table.column("image").combine_chunks()
    labels = table.column("label").to_pylist()
    paths = image_col.field("path").to_pylist()
    blobs = image_col.field("bytes").to_pylist()

    by_stem: dict[str, dict[int, bytes]] = defaultdict(dict)
    for p, label, data in zip(paths, labels, blobs):
        stem = Path(p).stem
        by_stem[stem][label] = data

    pairs = []
    for stem, by_label in by_stem.items():
        if 0 in by_label and 1 in by_label:
            pairs.append((stem, by_label[0], by_label[1]))
    return pairs


def _quadrant_bounds(size: int, grid: int) -> list[Tuple[int, int, int, int]]:
    step = size // grid
    return [
        (c * step, r * step, (c + 1) * step, (r + 1) * step)
        for r in range(grid)
        for c in range(grid)
    ]


def main() -> None:
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(PARQUET_PATH)
    print(f"Loaded {len(pairs)} image/mask pairs from {PARQUET_PATH}")

    pos_count = 0
    neg_count = 0

    for stem, img_bytes, mask_bytes in pairs:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

        mask_np = np.array(mask)

        for idx, (x0, y0, x1, y1) in enumerate(
            _quadrant_bounds(image.size[0], PATCH_GRID)
        ):
            image_patch = image.crop((x0, y0, x1, y1))
            mask_patch = mask_np[y0:y1, x0:x1]
            crack_pixels = int((mask_patch > CRACK_PIXEL_VALUE).sum())

            if crack_pixels >= CRACK_MIN_PIXELS:
                out = POSITIVE_DIR / f"crack_{stem}_q{idx}.jpg"
                pos_count += 1
            else:
                out = NEGATIVE_DIR / f"nocrack_{stem}_q{idx}.jpg"
                neg_count += 1
            image_patch.save(out, quality=92)

    print(f"Wrote {pos_count} Positive + {neg_count} Negative patches")
    print(f"Ratio positive: {pos_count / max(pos_count + neg_count, 1):.2%}")


if __name__ == "__main__":
    main()
