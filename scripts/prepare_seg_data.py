"""Download and organise the DeepCrack segmentation dataset.

Source: Zou, Zhang et al., "DeepCrack: A deep hierarchical feature
learning architecture for crack segmentation", Neurocomputing 338
(2019) 139-153. Repository hosting the images and masks:
https://github.com/yhlleo/DeepCrack

Restriction: the original release states "Usage is restricted to
non-commercial research and educational purposes." We redistribute
nothing — this script pulls the upstream ZIP on demand and writes the
derived layout only to the user's local machine.

Derived layout::

    data_seg/
        train/
            images/   <- 80% of upstream train_img
            masks/    <- matching train_lab
        val/
            images/   <- 20% of upstream train_img (seeded)
            masks/
        test/
            images/   <- upstream test_img
            masks/    <- upstream test_lab

``src.seg_dataset.CrackSegDataset`` consumes this layout directly.

Usage::

    python -m scripts.prepare_seg_data              # DeepCrack only
    python -m scripts.prepare_seg_data --keep-raw   # keep the ZIP + extracted tree
    python -m scripts.prepare_seg_data --force      # re-run even if data_seg/ exists
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_SEG_DIR = REPO_ROOT / "data_seg"
RAW_DIR = DATA_SEG_DIR / "_raw"

DEEPCRACK_URL = (
    "https://codeload.github.com/yhlleo/DeepCrack/zip/refs/heads/master"
)
DEEPCRACK_INNER_ZIP = "DeepCrack-master/dataset/DeepCrack.zip"
DEEPCRACK_TRAIN_IMG = "train_img"
DEEPCRACK_TRAIN_LAB = "train_lab"
DEEPCRACK_TEST_IMG = "test_img"
DEEPCRACK_TEST_LAB = "test_lab"

DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _download_zip(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream a ZIP to disk with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0) or 0)
        downloaded = 0
        with dest.open("wb") as fh:
            for piece in resp.iter_content(chunk_size=chunk):
                if not piece:
                    continue
                fh.write(piece)
                downloaded += len(piece)
                if total:
                    pct = downloaded * 100.0 / total
                    sys.stdout.write(
                        f"\r    downloading... {downloaded / 1e6:6.1f} / "
                        f"{total / 1e6:6.1f} MB  ({pct:5.1f}%)"
                    )
                else:
                    sys.stdout.write(
                        f"\r    downloading... {downloaded / 1e6:6.1f} MB"
                    )
                sys.stdout.flush()
        sys.stdout.write("\n")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def _copy_pair(
    src_img: Path,
    src_mask: Path,
    dst_images: Path,
    dst_masks: Path,
) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_images / src_img.name)
    shutil.copy2(src_mask, dst_masks / src_mask.name)


def _find_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Match image to mask by filename stem (ignoring extensions)."""
    for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = mask_dir / (image_path.stem + ext)
        if candidate.is_file():
            return candidate
    return None


def _collect_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img in _list_images(img_dir):
        mask = _find_mask(img, mask_dir)
        if mask is None:
            print(f"  ! no mask for {img.name}, skipping")
            continue
        pairs.append((img, mask))
    return pairs


def _split_train_val(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    pairs_sorted = sorted(pairs, key=lambda p: p[0].name)
    rng = random.Random(seed)
    pairs_shuffled = list(pairs_sorted)
    rng.shuffle(pairs_shuffled)
    n_val = int(len(pairs_shuffled) * val_ratio)
    return pairs_shuffled[n_val:], pairs_shuffled[:n_val]


def _cleanup_target() -> None:
    """Remove previous train/val/test contents (keep ``_raw`` by default)."""
    for subdir in ("train", "val", "test"):
        target = DATA_SEG_DIR / subdir
        if target.exists():
            shutil.rmtree(target)


def prepare_deepcrack(val_ratio: float, seed: int, keep_raw: bool) -> None:
    zip_path = RAW_DIR / "deepcrack.zip"
    extracted = RAW_DIR / "deepcrack_extracted"
    inner_zip_path = extracted / DEEPCRACK_INNER_ZIP
    dataset_root = extracted / "dataset"

    if not zip_path.is_file():
        print("- Downloading DeepCrack (yhlleo/DeepCrack master)...")
        _download_zip(DEEPCRACK_URL, zip_path)
    else:
        print(f"- Reusing cached ZIP at {zip_path}")

    if not inner_zip_path.is_file():
        print("- Extracting outer archive...")
        _extract_zip(zip_path, extracted)

    if not inner_zip_path.is_file():
        raise RuntimeError(
            f"Inner ZIP not found after extraction: {inner_zip_path}"
        )

    if not (dataset_root / DEEPCRACK_TRAIN_IMG).is_dir():
        print("- Extracting inner DeepCrack.zip...")
        _extract_zip(inner_zip_path, dataset_root)
    else:
        print(f"- Reusing extracted data at {dataset_root}")

    if not (dataset_root / DEEPCRACK_TRAIN_IMG).is_dir():
        raise RuntimeError(
            f"Expected {dataset_root / DEEPCRACK_TRAIN_IMG} not found."
        )

    train_pairs = _collect_pairs(
        dataset_root / DEEPCRACK_TRAIN_IMG,
        dataset_root / DEEPCRACK_TRAIN_LAB,
    )
    test_pairs = _collect_pairs(
        dataset_root / DEEPCRACK_TEST_IMG,
        dataset_root / DEEPCRACK_TEST_LAB,
    )
    print(
        f"  found {len(train_pairs)} train pairs, {len(test_pairs)} test pairs"
    )
    if not train_pairs:
        raise RuntimeError("No training pairs discovered — check upstream.")

    train_set, val_set = _split_train_val(train_pairs, val_ratio, seed)
    print(
        f"  split: train={len(train_set)} val={len(val_set)} test={len(test_pairs)}"
    )

    for split_name, pairs in (
        ("train", train_set),
        ("val", val_set),
        ("test", test_pairs),
    ):
        dst_img = DATA_SEG_DIR / split_name / "images"
        dst_mask = DATA_SEG_DIR / split_name / "masks"
        for img, mask in pairs:
            _copy_pair(img, mask, dst_img, dst_mask)
        print(f"  wrote {len(pairs):>4} pairs -> data_seg/{split_name}/")

    if not keep_raw:
        print("- Cleaning up raw download (pass --keep-raw to retain)...")
        shutil.rmtree(RAW_DIR, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the DeepCrack segmentation dataset."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of upstream training set to hold out for validation "
        "(default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for the train/val split (default: 42).",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded ZIP + extracted tree under data_seg/_raw.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if data_seg/{train,val,test} already exist.",
    )
    args = parser.parse_args()

    DATA_SEG_DIR.mkdir(parents=True, exist_ok=True)

    already_ready = all(
        (DATA_SEG_DIR / d / "images").is_dir()
        and any((DATA_SEG_DIR / d / "images").iterdir())
        for d in ("train", "val", "test")
    )
    if already_ready and not args.force:
        print(
            "data_seg/{train,val,test}/ already look populated. "
            "Pass --force to redo."
        )
        return

    _cleanup_target()
    prepare_deepcrack(
        val_ratio=args.val_ratio,
        seed=args.seed,
        keep_raw=args.keep_raw,
    )
    print("Done. Ready for `python -m src.seg_train`.")


if __name__ == "__main__":
    main()
