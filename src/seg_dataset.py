"""Dataset and transforms for binary crack segmentation.

Expected layout after ``scripts/prepare_seg_data.py`` completes::

    data_seg/
        train/
            images/   *.jpg|png
            masks/    *.png   (binary; >0 means crack, 0 means background)
        val/
            images/
            masks/
        test/
            images/
            masks/

Image / mask pairs are matched by filename stem, so ``images/foo.jpg``
binds to ``masks/foo.png`` regardless of extension.

Transforms have to be applied **jointly** to image and mask so the
spatial augmentations (flip / rotate / resize) stay aligned. We do this
with ``torchvision.transforms.v2`` which understands both tensors and
mask tensors in a single call — see ``get_train_transform``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from torchvision.transforms import v2 as T
    from torchvision import tv_tensors
except ImportError as exc:
    raise ImportError(
        "torchvision >= 0.17 is required for transforms.v2 (joint "
        "image/mask augmentation). Run `pip install -U torchvision`."
    ) from exc

from src.dataset import IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

DEFAULT_SEG_IMAGE_SIZE = 384


class CrackSegDataset(Dataset):
    """Paired (image, binary mask) dataset.

    Parameters
    ----------
    root:
        Directory containing ``images/`` and ``masks/`` sub-folders.
    transform:
        Optional joint transform that accepts a ``(tv_tensors.Image,
        tv_tensors.Mask)`` pair and returns a transformed pair.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {images_dir} and {masks_dir} to exist. "
                "Run `python -m scripts.prepare_seg_data` first."
            )

        mask_index = {
            p.stem: p
            for p in masks_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS
        }

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(images_dir.rglob("*")):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_index.get(img_path.stem)
            if mask_path is not None:
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(
                f"No image/mask pairs found under {self.root}."
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tv = tv_tensors.Image(image)
        mask_arr = (np.array(mask) > 0).astype(np.uint8)
        mask_tv = tv_tensors.Mask(torch.from_numpy(mask_arr))

        if self.transform is not None:
            image_tv, mask_tv = self.transform(image_tv, mask_tv)

        mask_tensor = mask_tv.to(dtype=torch.float32).unsqueeze(0)
        image_tensor = image_tv.to(dtype=torch.float32)

        return image_tensor, mask_tensor


def get_train_transform(image_size: int = DEFAULT_SEG_IMAGE_SIZE) -> Callable:
    """Joint train transform with flips, rotations and color jitter.

    ImageNet normalisation is applied to the image only; the mask stays
    in ``{0, 1}`` for the loss function.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = DEFAULT_SEG_IMAGE_SIZE) -> Callable:
    """Deterministic transform used for validation and inference."""
    return T.Compose(
        [
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
