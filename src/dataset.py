"""Dataset and transforms for the Mendeley concrete crack classification set.

Expected folder layout (as shipped on Mendeley data repository id ``5y9wdsg2zt``)::

    data/
        Positive/   -> images containing cracks
        Negative/   -> images with no cracks

The two folder names are case-insensitive. Labels follow the order defined in
``src.model.CLASS_NAMES`` (``0 = No Crack``, ``1 = Crack``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

POSITIVE_DIR_NAMES = {"positive", "crack", "cracked"}
NEGATIVE_DIR_NAMES = {"negative", "no_crack", "uncracked", "nocrack"}


def _resolve_class_dir(root: Path, candidates: set[str]) -> Path:
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in candidates:
            return child
    raise FileNotFoundError(
        f"Could not find a subdirectory of {root} matching {sorted(candidates)}."
    )


class CrackDataset(Dataset):
    """Flat crack/no-crack image folder dataset."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.root}")

        positive_dir = _resolve_class_dir(self.root, POSITIVE_DIR_NAMES)
        negative_dir = _resolve_class_dir(self.root, NEGATIVE_DIR_NAMES)

        self.samples: List[Tuple[Path, int]] = []
        for path in self._list_images(negative_dir):
            self.samples.append((path, 0))
        for path in self._list_images(positive_dir):
            self.samples.append((path, 1))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}.")

        self.transform = transform

    @staticmethod
    def _list_images(folder: Path) -> List[Path]:
        return sorted(
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_train_transform(image_size: int = 224) -> Callable:
    """Augmentations for training: flips, rotations, color jitter."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = 224) -> Callable:
    """Deterministic transform used for validation and inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
