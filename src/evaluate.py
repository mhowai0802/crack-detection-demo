"""Held-out evaluation for the trained crack classifier.

The training script already prints metrics at the end of the run, but
once the checkpoint is saved we want to re-evaluate without retraining
(e.g. to show a dashboard inside Streamlit or to sweep the decision
threshold). This module isolates that logic so it is importable from
both a script and the FastAPI backend.

Key ideas
---------
* Use the exact same seeded three-way split as ``src.train`` (via
  ``src.splits``) so numbers are directly comparable.
* Pick which held-out slice to score via the ``split`` argument:
  ``"test"`` (default, unbiased number) or ``"val"`` (the slice used for
  best-checkpoint selection — handy when a test set is unavailable).
* Collect the raw ``[N, 2]`` softmax probability matrix once; anything
  threshold-dependent (confusion matrix, precision / recall / F1) is
  derived cheaply without re-running the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.constants import CLASS_NAMES, CRACK_INDEX
from src.dataset import CrackDataset, get_eval_transform
from src.metrics import metrics_at_threshold  # re-exported for callers
from src.model import load_model
from src.splits import (
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_VAL_RATIO,
    three_way_split_indices,
)

SplitName = Literal["val", "test"]

__all__ = [
    "CLASS_NAMES",
    "CRACK_INDEX",
    "DEFAULT_SEED",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_VAL_RATIO",
    "EvaluationResult",
    "SplitName",
    "evaluate_checkpoint",
    "metrics_at_threshold",
]


@dataclass(frozen=True)
class EvaluationResult:
    """Raw evaluation artefacts independent of the decision threshold."""

    probs: np.ndarray          # [N, 2] softmax probabilities
    targets: np.ndarray        # [N] integer labels in {0, 1}
    sample_paths: List[Path]   # length-N absolute paths back to source images
    split: SplitName           # which slice (`"val"` or `"test"`) was scored
    num_train: int
    num_val: int
    num_test: int

    @property
    def num_samples(self) -> int:
        return int(self.targets.shape[0])


def _build_subset(
    data_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split: SplitName,
) -> Tuple[Subset, int, int, int]:
    full_eval = CrackDataset(data_dir, transform=get_eval_transform())
    n_total = len(full_eval)

    train_idx, val_idx, test_idx = three_way_split_indices(
        n_total, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    if split == "val":
        indices = val_idx
    elif split == "test":
        if n_test == 0:
            raise ValueError(
                "test split is empty — set test_ratio > 0 to hold out a "
                "test set, or pass split='val' instead."
            )
        indices = test_idx
    else:  # pragma: no cover - Literal guard
        raise ValueError(f"Unknown split: {split!r}")

    return Subset(full_eval, indices), n_train, n_val, n_test


def evaluate_checkpoint(
    model_path: Union[str, Path],
    data_dir: Union[str, Path],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
    split: SplitName = "test",
    batch_size: int = 64,
    num_workers: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> EvaluationResult:
    """Score ``model_path`` on the seeded held-out split.

    Args:
        model_path: Path to a ``.pt`` checkpoint saved by ``src.train``.
        data_dir: Root of the classification dataset (with ``Positive`` /
            ``Negative`` subfolders).
        val_ratio: Fraction of the dataset used as validation during
            training. Must match training for indices to line up.
        test_ratio: Fraction used as a fully held-out test set. Must
            match training for indices to line up.
        seed: RNG seed. Must match training.
        split: Which slice to score — ``"test"`` for the unbiased
            generalisation number, ``"val"`` to replay the checkpoint
            selection slice.
        batch_size: Eval batch size.
        num_workers: DataLoader workers; ``0`` is safer inside Streamlit.
        device: Device to run inference on.

    Returns:
        An :class:`EvaluationResult` holding raw probabilities, integer
        targets, and the source paths for every sample in the requested
        slice, plus all three split sizes for downstream display.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    subset, n_train, n_val, n_test = _build_subset(
        data_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split=split,
    )
    base_dataset: CrackDataset = subset.dataset  # type: ignore[assignment]
    sample_paths = [base_dataset.samples[i][0] for i in subset.indices]

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
    )

    model = load_model(model_path, device=device)
    model.eval()

    probs_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_chunks.append(probs)
            target_chunks.append(labels.numpy().astype(np.int64))

    probs = (
        np.concatenate(probs_chunks, axis=0)
        if probs_chunks
        else np.zeros((0, len(CLASS_NAMES)), dtype=np.float32)
    )
    targets = (
        np.concatenate(target_chunks, axis=0)
        if target_chunks
        else np.zeros((0,), dtype=np.int64)
    )

    return EvaluationResult(
        probs=probs,
        targets=targets,
        sample_paths=sample_paths,
        split=split,
        num_train=n_train,
        num_val=n_val,
        num_test=n_test,
    )
