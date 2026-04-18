"""Held-out evaluation for the trained crack classifier.

The training script already prints metrics at the end of the run, but
once the checkpoint is saved we want to re-evaluate without retraining
(e.g. to show a dashboard inside Streamlit or to sweep the decision
threshold). This module isolates that logic so it is importable from
both a script and the Streamlit UI.

Key ideas
---------
* Use the exact same seeded 80 / 20 split as ``src.train`` so metrics are
  comparable with what was printed at train time.
* Collect the raw ``[N, 2]`` softmax probability matrix once; anything
  threshold-dependent (confusion matrix, precision / recall / F1) is
  then derived cheaply without re-running the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import CrackDataset, get_eval_transform
from src.model import CLASS_NAMES, load_model

DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42
CRACK_INDEX = CLASS_NAMES.index("Crack")


@dataclass(frozen=True)
class EvaluationResult:
    """Raw evaluation artefacts independent of the decision threshold."""

    probs: np.ndarray          # [N, 2] softmax probabilities
    targets: np.ndarray        # [N] integer labels in {0, 1}
    sample_paths: List[Path]   # length-N absolute paths back to source images
    num_train: int
    num_val: int

    @property
    def num_samples(self) -> int:
        return int(self.targets.shape[0])


def _build_val_subset(
    data_dir: Path, val_ratio: float, seed: int
) -> Tuple[Subset, int, int]:
    full_eval = CrackDataset(data_dir, transform=get_eval_transform())
    n_total = len(full_eval)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    _, val_split = torch.utils.data.random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    val_indices = list(val_split)
    return Subset(full_eval, val_indices), n_train, n_val


def evaluate_checkpoint(
    model_path: Union[str, Path],
    data_dir: Union[str, Path],
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Union[str, torch.device] = "cpu",
) -> EvaluationResult:
    """Score ``model_path`` on the seeded held-out validation split.

    Args:
        model_path: Path to a ``.pt`` checkpoint saved by ``src.train``.
        data_dir: Root of the classification dataset (with ``Positive`` /
            ``Negative`` subfolders).
        val_ratio: Fraction of the dataset to use as validation. Must
            match the ratio used at training time for numbers to line up.
        seed: RNG seed. Must match training for numbers to line up.
        batch_size: Eval batch size.
        num_workers: DataLoader workers; ``0`` is safer inside Streamlit.
        device: Device to run inference on.

    Returns:
        An :class:`EvaluationResult` holding raw probabilities, integer
        targets, and the source paths for every validation sample.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    val_subset, n_train, n_val = _build_val_subset(
        data_dir, val_ratio=val_ratio, seed=seed
    )
    base_dataset: CrackDataset = val_subset.dataset  # type: ignore[assignment]
    sample_paths = [base_dataset.samples[i][0] for i in val_subset.indices]

    loader = DataLoader(
        val_subset,
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
        num_train=n_train,
        num_val=n_val,
    )


def metrics_at_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Mapping[str, object]:
    """Derive threshold-dependent metrics from stored probabilities.

    The model is treated as a binary crack detector: predict ``Crack``
    when ``P(Crack) >= threshold``. Using ``threshold=0.5`` reproduces the
    ``argmax`` decision rule that ``src.train`` reports.

    Returns a dict with:
        - ``accuracy`` (float)
        - ``confusion_matrix`` (np.ndarray, shape ``(2, 2)``, rows=true,
          cols=pred, row/col order ``[No Crack, Crack]``)
        - ``per_class`` (dict[str, dict[str, float]]) with
          ``precision`` / ``recall`` / ``f1`` / ``support`` per class
        - ``preds`` (np.ndarray) integer predictions
        - ``threshold`` (float) echoed back
    """
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"probs must be [N, {len(CLASS_NAMES)}]; got shape {probs.shape}"
        )
    if targets.shape[0] != probs.shape[0]:
        raise ValueError("targets and probs must have matching length")

    crack_probs = probs[:, CRACK_INDEX]
    pred_is_crack = crack_probs >= threshold
    preds = np.where(pred_is_crack, CRACK_INDEX, 1 - CRACK_INDEX).astype(np.int64)

    n = targets.shape[0]
    accuracy = float((preds == targets).sum()) / max(n, 1)

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1

    per_class = {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": per_class,
        "preds": preds,
        "threshold": float(threshold),
    }
