"""Training CLI for the binary crack segmentation U-Net.

Example
-------
    python -m src.seg_train --data-dir data_seg/ --epochs 20 \
        --output models/crack_segmenter.pt

Loss
----
BCE-with-logits (pixel-wise) + Dice loss, weighted equally. BCE alone
underweights the (rare) crack class on thin cracks; Dice drives the
optimiser toward maximising overlap even when cracks occupy <5% of
pixels.

Metric
------
Intersection-over-Union (IoU) on the validation split after each
epoch. The epoch with the best val IoU is saved; early stopping kicks
in after ``--patience`` epochs without improvement.

CPU feasibility
---------------
Default config (``base_channels=16``, ``image_size=384``,
``batch_size=4``) fits comfortably in laptop RAM. One epoch on
DeepCrack (240 images) takes ~5-8 min on a CPU, so 20 epochs ~ 2 hr.
For a short smoke-test run use ``--epochs 1 --image-size 128``.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.seg_dataset import (
    DEFAULT_SEG_IMAGE_SIZE,
    CrackSegDataset,
    get_eval_transform,
    get_train_transform,
)
from src.seg_model import build_unet, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss on sigmoid-ed logits.

    Both tensors are expected to be shape ``[N, 1, H, W]`` with targets
    already in ``{0, 1}``.
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (probs * targets).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> float:
    """Batch-mean IoU for the positive (crack) class."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        intersection = (preds * targets).sum()
        union = ((preds + targets) > 0).float().sum()
        if union.item() == 0:
            return 1.0 if intersection.item() == 0 else 0.0
        return float(intersection.item() / union.item())


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    bce_weight: float = 0.5,
) -> tuple[float, float]:
    """Run a training or eval epoch.

    Returns ``(mean_loss, mean_iou)`` averaged over the number of
    samples seen (not batches) so the report is comparable across
    different batch sizes.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_iou = 0.0
    total_items = 0

    for images, masks in tqdm(
        loader,
        leave=False,
        desc="train" if train_mode else "eval",
    ):
        images = images.to(device)
        masks = masks.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = bce_weight * bce(logits, masks) + (
                1.0 - bce_weight
            ) * dice_loss(logits, masks)

            if train_mode:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_iou += iou_score(logits, masks) * batch_size
        total_items += batch_size

    if total_items == 0:
        return 0.0, 0.0
    return total_loss / total_items, total_iou / total_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the binary crack segmentation U-Net."
    )
    parser.add_argument("--data-dir", default="data_seg/", type=Path)
    parser.add_argument("--output", default="models/crack_segmenter.pt", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=DEFAULT_SEG_IMAGE_SIZE)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="'auto' picks cuda > mps > cpu.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    train_ds = CrackSegDataset(
        args.data_dir / "train", transform=get_train_transform(args.image_size)
    )
    val_ds = CrackSegDataset(
        args.data_dir / "val", transform=get_eval_transform(args.image_size)
    )
    test_ds_path = args.data_dir / "test"
    test_ds = (
        CrackSegDataset(test_ds_path, transform=get_eval_transform(args.image_size))
        if test_ds_path.is_dir()
        else None
    )

    print(
        f"Data: train={len(train_ds)}  val={len(val_ds)}"
        + (f"  test={len(test_ds)}" if test_ds else "  test=-")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_unet(base_channels=args.base_channels).to(device)
    print(f"Model: UNet(base={args.base_channels}), params={count_parameters(model):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = -1.0
    best_epoch = -1
    stale = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = run_one_epoch(
            model, train_loader, device, optimizer, bce_weight=args.bce_weight
        )
        val_loss, val_iou = run_one_epoch(
            model, val_loader, device, optimizer=None, bce_weight=args.bce_weight
        )
        scheduler.step()
        dt = time.time() - t0

        improved = val_iou > best_val_iou
        marker = "  *" if improved else ""
        print(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.3f}  "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({dt:.1f}s){marker}"
        )

        if improved:
            best_val_iou = val_iou
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "base_channels": args.base_channels,
                    "image_size": args.image_size,
                    "val_iou": val_iou,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(
                    f"Early stop: no val IoU improvement for {args.patience} epochs."
                )
                break

    print(f"Best val IoU: {best_val_iou:.3f} at epoch {best_epoch}")
    print(f"Saved to {args.output}")

    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        best_state = torch.load(args.output, map_location=device)
        model.load_state_dict(best_state["state_dict"])
        _, test_iou = run_one_epoch(
            model, test_loader, device, optimizer=None, bce_weight=args.bce_weight
        )
        print(f"Held-out test IoU (best-val checkpoint): {test_iou:.3f}")


if __name__ == "__main__":
    main()
