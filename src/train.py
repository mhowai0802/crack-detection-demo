"""Training CLI for the concrete crack classifier.

Example
-------
    python -m src.train --data-dir data/ --epochs 5 --batch-size 64 \
        --output models/crack_classifier.pt

Strategy
--------
The backbone is frozen for the first ``--freeze-epochs`` epochs so only the
new classifier head learns. After that the whole network is unfrozen for
end-to-end fine-tuning at a lower learning rate.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.dataset import CrackDataset, get_eval_transform, get_train_transform
from src.model import CLASS_NAMES, build_model, freeze_backbone, unfreeze_all


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(
    data_dir: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """80/20 (configurable) split with separate train/eval transforms."""
    full_train = CrackDataset(data_dir, transform=get_train_transform())
    full_eval = CrackDataset(data_dir, transform=get_eval_transform())

    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_subset = Subset(full_train, list(train_split))
    val_subset = Subset(full_eval, list(val_split))
    return train_subset, val_subset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crack classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for head")
    parser.add_argument("--finetune-lr", type=float, default=1e-4, help="LR after unfreeze")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("models/crack_classifier.pt")
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading dataset from: {args.data_dir.resolve()}")
    train_set, val_set = split_dataset(args.data_dir, args.val_ratio, args.seed)
    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: train the new head only.
    freeze_backbone(model)
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("Unfreezing backbone for full fine-tuning")
            unfreeze_all(model)
            optimizer = Adam(model.parameters(), lr=args.finetune_lr)
            scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, f"Epoch {epoch} train"
        )
        val_loss, val_acc, val_preds, val_targets = run_epoch(
            model, val_loader, criterion, None, device, f"Epoch {epoch} val"
        )
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved new best model to {args.output} (val_acc={val_acc:.4f})")

    duration = time.time() - start
    print(f"\nTraining finished in {duration / 60:.1f} min. Best val acc: {best_val_acc:.4f}")

    print("\nFinal validation report:")
    print(classification_report(val_targets, val_preds, target_names=CLASS_NAMES))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(val_targets, val_preds))


if __name__ == "__main__":
    main()
