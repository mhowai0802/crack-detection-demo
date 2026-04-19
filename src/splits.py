"""Deterministic train / val / test split for the crack dataset.

One place, one recipe. Both ``src.train`` and ``src.evaluate`` import from
here so the indices line up exactly — otherwise the "test" metrics you
report would silently leak with training samples.

Why three-way instead of train / val?
-------------------------------------
With only two splits, the ``val`` set does double duty: we pick the best
checkpoint by *and* report the final accuracy on the same batch of images.
That inflates the headline number (selection bias on whatever split
happened to look good at that epoch). A held-out ``test`` split, never
touched during training or hyper-parameter selection, gives an honest
estimate of generalisation.

Contract
--------
``three_way_split_indices(n_total, val_ratio, test_ratio, seed)``
returns three disjoint, exhaustive lists of indices over
``range(n_total)``. The same ``(n_total, val_ratio, test_ratio, seed)``
always yields the same three lists, across training and evaluation.

Degenerate cases
----------------
- ``test_ratio == 0`` reproduces the old two-way behaviour: the third
  returned list is empty, and the first two match the legacy
  ``random_split(range(N), [n_train, n_val], seed)`` indices used by the
  pre-refactor code. That keeps existing checkpoints consistent.
- ``val_ratio == 0`` is accepted but strongly discouraged: without a val
  split there is no principled way to pick the best checkpoint.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import random_split

DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15
DEFAULT_SEED: int = 42

__all__ = [
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_SEED",
    "three_way_split_indices",
    "split_sizes",
]


def split_sizes(
    n_total: int, val_ratio: float, test_ratio: float
) -> Tuple[int, int, int]:
    """Resolve ``(n_train, n_val, n_test)`` from ratios, floor-rounded.

    ``n_train`` absorbs the rounding slack so the three counts always
    sum to ``n_total``.
    """
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0, "
            f"got {val_ratio} + {test_ratio}"
        )
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    return n_train, n_val, n_test


def three_way_split_indices(
    n_total: int,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[int], List[int], List[int]]:
    """Return ``(train_idx, val_idx, test_idx)`` as disjoint index lists.

    Uses :func:`torch.utils.data.random_split` with a seeded generator so
    the shuffle is reproducible across Python processes and machines (as
    long as the PyTorch version matches).
    """
    n_train, n_val, n_test = split_sizes(n_total, val_ratio, test_ratio)
    generator = torch.Generator().manual_seed(int(seed))
    train_split, val_split, test_split = random_split(
        range(n_total), [n_train, n_val, n_test], generator=generator
    )
    return list(train_split), list(val_split), list(test_split)
