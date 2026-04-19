"""Threshold-dependent classification metrics.

Kept torch-free so both the FastAPI backend and the Streamlit frontend
can import it without pulling in ``torch`` / ``torchvision``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from src.constants import CLASS_NAMES, CRACK_INDEX


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration.

    Uses ``np.trapezoid`` when available (numpy >= 2.0) and falls back
    to the deprecated ``np.trapz`` otherwise, so the module stays
    compatible across numpy versions.
    """
    fn = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(fn(y, x))


def metrics_at_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Mapping[str, object]:
    """Derive threshold-dependent metrics from stored probabilities.

    The classifier is treated as a binary crack detector: predict
    ``Crack`` when ``P(Crack) >= threshold``. ``threshold=0.5``
    reproduces the ``argmax`` decision rule that ``src.train`` reports.

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


def roc_curve_points(
    probs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Compute a ROC curve for the ``Crack`` (positive) class.

    ROC curves plot **True Positive Rate** (recall for the positive
    class) against **False Positive Rate** (1 - recall for the negative
    class) as the decision threshold sweeps from +inf down to -inf. An
    ideal classifier hugs the top-left; a random classifier sits on the
    diagonal. The **AUC** (area under the curve) is the probability
    that a random positive sample scores higher than a random negative
    one — threshold-independent, so it's a useful single-number quality
    summary.

    Implementation follows the same vectorised sweep as
    ``sklearn.metrics.roc_curve`` but is pure numpy so ``src.metrics``
    stays torch / sklearn free.

    Args:
        probs: ``[N, 2]`` softmax probabilities (column order matches
            ``CLASS_NAMES``, so ``probs[:, CRACK_INDEX]`` is the crack
            score).
        targets: ``[N]`` integer labels (``0`` for ``No Crack``, ``1``
            for ``Crack``).

    Returns:
        Dict with:
            - ``fpr`` (np.ndarray): false positive rate at each
              retained threshold, monotonically non-decreasing, starts
              at ``0.0``, ends at ``1.0``.
            - ``tpr`` (np.ndarray): true positive rate at each
              retained threshold, same length as ``fpr``.
            - ``thresholds`` (np.ndarray): the P(Crack) cut-offs
              corresponding to each ``(fpr, tpr)`` pair. The first
              entry is ``+inf`` (predict nothing as crack).
            - ``auc`` (float): area under the ROC curve. Returns
              ``NaN`` when only one class is present in ``targets``.
    """
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"probs must be [N, {len(CLASS_NAMES)}]; got shape {probs.shape}"
        )
    if targets.shape[0] != probs.shape[0]:
        raise ValueError("targets and probs must have matching length")

    n = int(targets.shape[0])
    empty = np.array([0.0, 1.0])
    if n == 0:
        return {
            "fpr": empty,
            "tpr": empty,
            "thresholds": np.array([np.inf, -np.inf]),
            "auc": float("nan"),
        }

    scores = probs[:, CRACK_INDEX].astype(np.float64)
    y_true = (targets == CRACK_INDEX).astype(np.int64)

    # Sort by score descending so we sweep from strict to permissive.
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    y_sorted = y_true[order]

    # Collapse stretches of equal score into a single threshold point.
    distinct_idxs = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.concatenate([distinct_idxs, [n - 1]])

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    total_pos = int(tps[-1])
    total_neg = int(fps[-1])

    if total_pos == 0 or total_neg == 0:
        return {
            "fpr": empty,
            "tpr": empty,
            "thresholds": np.array([np.inf, -np.inf]),
            "auc": float("nan"),
        }

    tpr = tps / total_pos
    fpr = fps / total_neg
    thresholds = scores_sorted[threshold_idxs]

    # Prepend the "predict nothing positive" point so the curve always
    # starts at the origin and ends at (1, 1).
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[np.inf], thresholds])

    auc = _trapezoid(tpr, fpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc),
    }


def point_on_roc(
    roc: Mapping[str, np.ndarray | float], threshold: float
) -> Dict[str, float]:
    """Locate ``(fpr, tpr)`` on a ROC curve at a given P(Crack) cut-off.

    Since the curve is discretised at score-change boundaries, we pick
    the operating point as the *strictest* cut-off whose threshold is
    ``<= threshold``. If ``threshold`` is above every score, the curve
    is at the origin; if below every score, at ``(1, 1)``.
    """
    fpr = np.asarray(roc["fpr"])
    tpr = np.asarray(roc["tpr"])
    thresholds = np.asarray(roc["thresholds"])
    if thresholds.size == 0:
        return {"fpr": 0.0, "tpr": 0.0, "threshold": float(threshold)}

    # thresholds is monotonically non-increasing; find first index
    # where threshold_at_idx <= requested threshold.
    mask = thresholds <= threshold
    if not mask.any():
        idx = thresholds.size - 1
    else:
        idx = int(np.argmax(mask))
    return {
        "fpr": float(fpr[idx]),
        "tpr": float(tpr[idx]),
        "threshold": float(threshold),
    }
