
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def msp_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute maximum softmax probability as the ID confidence score.

    Larger values indicate that a sample is more likely to come from the
    in-distribution data.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be a 2D tensor of shape [N, C], "
            f"but got shape {tuple(logits.shape)}"
        )

    logits = logits.detach().float()
    probs = F.softmax(logits, dim=1)

    return probs.max(dim=1).values


def _validate_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize ID/OOD score tensors."""
    id_scores = id_scores.detach().float().reshape(-1)
    ood_scores = ood_scores.detach().float().reshape(-1)

    if id_scores.numel() == 0:
        raise RuntimeError("ID scores are empty.")

    if ood_scores.numel() == 0:
        raise RuntimeError("OOD scores are empty.")

    if id_scores.device != ood_scores.device:
        ood_scores = ood_scores.to(id_scores.device)

    if not torch.isfinite(id_scores).all():
        raise RuntimeError("ID scores contain NaN or Inf values.")

    if not torch.isfinite(ood_scores).all():
        raise RuntimeError("OOD scores contain NaN or Inf values.")

    return id_scores, ood_scores


def _binary_clf_curve(
    y_true: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cumulative FP and TP counts at distinct score thresholds.

    Args:
        y_true:
            Binary labels where 1 denotes the positive class and 0 denotes
            the negative class.
        scores:
            Detection scores. Larger values indicate a higher probability
            of belonging to the positive class.

    Returns:
        fps:
            Cumulative false-positive counts at each distinct threshold.
        tps:
            Cumulative true-positive counts at each distinct threshold.

    Notes:
        Samples with scores greater than or equal to a threshold are
        predicted as positive. Samples with the same score are processed
        together, preventing arbitrary tie breaking.
    """
    y_true = y_true.reshape(-1).bool()
    scores = scores.reshape(-1).float()

    if y_true.numel() != scores.numel():
        raise ValueError(
            f"y_true and scores must have the same number of elements, "
            f"but got {y_true.numel()} and {scores.numel()}."
        )

    if scores.numel() == 0:
        raise RuntimeError("Empty scores for OOD metric computation.")

    order = torch.argsort(scores, descending=True, stable=True)
    y_true_sorted = y_true[order]
    scores_sorted = scores[order]

    # Keep the final index of every group of identical scores.
    distinct_value_indices = torch.where(
        scores_sorted[1:] != scores_sorted[:-1]
    )[0]

    threshold_indices = torch.cat(
        [
            distinct_value_indices,
            torch.tensor(
                [scores_sorted.numel() - 1],
                dtype=torch.long,
                device=scores.device,
            ),
        ]
    )

    cumulative_positives = torch.cumsum(
        y_true_sorted.to(torch.float64),
        dim=0,
    )

    tps = cumulative_positives[threshold_indices]
    fps = (
        threshold_indices.to(torch.float64)
        + 1.0
        - tps
    )

    return fps, tps


def tnr_at_tpr95(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> float:
    """Compute TNR at the threshold whose ID TPR is closest to 95%.

    ID samples are treated as the positive class. Larger scores indicate
    that a sample is more ID-like.
    """
    id_scores, ood_scores = _validate_scores(id_scores, ood_scores)

    scores = torch.cat([id_scores, ood_scores])
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_id = float(id_scores.numel())
    n_ood = float(ood_scores.numel())

    tpr = tps / n_id
    fpr = fps / n_ood

    # Match the implementation used in the Mahalanobis OOD evaluation code:
    # choose the operating point whose ID true-positive rate is closest to 0.95.
    index = torch.argmin(torch.abs(tpr - 0.95))

    tnr95 = 1.0 - fpr[index]

    return 100.0 * float(tnr95.item())


def auroc(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> float:
    """Compute AUROC with ID samples treated as the positive class."""
    id_scores, ood_scores = _validate_scores(id_scores, ood_scores)

    scores = torch.cat([id_scores, ood_scores])
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_id = float(id_scores.numel())
    n_ood = float(ood_scores.numel())

    zero = torch.zeros(
        1,
        dtype=torch.float64,
        device=scores.device,
    )

    # _binary_clf_curve already contains the final point (1, 1).
    # Add the initial point (0, 0).
    tpr = torch.cat([zero, tps / n_id])
    fpr = torch.cat([zero, fps / n_ood])

    area = torch.trapz(tpr, fpr)

    return 100.0 * float(area.item())


def detection_accuracy(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> float:
    """Compute maximum balanced ID/OOD detection accuracy.

    ID samples are treated as the positive class, and larger scores indicate
    that a sample is more ID-like.

    The metric follows:

        DetAcc = max_tau 0.5 * (TPR(tau) + TNR(tau))

    This gives equal weight to ID and OOD samples and therefore does not
    depend directly on the relative numbers of ID and OOD test samples.
    """
    id_scores, ood_scores = _validate_scores(id_scores, ood_scores)

    scores = torch.cat([id_scores, ood_scores])
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_id = float(id_scores.numel())
    n_ood = float(ood_scores.numel())

    zero = torch.zeros(
        1,
        dtype=torch.float64,
        device=scores.device,
    )

    # Add the operating point corresponding to a threshold above the maximum
    # score. At this point, every sample is predicted as OOD:
    #
    # TPR = 0, FPR = 0, TNR = 1.
    #
    # _binary_clf_curve already includes the opposite endpoint where every
    # sample is predicted as ID.
    tpr = torch.cat([zero, tps / n_id])
    fpr = torch.cat([zero, fps / n_ood])
    tnr = 1.0 - fpr

    balanced_accuracy = 0.5 * (tpr + tnr)
    best_accuracy = balanced_accuracy.max()

    return 100.0 * float(best_accuracy.item())


def aupr(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    positive: str,
) -> float:
    """Compute AUPR-In or AUPR-Out.

    Args:
        id_scores:
            MSP scores for ID samples. Larger scores indicate more ID-like
            samples.
        ood_scores:
            MSP scores for OOD samples.
        positive:
            ``"in"`` treats ID as the positive class.
            ``"out"`` treats OOD as the positive class.

    Returns:
        Area under the corresponding precision-recall curve, expressed as
        a percentage.
    """
    id_scores, ood_scores = _validate_scores(id_scores, ood_scores)

    if positive not in {"in", "out"}:
        raise ValueError(
            f"positive must be either 'in' or 'out', but got {positive!r}"
        )

    if positive == "in":
        # ID is positive; larger MSP means more likely to be positive.
        scores = torch.cat([id_scores, ood_scores])
        y_true = torch.cat(
            [
                torch.ones_like(id_scores, dtype=torch.long),
                torch.zeros_like(ood_scores, dtype=torch.long),
            ]
        )
    else:
        # OOD is positive. Since lower MSP means more OOD-like, negate MSP
        # so that larger values consistently indicate the positive class.
        scores = torch.cat([-id_scores, -ood_scores])
        y_true = torch.cat(
            [
                torch.zeros_like(id_scores, dtype=torch.long),
                torch.ones_like(ood_scores, dtype=torch.long),
            ]
        )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_positive = float(y_true.sum().item())

    precision = tps / torch.clamp(tps + fps, min=1.0)
    recall = tps / n_positive

    one = torch.ones(
        1,
        dtype=torch.float64,
        device=scores.device,
    )
    zero = torch.zeros(
        1,
        dtype=torch.float64,
        device=scores.device,
    )

    # Precision-recall curve begins at recall = 0, precision = 1.
    precision = torch.cat([one, precision])
    recall = torch.cat([zero, recall])

    area = torch.trapz(precision, recall)

    return 100.0 * float(area.item())


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> Dict[str, float]:
    """Compute all OOD detection metrics used by the project."""
    id_scores, ood_scores = _validate_scores(id_scores, ood_scores)

    return {
        "TNR95": tnr_at_tpr95(id_scores, ood_scores),
        "AUROC": auroc(id_scores, ood_scores),
        "DetAcc": detection_accuracy(id_scores, ood_scores),
        "AUPR_In": aupr(
            id_scores,
            ood_scores,
            positive="in",
        ),
        "AUPR_Out": aupr(
            id_scores,
            ood_scores,
            positive="out",
        ),
    }

