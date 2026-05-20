from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def msp_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability used as ID confidence score."""
    logits = logits.detach().float()
    probs = F.softmax(logits, dim=1)
    return probs.max(dim=1).values


def _binary_clf_curve(y_true: torch.Tensor, scores: torch.Tensor):
    """Compute cumulative FP/TP counts over descending thresholds.

    Args:
        y_true: 1 for ID, 0 for OOD.
        scores: Larger means more ID-like.
    """
    y_true = y_true.bool()
    scores = scores.float()

    if scores.numel() == 0:
        raise RuntimeError("Empty scores for OOD metric computation.")

    order = torch.argsort(scores, descending=True, stable=True)
    y_true = y_true[order]
    scores = scores[order]

    distinct = torch.where(scores[1:] != scores[:-1])[0]
    threshold_idxs = torch.cat(
        [
            distinct,
            torch.tensor([scores.numel() - 1], device=scores.device),
        ]
    )

    tps = torch.cumsum(y_true.float(), dim=0)[threshold_idxs]
    fps = 1 + threshold_idxs.float() - tps

    return fps, tps


def tnr_at_tpr95(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    scores = torch.cat([id_scores, ood_scores]).float()
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_pos = float(id_scores.numel())
    n_neg = float(ood_scores.numel())

    tpr = tps / max(n_pos, 1.0)
    fpr = fps / max(n_neg, 1.0)

    idx = torch.argmin(torch.abs(tpr - 0.95))
    tnr95 = 1.0 - float(fpr[idx].item())

    return 100.0 * tnr95


def auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    scores = torch.cat([id_scores, ood_scores]).float()
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_pos = float(id_scores.numel())
    n_neg = float(ood_scores.numel())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr = torch.cat([torch.zeros(1, device=scores.device), tps / n_pos])
    fpr = torch.cat([torch.zeros(1, device=scores.device), fps / n_neg])

    area = torch.trapz(tpr, fpr)
    return 100.0 * float(area.item())


def detection_accuracy(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    scores = torch.cat([id_scores, ood_scores]).float()
    y_true = torch.cat(
        [
            torch.ones_like(id_scores, dtype=torch.long),
            torch.zeros_like(ood_scores, dtype=torch.long),
        ]
    )

    thresholds = torch.unique(scores)
    best = 0.0

    for tau in thresholds:
        pred = (scores > tau).long()
        acc = (pred == y_true).float().mean()
        best = max(best, float(acc.item()))

    return 100.0 * best


def aupr(id_scores: torch.Tensor, ood_scores: torch.Tensor, positive: str) -> float:
    if positive not in {"in", "out"}:
        raise ValueError("positive must be 'in' or 'out'")

    if positive == "in":
        scores = torch.cat([id_scores, ood_scores]).float()
        y_true = torch.cat(
            [
                torch.ones_like(id_scores, dtype=torch.long),
                torch.zeros_like(ood_scores, dtype=torch.long),
            ]
        )
    else:
        # For AUPR-Out, OOD is positive. Lower MSP means more OOD-like,
        # therefore use -MSP as the positive score.
        scores = torch.cat([-id_scores, -ood_scores]).float()
        y_true = torch.cat(
            [
                torch.zeros_like(id_scores, dtype=torch.long),
                torch.ones_like(ood_scores, dtype=torch.long),
            ]
        )

    fps, tps = _binary_clf_curve(y_true, scores)

    n_pos = float(y_true.sum().item())
    if n_pos == 0:
        return float("nan")

    precision = tps / torch.clamp(tps + fps, min=1.0)
    recall = tps / n_pos

    precision = torch.cat(
        [
            torch.ones(1, device=scores.device),
            precision,
        ]
    )
    recall = torch.cat(
        [
            torch.zeros(1, device=scores.device),
            recall,
        ]
    )

    area = torch.trapz(precision, recall)
    return 100.0 * float(area.item())


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> Dict[str, float]:
    id_scores = id_scores.detach().float().cpu()
    ood_scores = ood_scores.detach().float().cpu()

    return {
        "TNR95": tnr_at_tpr95(id_scores, ood_scores),
        "AUROC": auroc(id_scores, ood_scores),
        "DetAcc": detection_accuracy(id_scores, ood_scores),
        "AUPR_In": aupr(id_scores, ood_scores, positive="in"),
        "AUPR_Out": aupr(id_scores, ood_scores, positive="out"),
    }
