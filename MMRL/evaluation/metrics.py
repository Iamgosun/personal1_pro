from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    num_classes = int(logits.shape[1])
    f1_scores = []

    for cls_idx in range(num_classes):
        pred_pos = preds == cls_idx
        true_pos = labels == cls_idx

        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()

        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2.0 * tp / denom)
        f1_scores.append(f1)

    return float(sum(f1_scores) / max(len(f1_scores), 1) * 100.0)


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_probs = F.log_softmax(logits, dim=1)
    return float(F.nll_loss(log_probs, labels, reduction="mean").item())


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    score = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return float(score.item())


def calibration_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[float, List[Dict[str, float]]]:
    probs = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)

    bins: List[Dict[str, float]] = []
    ece = 0.0
    total = int(labels.numel())

    for idx in range(n_bins):
        left = float(bin_edges[idx].item())
        right = float(bin_edges[idx + 1].item())

        if idx == 0:
            in_bin = (confidences >= bin_edges[idx]) & (confidences <= bin_edges[idx + 1])
        else:
            in_bin = (confidences > bin_edges[idx]) & (confidences <= bin_edges[idx + 1])

        count = int(in_bin.sum().item())
        fraction = float(count / total) if total > 0 else 0.0

        if count > 0:
            bin_conf = float(confidences[in_bin].mean().item() * 100.0)
            bin_acc = float(correctness[in_bin].mean().item() * 100.0)
            gap = abs(bin_acc - bin_conf)
            weighted_gap = gap * fraction
            correct_count = int(correctness[in_bin].sum().item())
        else:
            bin_conf = 0.0
            bin_acc = 0.0
            gap = 0.0
            weighted_gap = 0.0
            correct_count = 0

        ece += weighted_gap

        bins.append(
            {
                "bin_index": idx,
                "range_left": left,
                "range_right": right,
                "count": count,
                "fraction": fraction,
                "correct_count": correct_count,
                "avg_confidence": bin_conf,
                "avg_accuracy": bin_acc,
                "gap": gap,
                "weighted_gap": weighted_gap,
            }
        )

    return float(ece), bins


def build_classification_calibration_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Dict[str, object]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    acc = accuracy(logits, labels)
    err = 100.0 - acc
    mf1 = macro_f1(logits, labels)
    nll = negative_log_likelihood(logits, labels)
    brier = brier_score(logits, labels)
    ece, bins = calibration_bins(logits, labels, n_bins=n_bins)

    metrics = {
        "accuracy": acc,
        "error": err,
        "macro_f1": mf1,
        "nll": nll,
        "brier": brier,
        "ece": ece,
    }

    return {
        "schema_version": 1,
        "num_samples": int(labels.numel()),
        "metrics": metrics,
        "prediction": {
            "accuracy": acc,
            "error": err,
            "macro_f1": mf1,
            "nll": nll,
            "brier": brier,
        },
        "calibration": {
            "ece": ece,
            "n_bins": int(n_bins),
            "bins": bins,
        },
        "ood": {},
    }


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def save_metric_report(
    output_dir: str,
    split: str,
    report: Dict[str, object],
) -> Dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / f"{split}_metrics.json"
    metrics_csv_path = outdir / f"{split}_metrics.csv"
    bins_csv_path = outdir / f"{split}_calibration_bins.csv"

    report = _to_builtin(report)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    metric_row = dict(report.get("metrics", {}))
    metric_row["num_samples"] = report.get("num_samples", 0)

    with metrics_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row.keys()))
        writer.writeheader()
        writer.writerow(metric_row)

    bins = report.get("calibration", {}).get("bins", [])
    if bins:
        with bins_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
            writer.writeheader()
            writer.writerows(bins)
    else:
        with bins_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "bin_index",
                    "range_left",
                    "range_right",
                    "count",
                    "fraction",
                    "correct_count",
                    "avg_confidence",
                    "avg_accuracy",
                    "gap",
                    "weighted_gap",
                ]
            )

    return {
        "json": str(json_path),
        "metrics_csv": str(metrics_csv_path),
        "bins_csv": str(bins_csv_path),
    }