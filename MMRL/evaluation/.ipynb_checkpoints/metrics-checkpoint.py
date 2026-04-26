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


class TemperatureScaler(torch.nn.Module):
    """Post-hoc temperature scaling for classification logits.

    This module does not change predictions' argmax. It only rescales logits
    before softmax, usually improving NLL/ECE when the original model is
    over-confident or under-confident.
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")

        self.log_temperature = torch.nn.Parameter(
            torch.tensor(float(init_temperature)).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=1e-6, max=1e6)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
    device: str | torch.device | None = None,
) -> float:
    """Fit one scalar temperature on validation logits by minimizing NLL.

    Args:
        logits: Validation logits with shape [N, C].
        labels: Validation labels with shape [N].
        max_iter: LBFGS maximum iterations.
        lr: LBFGS learning rate.
        device: Device used to optimize temperature.

    Returns:
        A positive scalar temperature.
    """
    if logits.numel() == 0 or labels.numel() == 0:
        return 1.0

    if device is None:
        device = logits.device

    logits = logits.detach().to(device).float()
    labels = labels.detach().to(device).long()

    scaler = TemperatureScaler(init_temperature=1.0).to(device)

    optimizer = torch.optim.LBFGS(
        [scaler.log_temperature],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(scaler.temperature.detach().cpu().item())

    if not torch.isfinite(torch.tensor(temperature)):
        return 1.0

    return temperature


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply a fitted temperature to logits."""
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


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
            in_bin = (confidences >= bin_edges[idx]) & (
                confidences <= bin_edges[idx + 1]
            )
        else:
            in_bin = (confidences > bin_edges[idx]) & (
                confidences <= bin_edges[idx + 1]
            )

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


def _write_metrics_csv(path: Path, metrics: Dict[str, object], num_samples: int):
    metric_row = dict(metrics)
    metric_row["num_samples"] = num_samples

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_row.keys()))
        writer.writeheader()
        writer.writerow(metric_row)


def _write_bins_csv(path: Path, bins: List[Dict[str, object]]):
    default_fields = [
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

    with path.open("w", encoding="utf-8", newline="") as f:
        if bins:
            writer = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
            writer.writeheader()
            writer.writerows(bins)
        else:
            writer = csv.writer(f)
            writer.writerow(default_fields)


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

    calibrated_metrics_csv_path = outdir / f"{split}_metrics_calibrated.csv"
    calibrated_bins_csv_path = outdir / f"{split}_calibration_bins_calibrated.csv"

    report = _to_builtin(report)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    num_samples = int(report.get("num_samples", 0))

    _write_metrics_csv(
        metrics_csv_path,
        dict(report.get("metrics", {})),
        num_samples=num_samples,
    )

    _write_bins_csv(
        bins_csv_path,
        list(report.get("calibration", {}).get("bins", [])),
    )

    saved_paths = {
        "json": str(json_path),
        "metrics_csv": str(metrics_csv_path),
        "bins_csv": str(bins_csv_path),
    }

    metrics_calibrated = report.get("metrics_calibrated")
    calibration_calibrated = report.get("calibration_calibrated")

    if metrics_calibrated is not None:
        calibrated_metric_row = dict(metrics_calibrated)

        temperature_info = report.get("temperature_scaling", {})
        if isinstance(temperature_info, dict) and "temperature" in temperature_info:
            calibrated_metric_row["temperature"] = temperature_info["temperature"]

        _write_metrics_csv(
            calibrated_metrics_csv_path,
            calibrated_metric_row,
            num_samples=num_samples,
        )
        saved_paths["metrics_calibrated_csv"] = str(calibrated_metrics_csv_path)

    if calibration_calibrated is not None:
        _write_bins_csv(
            calibrated_bins_csv_path,
            list(calibration_calibrated.get("bins", [])),
        )
        saved_paths["bins_calibrated_csv"] = str(calibrated_bins_csv_path)

    return saved_paths