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
    """Fit one scalar temperature on validation logits by minimizing NLL."""
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


def fixed_calibration_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    """Fixed-width ECE bins.

    Units:
        avg_confidence, avg_accuracy, gap, weighted_gap, and returned ECE are
        percentage points, matching the rest of this repository's metrics.
    """
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
                "bin_type": "fixed_width",
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


def adaptive_calibration_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    """Adaptive ECE bins using confidence quantiles.

    This implements AECE with the same gap formula as ECE, but bin boundaries
    are confidence quantiles rather than fixed-width intervals.

    Units:
        avg_confidence, avg_accuracy, gap, weighted_gap, and returned AECE are
        percentage points, matching the rest of this repository's metrics.
    """
    probs = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    total = int(labels.numel())
    if total == 0:
        return float("nan"), []

    quantiles = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=logits.device)
    bin_edges = torch.quantile(confidences, quantiles)

    bins: List[Dict[str, float]] = []
    aece = 0.0

    for idx in range(n_bins):
        left_t = bin_edges[idx]
        right_t = bin_edges[idx + 1]

        # Include the left edge in the first bin so confidence==min is not lost.
        if idx == 0:
            in_bin = (confidences >= left_t) & (confidences <= right_t)
        else:
            in_bin = (confidences > left_t) & (confidences <= right_t)

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

        aece += weighted_gap

        bins.append(
            {
                "bin_index": idx,
                "bin_type": "adaptive_quantile",
                "range_left": float(left_t.item()),
                "range_right": float(right_t.item()),
                "count": count,
                "fraction": fraction,
                "correct_count": correct_count,
                "avg_confidence": bin_conf,
                "avg_accuracy": bin_acc,
                "gap": gap,
                "weighted_gap": weighted_gap,
            }
        )

    return float(aece), bins


def calibration_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, float]]]:
    """Backward-compatible alias for the repository's previous ECE function."""
    return fixed_calibration_bins(logits=logits, labels=labels, n_bins=n_bins)


def confidence_threshold_coverage_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: List[float] | None = None,
) -> Dict[str, object]:
    """Confidence-threshold coverage used by the high-confidence experiments.

    Definition:
        S_tau = {i | max_c p_i,c > tau}
        coverage_tau = |S_tau| / N
        accuracy_tau = accuracy on S_tau
        valid_coverage_tau = coverage_tau if accuracy_tau >= tau else 0

    This is intentionally separate from risk-coverage / AURC selective
    prediction. It follows the experiment definition based on confidence
    thresholds 99 / 95 / 90 / 85 / 80%.
    """
    if thresholds is None:
        thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]

    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    probs = F.softmax(logits, dim=1)
    confidences, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    total = int(labels.numel())
    rows: List[Dict[str, object]] = []
    metrics: Dict[str, object] = {}

    for tau in thresholds:
        tau = float(tau)
        tau_pct = tau * 100.0
        tau_key = str(int(round(tau_pct)))

        selected = confidences > tau
        count = int(selected.sum().item())
        coverage = float(count / total) if total > 0 else 0.0

        if count > 0:
            selected_accuracy = float(correctness[selected].mean().item())
            selected_accuracy_pct = selected_accuracy * 100.0
            valid = bool(selected_accuracy >= tau)
        else:
            selected_accuracy_pct = None
            valid = False

        valid_coverage = coverage if valid else 0.0

        # Flat scalar metrics for result_parser aggregation.
        metrics[f"confthr_{tau_key}_coverage"] = coverage * 100.0
        metrics[f"confthr_{tau_key}_accuracy"] = selected_accuracy_pct
        metrics[f"confthr_{tau_key}_valid"] = float(valid)
        metrics[f"confthr_{tau_key}_valid_coverage"] = valid_coverage * 100.0
        metrics[f"confthr_{tau_key}_count"] = count

        rows.append(
            {
                "threshold": tau,
                "threshold_pct": tau_pct,
                "count": count,
                "total": total,
                "coverage": coverage * 100.0,
                "accuracy": selected_accuracy_pct,
                "valid": valid,
                "valid_coverage": valid_coverage * 100.0,
            }
        )

    return {
        "type": "confidence_threshold_coverage",
        "comparison": "max_softmax_confidence > threshold",
        "valid_rule": "accuracy_on_selected_samples >= threshold",
        "units": "percentage_points_for_coverage_and_accuracy",
        "thresholds": thresholds,
        "metrics": metrics,
        "rows": rows,
    }


def _binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """
    AUROC for binary targets.

    Args:
        scores: Higher means more likely positive.
        targets: 1 for positive, 0 for negative.

    Returns:
        AUROC in [0, 1]. Returns NaN if only one class is present.
    """
    scores = scores.detach().float().reshape(-1)
    targets = targets.detach().long().reshape(-1)

    n = int(scores.numel())
    if n == 0:
        return float("nan")

    positives = targets == 1
    negatives = targets == 0
    n_pos = int(positives.sum().item())
    n_neg = int(negatives.sum().item())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]

    ranks = torch.empty(n, dtype=torch.float64, device=scores.device)

    # Average ranks for ties. Ranks are 1-based.
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1

        avg_rank = 0.5 * (float(start + 1) + float(end))
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = ranks[positives].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc.detach().cpu().item())


def _risk_coverage_curve_from_uncertainty(
    uncertainty: torch.Tensor,
    errors: torch.Tensor,
    score_name: str,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Build risk-coverage curve.

    uncertainty:
        Higher means more uncertain / more likely to be rejected.

    Sorting:
        Keep least uncertain samples first. As coverage increases, include more
        uncertain samples.

    AURC:
        Mean selective risk over all coverage levels k/N.

    EAURC:
        AURC minus optimal AURC for the same number of errors.
    """
    uncertainty = uncertainty.detach().float().reshape(-1)
    errors = errors.detach().float().reshape(-1)

    n = int(errors.numel())
    if n == 0:
        return float("nan"), float("nan"), []

    order = torch.argsort(uncertainty, descending=False, stable=True)
    sorted_errors = errors[order]

    cum_errors = torch.cumsum(sorted_errors, dim=0)
    counts = torch.arange(1, n + 1, device=errors.device, dtype=torch.float32)
    coverage = counts / float(n)
    risk = cum_errors / counts
    selective_accuracy = 1.0 - risk

    aurc = float(risk.mean().item())

    # Optimal curve keeps all correct samples before all wrong samples.
    optimal_errors = torch.sort(errors, descending=False).values
    optimal_cum_errors = torch.cumsum(optimal_errors, dim=0)
    optimal_risk = optimal_cum_errors / counts
    optimal_aurc = float(optimal_risk.mean().item())
    eaurc = aurc - optimal_aurc

    curve: List[Dict[str, float]] = []
    for idx in range(n):
        kept_index = int(idx + 1)
        curve.append(
            {
                "score_name": score_name,
                "rank": kept_index,
                "coverage": float(coverage[idx].item()),
                "risk": float(risk[idx].item()),
                "selective_accuracy": float(selective_accuracy[idx].item()),
                "num_kept": kept_index,
                "num_total": n,
                "num_errors_kept": float(cum_errors[idx].item()),
                "threshold_uncertainty": float(uncertainty[order[idx]].item()),
            }
        )

    return aurc, eaurc, curve


def _coverage_summary_from_curve(
    curve: List[Dict[str, float]],
    coverages: List[float] | None = None,
) -> List[Dict[str, float]]:
    if coverages is None:
        coverages = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1]

    if not curve:
        return []

    rows: List[Dict[str, float]] = []
    n = int(curve[-1]["num_total"])
    score_name = str(curve[-1]["score_name"])

    for cov in coverages:
        cov = float(cov)
        k = max(1, min(n, int(round(cov * n))))
        row = dict(curve[k - 1])
        row["requested_coverage"] = cov
        row["score_name"] = score_name
        rows.append(row)

    return rows


def selective_prediction_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    """
    Build risk-coverage selective prediction metrics from logits.

    Note:
        This is NOT the confidence-threshold coverage used in the main
        high-confidence coverage experiment. That experiment is implemented by
        confidence_threshold_coverage_report().
    """
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    errors = (preds != labels).long()

    confidences, _ = probs.max(dim=1)
    top2 = torch.topk(probs, k=min(2, probs.shape[1]), dim=1).values

    if top2.shape[1] == 1:
        margin = torch.ones_like(confidences)
    else:
        margin = top2[:, 0] - top2[:, 1]

    entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(dim=1)

    uncertainty_scores = {
        "least_confidence": 1.0 - confidences,
        "entropy": entropy,
        "margin_uncertainty": 1.0 - margin,
    }

    metrics: Dict[str, float] = {}
    curves: Dict[str, List[Dict[str, float]]] = {}
    coverage_summary: Dict[str, List[Dict[str, float]]] = {}

    for score_name, uncertainty in uncertainty_scores.items():
        aurc, eaurc, curve = _risk_coverage_curve_from_uncertainty(
            uncertainty=uncertainty,
            errors=errors,
            score_name=score_name,
        )
        auroc = _binary_auroc(uncertainty, errors)

        metrics[f"{score_name}_aurc"] = aurc
        metrics[f"{score_name}_eaurc"] = eaurc
        metrics[f"{score_name}_error_auroc"] = auroc

        curves[score_name] = curve
        coverage_summary[score_name] = _coverage_summary_from_curve(curve)

    return {
        "type": "risk_coverage_selective_prediction",
        "note": (
            "These are AURC/EAURC and error-detection metrics. "
            "They are distinct from confidence-threshold coverage."
        ),
        "metrics": metrics,
        "curves": curves,
        "coverage_summary": coverage_summary,
        "error_rate": float(errors.float().mean().item()),
        "num_errors": int(errors.sum().item()),
        "num_correct": int((errors == 0).sum().item()),
    }


def build_classification_calibration_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, object]:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu()

    acc = accuracy(logits, labels)
    err = 100.0 - acc
    mf1 = macro_f1(logits, labels)
    nll = negative_log_likelihood(logits, labels)
    brier = brier_score(logits, labels)

    ece, fixed_bins = fixed_calibration_bins(logits, labels, n_bins=n_bins)
    aece, adaptive_bins = adaptive_calibration_bins(logits, labels, n_bins=n_bins)

    confthr = confidence_threshold_coverage_report(logits, labels)
    selective = selective_prediction_report(logits, labels)

    metrics: Dict[str, object] = {
        "accuracy": acc,
        "error": err,
        "macro_f1": mf1,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "aece": aece,
    }

    # Flatten confidence-threshold coverage first because these are part of the
    # formal experiment protocol.
    for key, value in confthr["metrics"].items():
        metrics[key] = value

    # Keep risk-coverage metrics for supplementary analysis. They should not be
    # interpreted as confidence-threshold coverage.
    for key, value in selective["metrics"].items():
        metrics[key] = value

    return {
        "schema_version": 3,
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
            "aece": aece,
            "n_bins": int(n_bins),
            "fixed_bins": fixed_bins,
            # Backward-compatible key for old code expecting calibration["bins"].
            "bins": fixed_bins,
            "adaptive_bins": adaptive_bins,
        },
        "confidence_threshold_coverage": confthr,
        "selective_prediction": selective,
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
        "bin_type",
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


def _write_generic_rows_csv(path: Path, rows: List[Dict[str, object]]):
    with path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow([])


def _flatten_selective_curves(report: Dict[str, object]) -> List[Dict[str, object]]:
    selective = report.get("selective_prediction", {})
    if not isinstance(selective, dict):
        return []

    curves = selective.get("curves", {})
    if not isinstance(curves, dict):
        return []

    rows: List[Dict[str, object]] = []
    for _score_name, curve in curves.items():
        if isinstance(curve, list):
            rows.extend(curve)

    return rows


def _flatten_selective_summary(report: Dict[str, object]) -> List[Dict[str, object]]:
    selective = report.get("selective_prediction", {})
    if not isinstance(selective, dict):
        return []

    summaries = selective.get("coverage_summary", {})
    if not isinstance(summaries, dict):
        return []

    rows: List[Dict[str, object]] = []
    for _score_name, summary_rows in summaries.items():
        if isinstance(summary_rows, list):
            rows.extend(summary_rows)

    return rows


def save_metric_report(
    output_dir: str,
    split: str,
    report: Dict[str, object],
) -> Dict[str, str]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # New canonical JSON name.
    report_json_path = outdir / f"{split}_report.json"

    metrics_csv_path = outdir / f"{split}_metrics.csv"

    fixed_bins_csv_path = outdir / f"{split}_calibration_fixed_bins.csv"
    adaptive_bins_csv_path = outdir / f"{split}_calibration_adaptive_bins.csv"

    # Backward-compatible fixed-bin filename.
    legacy_bins_csv_path = outdir / f"{split}_calibration_bins.csv"

    confthr_csv_path = outdir / f"{split}_confidence_threshold_coverage.csv"

    selective_curve_csv_path = outdir / f"{split}_risk_coverage_curve.csv"
    selective_summary_csv_path = outdir / f"{split}_selective_summary.csv"

    calibrated_metrics_csv_path = outdir / f"{split}_metrics_calibrated.csv"
    calibrated_fixed_bins_csv_path = outdir / f"{split}_calibration_fixed_bins_calibrated.csv"
    calibrated_adaptive_bins_csv_path = (
        outdir / f"{split}_calibration_adaptive_bins_calibrated.csv"
    )
    calibrated_legacy_bins_csv_path = outdir / f"{split}_calibration_bins_calibrated.csv"
    calibrated_confthr_csv_path = (
        outdir / f"{split}_confidence_threshold_coverage_calibrated.csv"
    )
    calibrated_selective_curve_csv_path = (
        outdir / f"{split}_risk_coverage_curve_calibrated.csv"
    )
    calibrated_selective_summary_csv_path = (
        outdir / f"{split}_selective_summary_calibrated.csv"
    )

    report = _to_builtin(report)

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    num_samples = int(report.get("num_samples", 0))

    _write_metrics_csv(
        metrics_csv_path,
        dict(report.get("metrics", {})),
        num_samples=num_samples,
    )

    calibration = report.get("calibration", {})
    if not isinstance(calibration, dict):
        calibration = {}

    fixed_bins = list(calibration.get("fixed_bins", calibration.get("bins", [])))
    adaptive_bins = list(calibration.get("adaptive_bins", []))

    _write_bins_csv(fixed_bins_csv_path, fixed_bins)
    _write_bins_csv(adaptive_bins_csv_path, adaptive_bins)
    _write_bins_csv(legacy_bins_csv_path, fixed_bins)

    confthr_rows = list(
        report.get("confidence_threshold_coverage", {}).get("rows", [])
    )
    _write_generic_rows_csv(confthr_csv_path, confthr_rows)

    selective_curve_rows = _flatten_selective_curves(report)
    selective_summary_rows = _flatten_selective_summary(report)

    _write_generic_rows_csv(selective_curve_csv_path, selective_curve_rows)
    _write_generic_rows_csv(selective_summary_csv_path, selective_summary_rows)

    saved_paths = {
        "json": str(report_json_path),
        "metrics_csv": str(metrics_csv_path),
        "bins_csv": str(legacy_bins_csv_path),
        "fixed_bins_csv": str(fixed_bins_csv_path),
        "adaptive_bins_csv": str(adaptive_bins_csv_path),
        "confidence_threshold_coverage_csv": str(confthr_csv_path),
        "risk_coverage_curve_csv": str(selective_curve_csv_path),
        "selective_summary_csv": str(selective_summary_csv_path),
    }

    metrics_calibrated = report.get("metrics_calibrated")
    calibration_calibrated = report.get("calibration_calibrated")
    confthr_calibrated = report.get("confidence_threshold_coverage_calibrated")
    selective_calibrated = report.get("selective_prediction_calibrated")

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
        cal_cal = calibration_calibrated if isinstance(calibration_calibrated, dict) else {}
        fixed_cal = list(cal_cal.get("fixed_bins", cal_cal.get("bins", [])))
        adaptive_cal = list(cal_cal.get("adaptive_bins", []))

        _write_bins_csv(calibrated_fixed_bins_csv_path, fixed_cal)
        _write_bins_csv(calibrated_adaptive_bins_csv_path, adaptive_cal)
        _write_bins_csv(calibrated_legacy_bins_csv_path, fixed_cal)

        saved_paths["bins_calibrated_csv"] = str(calibrated_legacy_bins_csv_path)
        saved_paths["fixed_bins_calibrated_csv"] = str(calibrated_fixed_bins_csv_path)
        saved_paths["adaptive_bins_calibrated_csv"] = str(
            calibrated_adaptive_bins_csv_path
        )

    if confthr_calibrated is not None:
        calibrated_confthr_rows = list(confthr_calibrated.get("rows", []))
        _write_generic_rows_csv(calibrated_confthr_csv_path, calibrated_confthr_rows)
        saved_paths["confidence_threshold_coverage_calibrated_csv"] = str(
            calibrated_confthr_csv_path
        )

    if selective_calibrated is not None:
        calibrated_tmp_report = {
            "selective_prediction": selective_calibrated,
        }
        calibrated_curve_rows = _flatten_selective_curves(calibrated_tmp_report)
        calibrated_summary_rows = _flatten_selective_summary(calibrated_tmp_report)

        _write_generic_rows_csv(
            calibrated_selective_curve_csv_path,
            calibrated_curve_rows,
        )
        _write_generic_rows_csv(
            calibrated_selective_summary_csv_path,
            calibrated_summary_rows,
        )

        saved_paths["risk_coverage_curve_calibrated_csv"] = str(
            calibrated_selective_curve_csv_path
        )
        saved_paths["selective_summary_calibrated_csv"] = str(
            calibrated_selective_summary_csv_path
        )

    return saved_paths
