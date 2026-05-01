from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev


SEED_DIR_PATTERN = re.compile(r"seed(\d+)$")


def _safe_mean(values):
    return mean(values) if values else None


def _safe_std(values):
    if not values:
        return None
    return pstdev(values) if len(values) > 1 else 0.0


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_seed(metrics_path: Path):
    match = SEED_DIR_PATTERN.fullmatch(metrics_path.parent.name)
    if match:
        return int(match.group(1))
    return None


def _infer_case_root(metrics_path: Path):
    if SEED_DIR_PATTERN.fullmatch(metrics_path.parent.name):
        return metrics_path.parent.parent
    return metrics_path.parent


def _discover_metric_files(root: Path, split: str):
    return sorted(root.rglob(f"{split}_metrics.json"))


def _merge_nested_selective_metrics(report: dict) -> dict:
    """
    Backward/forward compatibility.

    Newer metrics.py flattens selective scalar metrics into report["metrics"].
    If an older report only has report["selective_prediction"]["metrics"],
    merge them here so summary CSV still reports AURC / AUROC.
    """
    metrics = report.setdefault("metrics", {})
    selective = report.get("selective_prediction", {})

    if not isinstance(metrics, dict) or not isinstance(selective, dict):
        return report

    selective_metrics = selective.get("metrics", {})
    if not isinstance(selective_metrics, dict):
        return report

    for key, value in selective_metrics.items():
        if key not in metrics and isinstance(value, (int, float)):
            metrics[key] = value

    return report


def _aggregate_scalar_metrics(reports, metrics_key: str = "metrics"):
    metric_names = set()

    for report in reports:
        metric_block = report.get(metrics_key, {})
        if isinstance(metric_block, dict):
            metric_names.update(metric_block.keys())

    summary = {}

    for metric_name in sorted(metric_names):
        values = []

        for report in reports:
            metric_block = report.get(metrics_key, {})
            if not isinstance(metric_block, dict):
                continue

            value = metric_block.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))

        if values:
            summary[metric_name] = {
                "values": values,
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }

    return summary


def _aggregate_bins(reports, calibration_key: str = "calibration"):
    all_bins = []

    for report in reports:
        calibration_block = report.get(calibration_key, {})
        if not isinstance(calibration_block, dict):
            continue

        bins = calibration_block.get("bins", [])
        if bins:
            all_bins.append(bins)

    if not all_bins:
        return []

    n_bins = len(all_bins[0])
    aggregated = []

    numeric_keys = [
        "count",
        "fraction",
        "correct_count",
        "avg_confidence",
        "avg_accuracy",
        "gap",
        "weighted_gap",
    ]

    for bin_idx in range(n_bins):
        first = all_bins[0][bin_idx]

        row = {
            "bin_index": first["bin_index"],
            "range_left": first["range_left"],
            "range_right": first["range_right"],
        }

        for key in numeric_keys:
            values = []

            for bins in all_bins:
                if bin_idx >= len(bins):
                    continue

                value = bins[bin_idx].get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))

            row[f"{key}_mean"] = _safe_mean(values)
            row[f"{key}_std"] = _safe_std(values)

        aggregated.append(row)

    return aggregated


def _aggregate_temperature(reports):
    values = []

    for report in reports:
        temp_info = report.get("temperature_scaling", {})
        if not isinstance(temp_info, dict):
            continue

        temperature = temp_info.get("temperature")
        if isinstance(temperature, (int, float)):
            values.append(float(temperature))

    return {
        "values": values,
        "mean": _safe_mean(values),
        "std": _safe_std(values),
    }


def _extract_selective_block(report: dict, key: str = "selective_prediction"):
    block = report.get(key, {})
    return block if isinstance(block, dict) else {}


def _aggregate_selective_scalar_metrics(
    reports,
    selective_key: str = "selective_prediction",
):
    metric_names = set()

    for report in reports:
        selective = _extract_selective_block(report, selective_key)
        metrics = selective.get("metrics", {})
        if isinstance(metrics, dict):
            metric_names.update(metrics.keys())

    summary = {}

    for metric_name in sorted(metric_names):
        values = []

        for report in reports:
            selective = _extract_selective_block(report, selective_key)
            metrics = selective.get("metrics", {})
            if not isinstance(metrics, dict):
                continue

            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))

        if values:
            summary[metric_name] = {
                "values": values,
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }

    return summary


def _aggregate_selective_summary(
    reports,
    selective_key: str = "selective_prediction",
):
    """
    Aggregate fixed coverage summary rows across seeds.

    Expected report structure:
        report["selective_prediction"]["coverage_summary"][score_name] = [
            {
                "score_name": "...",
                "requested_coverage": 0.9,
                "coverage": ...,
                "risk": ...,
                "selective_accuracy": ...,
                ...
            },
            ...
        ]
    """
    grouped = {}

    for report in reports:
        selective = _extract_selective_block(report, selective_key)
        coverage_summary = selective.get("coverage_summary", {})

        if not isinstance(coverage_summary, dict):
            continue

        for score_name, rows in coverage_summary.items():
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue

                requested_coverage = row.get("requested_coverage", row.get("coverage"))
                if not isinstance(requested_coverage, (int, float)):
                    continue

                group_key = (str(score_name), float(requested_coverage))
                grouped.setdefault(group_key, []).append(row)

    numeric_keys = [
        "coverage",
        "risk",
        "selective_accuracy",
        "num_kept",
        "num_total",
        "num_errors_kept",
        "threshold_uncertainty",
    ]

    aggregated = []

    for (score_name, requested_coverage), rows in sorted(grouped.items()):
        out = {
            "score_name": score_name,
            "requested_coverage": requested_coverage,
            "num_seeds": len(rows),
        }

        for key in numeric_keys:
            values = []

            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))

            out[f"{key}_mean"] = _safe_mean(values)
            out[f"{key}_std"] = _safe_std(values)

        aggregated.append(out)

    return aggregated


def _aggregate_risk_coverage_curve(
    reports,
    selective_key: str = "selective_prediction",
):
    """
    Aggregate full risk-coverage curves across seeds.

    Usually test set size is identical across seeds, so rank/coverage align.
    If a report has a different number of points, rows are matched by
    score_name + rank.
    """
    grouped = {}

    for report in reports:
        selective = _extract_selective_block(report, selective_key)
        curves = selective.get("curves", {})

        if not isinstance(curves, dict):
            continue

        for score_name, curve in curves.items():
            if not isinstance(curve, list):
                continue

            for row in curve:
                if not isinstance(row, dict):
                    continue

                rank = row.get("rank")
                if not isinstance(rank, (int, float)):
                    continue

                group_key = (str(score_name), int(rank))
                grouped.setdefault(group_key, []).append(row)

    numeric_keys = [
        "coverage",
        "risk",
        "selective_accuracy",
        "num_kept",
        "num_total",
        "num_errors_kept",
        "threshold_uncertainty",
    ]

    aggregated = []

    for (score_name, rank), rows in sorted(grouped.items()):
        out = {
            "score_name": score_name,
            "rank": rank,
            "num_seeds": len(rows),
        }

        for key in numeric_keys:
            values = []

            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))

            out[f"{key}_mean"] = _safe_mean(values)
            out[f"{key}_std"] = _safe_std(values)

        aggregated.append(out)

    return aggregated


def _write_bins_csv(path: Path, bin_rows: list):
    default_fields = [
        "bin_index",
        "range_left",
        "range_right",
        "count_mean",
        "count_std",
        "fraction_mean",
        "fraction_std",
        "correct_count_mean",
        "correct_count_std",
        "avg_confidence_mean",
        "avg_confidence_std",
        "avg_accuracy_mean",
        "avg_accuracy_std",
        "gap_mean",
        "gap_std",
        "weighted_gap_mean",
        "weighted_gap_std",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        if bin_rows:
            writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bin_rows)
        else:
            writer = csv.writer(f)
            writer.writerow(default_fields)


def _write_rows_csv(path: Path, rows: list, default_fields: list | None = None):
    with path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            fieldnames = []
            for row in rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(default_fields or [])


def _flatten_summary_row(case_root: Path, split: str, summary: dict):
    row = {
        "case_root": str(case_root),
        "split": split,
        "num_seeds": summary["num_seeds"],
        "seeds": " ".join(str(s) for s in summary["seeds"] if s is not None),
    }

    for metric_name, stats in summary.get("metrics", {}).items():
        row[f"{metric_name}_mean"] = stats["mean"]
        row[f"{metric_name}_std"] = stats["std"]

    for metric_name, stats in summary.get("metrics_calibrated", {}).items():
        row[f"{metric_name}_calibrated_mean"] = stats["mean"]
        row[f"{metric_name}_calibrated_std"] = stats["std"]

    # Also expose nested selective scalar metrics, even if they were not
    # flattened into report["metrics"] by an older metrics.py.

    temperature = summary.get("temperature", {})
    if isinstance(temperature, dict) and temperature.get("values"):
        row["temperature_mean"] = temperature["mean"]
        row["temperature_std"] = temperature["std"]

    return row


def _write_summary_files(case_root: Path, split: str, summary: dict):
    json_path = case_root / f"{split}_summary.json"
    csv_path = case_root / f"{split}_summary.csv"
    bins_csv_path = case_root / f"{split}_summary_bins.csv"
    bins_calibrated_csv_path = case_root / f"{split}_summary_bins_calibrated.csv"

    selective_summary_csv_path = case_root / f"{split}_summary_selective.csv"
    risk_curve_csv_path = case_root / f"{split}_summary_risk_coverage_curve.csv"

    selective_summary_calibrated_csv_path = (
        case_root / f"{split}_summary_selective_calibrated.csv"
    )
    risk_curve_calibrated_csv_path = (
        case_root / f"{split}_summary_risk_coverage_curve_calibrated.csv"
    )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat_row = _flatten_summary_row(case_root, split, summary)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    _write_bins_csv(
        bins_csv_path,
        summary.get("calibration", {}).get("bins", []),
    )

    calibrated_bin_rows = summary.get("calibration_calibrated", {}).get("bins", [])
    if calibrated_bin_rows:
        _write_bins_csv(bins_calibrated_csv_path, calibrated_bin_rows)

    selective_rows = summary.get("selective_prediction", {}).get(
        "coverage_summary",
        [],
    )
    risk_curve_rows = summary.get("selective_prediction", {}).get(
        "risk_coverage_curve",
        [],
    )

    if selective_rows:
        _write_rows_csv(selective_summary_csv_path, selective_rows)

    if risk_curve_rows:
        _write_rows_csv(risk_curve_csv_path, risk_curve_rows)

    selective_calibrated_rows = summary.get(
        "selective_prediction_calibrated",
        {},
    ).get("coverage_summary", [])

    risk_curve_calibrated_rows = summary.get(
        "selective_prediction_calibrated",
        {},
    ).get("risk_coverage_curve", [])

    if selective_calibrated_rows:
        _write_rows_csv(
            selective_summary_calibrated_csv_path,
            selective_calibrated_rows,
        )

    if risk_curve_calibrated_rows:
        _write_rows_csv(
            risk_curve_calibrated_csv_path,
            risk_curve_calibrated_rows,
        )

    saved = {
        "json": json_path,
        "csv": csv_path,
        "bins_csv": bins_csv_path,
    }

    if calibrated_bin_rows:
        saved["bins_calibrated_csv"] = bins_calibrated_csv_path

    if selective_rows:
        saved["selective_summary_csv"] = selective_summary_csv_path

    if risk_curve_rows:
        saved["risk_coverage_curve_csv"] = risk_curve_csv_path

    if selective_calibrated_rows:
        saved["selective_summary_calibrated_csv"] = (
            selective_summary_calibrated_csv_path
        )

    if risk_curve_calibrated_rows:
        saved["risk_coverage_curve_calibrated_csv"] = (
            risk_curve_calibrated_csv_path
        )

    return saved


def aggregate_case(case_root: Path, metric_files: list, split: str):
    loaded = []
    seeds = []

    for metric_file in metric_files:
        report = _load_json(metric_file)
        report = _merge_nested_selective_metrics(report)
        loaded.append(report)
        seeds.append(_infer_seed(metric_file))

    metrics_summary = _aggregate_scalar_metrics(
        loaded,
        metrics_key="metrics",
    )
    bins_summary = _aggregate_bins(
        loaded,
        calibration_key="calibration",
    )

    metrics_calibrated_summary = _aggregate_scalar_metrics(
        loaded,
        metrics_key="metrics_calibrated",
    )
    bins_calibrated_summary = _aggregate_bins(
        loaded,
        calibration_key="calibration_calibrated",
    )

    temperature_summary = _aggregate_temperature(loaded)

    selective_metrics_summary = _aggregate_selective_scalar_metrics(
        loaded,
        selective_key="selective_prediction",
    )
    selective_coverage_summary = _aggregate_selective_summary(
        loaded,
        selective_key="selective_prediction",
    )
    risk_coverage_curve_summary = _aggregate_risk_coverage_curve(
        loaded,
        selective_key="selective_prediction",
    )

    selective_metrics_calibrated_summary = _aggregate_selective_scalar_metrics(
        loaded,
        selective_key="selective_prediction_calibrated",
    )
    selective_coverage_calibrated_summary = _aggregate_selective_summary(
        loaded,
        selective_key="selective_prediction_calibrated",
    )
    risk_coverage_curve_calibrated_summary = _aggregate_risk_coverage_curve(
        loaded,
        selective_key="selective_prediction_calibrated",
    )

    summary = {
        "schema_version": 2,
        "split": split,
        "case_root": str(case_root),
        "num_seeds": len(loaded),
        "seeds": seeds,
        "metrics": metrics_summary,
        "metrics_calibrated": metrics_calibrated_summary,
        "temperature": temperature_summary,
        "calibration": {
            "n_bins": (
                loaded[0].get("calibration", {}).get("n_bins", 0)
                if loaded
                else 0
            ),
            "bins": bins_summary,
        },
        "calibration_calibrated": {
            "n_bins": (
                loaded[0].get("calibration_calibrated", {}).get("n_bins", 0)
                if loaded
                else 0
            ),
            "bins": bins_calibrated_summary,
        },
        "selective_prediction": {
            "metrics": selective_metrics_summary,
            "coverage_summary": selective_coverage_summary,
            "risk_coverage_curve": risk_coverage_curve_summary,
        },
        "selective_prediction_calibrated": {
            "metrics": selective_metrics_calibrated_summary,
            "coverage_summary": selective_coverage_calibrated_summary,
            "risk_coverage_curve": risk_coverage_curve_calibrated_summary,
        },
        "ood": {},
    }

    saved = _write_summary_files(case_root, split, summary)
    return summary, saved


def _write_global_summary(root: Path, split: str, global_rows: list):
    global_csv = root / f"aggregated_{split}_summary.csv"

    if not global_rows:
        return None

    fieldnames = []
    for row in global_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with global_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(global_rows)

    return global_csv


def aggregate_directory(root_dir: str, split: str = "test"):
    root = Path(root_dir)
    metric_files = _discover_metric_files(root, split=split)

    if not metric_files:
        print(f"No {split}_metrics.json found under {root}")
        return []

    grouped = {}

    for metric_file in metric_files:
        case_root = _infer_case_root(metric_file)
        grouped.setdefault(case_root, []).append(metric_file)

    global_rows = []

    for case_root in sorted(grouped.keys()):
        summary, saved = aggregate_case(case_root, grouped[case_root], split=split)

        row = _flatten_summary_row(case_root, split, summary)
        row["case_root"] = (
            str(case_root.relative_to(root))
            if case_root != root
            else "."
        )

        global_rows.append(row)

        print(f"[OK] {case_root}")
        print(f"  saved: {saved['json']}")
        print(f"  saved: {saved['csv']}")
        print(f"  saved: {saved['bins_csv']}")

        for key in [
            "bins_calibrated_csv",
            "selective_summary_csv",
            "risk_coverage_curve_csv",
            "selective_summary_calibrated_csv",
            "risk_coverage_curve_calibrated_csv",
        ]:
            if key in saved:
                print(f"  saved: {saved[key]}")

    global_csv = _write_global_summary(root, split, global_rows)

    if global_csv is not None:
        print(f"[OK] saved global summary: {global_csv}")

    return global_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    aggregate_directory(args.directory, split=args.split)


if __name__ == "__main__":
    main()