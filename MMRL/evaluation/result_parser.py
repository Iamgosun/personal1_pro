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

    saved = {
        "json": json_path,
        "csv": csv_path,
        "bins_csv": bins_csv_path,
    }

    if calibrated_bin_rows:
        saved["bins_calibrated_csv"] = bins_calibrated_csv_path

    return saved


def aggregate_case(case_root: Path, metric_files: list, split: str):
    loaded = []
    seeds = []

    for metric_file in metric_files:
        report = _load_json(metric_file)
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

    summary = {
        "schema_version": 1,
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

        if "bins_calibrated_csv" in saved:
            print(f"  saved: {saved['bins_calibrated_csv']}")

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