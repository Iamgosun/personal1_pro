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


def _aggregate_scalar_metrics(reports):
    metric_names = set()
    for report in reports:
        metric_names.update(report.get("metrics", {}).keys())

    summary = {}
    for metric_name in sorted(metric_names):
        values = []
        for report in reports:
            value = report.get("metrics", {}).get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))

        if values:
            summary[metric_name] = {
                "values": values,
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }

    return summary


def _aggregate_bins(reports):
    all_bins = [report.get("calibration", {}).get("bins", []) for report in reports]
    if not all_bins or not all_bins[0]:
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
                if bin_idx < len(bins):
                    value = bins[bin_idx].get(key)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
            row[f"{key}_mean"] = _safe_mean(values)
            row[f"{key}_std"] = _safe_std(values)

        aggregated.append(row)

    return aggregated


def _write_summary_files(case_root: Path, split: str, summary: dict, bin_rows: list):
    json_path = case_root / f"{split}_summary.json"
    csv_path = case_root / f"{split}_summary.csv"
    bins_csv_path = case_root / f"{split}_summary_bins.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    flat_row = {
        "case_root": str(case_root),
        "split": split,
        "num_seeds": summary["num_seeds"],
        "seeds": " ".join(str(s) for s in summary["seeds"] if s is not None),
    }

    for metric_name, stats in summary["metrics"].items():
        flat_row[f"{metric_name}_mean"] = stats["mean"]
        flat_row[f"{metric_name}_std"] = stats["std"]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    if bin_rows:
        with bins_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bin_rows)
    else:
        with bins_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
            )

    return {
        "json": json_path,
        "csv": csv_path,
        "bins_csv": bins_csv_path,
    }


def aggregate_case(case_root: Path, metric_files: list, split: str):
    loaded = []
    seeds = []

    for metric_file in metric_files:
        report = _load_json(metric_file)
        loaded.append(report)
        seeds.append(_infer_seed(metric_file))

    metrics_summary = _aggregate_scalar_metrics(loaded)
    bins_summary = _aggregate_bins(loaded)

    summary = {
        "schema_version": 1,
        "split": split,
        "case_root": str(case_root),
        "num_seeds": len(loaded),
        "seeds": seeds,
        "metrics": metrics_summary,
        "calibration": {
            "n_bins": loaded[0].get("calibration", {}).get("n_bins", 0) if loaded else 0,
            "bins": bins_summary,
        },
        "ood": {},
    }

    saved = _write_summary_files(case_root, split, summary, bins_summary)
    return summary, saved


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

        row = {
            "case_root": str(case_root.relative_to(root)) if case_root != root else ".",
            "split": split,
            "num_seeds": summary["num_seeds"],
            "seeds": " ".join(str(s) for s in summary["seeds"] if s is not None),
        }

        for metric_name, stats in summary["metrics"].items():
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]

        global_rows.append(row)

        print(f"[OK] {case_root}")
        print(f"  saved: {saved['json']}")
        print(f"  saved: {saved['csv']}")
        print(f"  saved: {saved['bins_csv']}")

    global_csv = root / f"aggregated_{split}_summary.csv"
    if global_rows:
        fieldnames = list(global_rows[0].keys())
        with global_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(global_rows)
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