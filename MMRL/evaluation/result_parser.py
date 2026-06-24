from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

SEED_DIR_PATTERN = re.compile(r"seed(\d+)$")

OOD_METRIC_KEYS = ["TNR95", "AUROC", "DetAcc", "AUPR_In", "AUPR_Out"]
OOD_COUNT_KEYS = ["num_id", "num_ood"]


def _safe_mean(values: list[float]):
    return mean(values) if values else None


def _safe_std(values: list[float]):
    if not values:
        return None
    return pstdev(values) if len(values) > 1 else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_seed(report_path: Path):
    match = SEED_DIR_PATTERN.fullmatch(report_path.parent.name)
    if match:
        return int(match.group(1))
    return None


def _infer_case_root(report_path: Path) -> Path:
    if SEED_DIR_PATTERN.fullmatch(report_path.parent.name):
        return report_path.parent.parent
    return report_path.parent


def _discover_report_files(root: Path, split: str) -> list[Path]:
    """Discover canonical report files only.

    This intentionally does not read {split}_metrics.json. The canonical
    per-seed JSON report is {split}_report.json.
    """
    return sorted(root.rglob(f"{split}_report.json"))


def _to_float(value: Any):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None

# 因为TipA这东西为NAN，不能汇总
SKIP_METRIC_KEYS = {"nll", "brier"}

def _aggregate_scalar_block(
    reports: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    names: set[str] = set()
    for report in reports:
        block = report.get(key, {})
        if isinstance(block, dict):
            names.update(block.keys())

    out: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        if key in {"metrics", "metrics_calibrated"} and name in SKIP_METRIC_KEYS:
            continue

        values: list[float] = []
        for report in reports:
            block = report.get(key, {})
            if not isinstance(block, dict):
                continue
            value = _to_float(block.get(name))
            if value is not None:
                values.append(value)

        if values:
            out[name] = {
                "values": values,
                "mean": _safe_mean(values),
                "std": _safe_std(values),
            }

    return out




def _aggregate_rows(
    reports: list[dict[str, Any]],
    block_path: list[str],
    group_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for report in reports:
        block: Any = report
        for key in block_path:
            if not isinstance(block, dict):
                block = None
                break
            block = block.get(key)

        if not isinstance(block, list):
            continue

        for row in block:
            if not isinstance(row, dict):
                continue
            group = tuple(row.get(k) for k in group_keys)
            grouped.setdefault(group, []).append(row)

    aggregated: list[dict[str, Any]] = []

    for group, rows in sorted(grouped.items(), key=lambda x: str(x[0])):
        out = {key: value for key, value in zip(group_keys, group)}
        out["num_seeds"] = len(rows)

        numeric_keys: list[str] = []
        for row in rows:
            for key, value in row.items():
                if key in group_keys:
                    continue
                if _to_float(value) is not None and key not in numeric_keys:
                    numeric_keys.append(key)

        for key in numeric_keys:
            values = []
            for row in rows:
                value = _to_float(row.get(key))
                if value is not None:
                    values.append(value)
            out[f"{key}_mean"] = _safe_mean(values)
            out[f"{key}_std"] = _safe_std(values)

        aggregated.append(out)

    return aggregated


def _aggregate_calibration_bins(
    reports: list[dict[str, Any]],
    bin_key: str,
    calibration_key: str = "calibration",
) -> list[dict[str, Any]]:
    all_bins: list[list[dict[str, Any]]] = []

    for report in reports:
        calibration = report.get(calibration_key, {})
        if not isinstance(calibration, dict):
            continue
        rows = calibration.get(bin_key, [])
        if isinstance(rows, list) and rows:
            all_bins.append(rows)

    if not all_bins:
        return []

    n_bins = len(all_bins[0])
    numeric_keys = [
        "count",
        "fraction",
        "correct_count",
        "avg_confidence",
        "avg_accuracy",
        "gap",
        "weighted_gap",
    ]

    aggregated: list[dict[str, Any]] = []

    for idx in range(n_bins):
        first = all_bins[0][idx]
        row = {
            "bin_index": first.get("bin_index", idx),
            "bin_type": first.get("bin_type", bin_key),
            "range_left": first.get("range_left"),
            "range_right": first.get("range_right"),
            "num_seeds": 0,
        }

        for key in numeric_keys:
            values = []
            for bins in all_bins:
                if idx >= len(bins):
                    continue
                value = _to_float(bins[idx].get(key))
                if value is not None:
                    values.append(value)

            row[f"{key}_mean"] = _safe_mean(values)
            row[f"{key}_std"] = _safe_std(values)

        row["num_seeds"] = len(all_bins)
        aggregated.append(row)

    return aggregated


def _aggregate_selective_coverage_summary(
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    for report in reports:
        selective = report.get("selective_prediction", {})
        if not isinstance(selective, dict):
            continue

        coverage_summary = selective.get("coverage_summary", {})
        if not isinstance(coverage_summary, dict):
            continue

        for _score_name, rows in coverage_summary.items():
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue

                group = (
                    row.get("score_name"),
                    row.get("requested_coverage"),
                )
                grouped.setdefault(group, []).append(row)

    aggregated: list[dict[str, Any]] = []

    for group, rows in sorted(grouped.items(), key=lambda x: str(x[0])):
        out = {
            "score_name": group[0],
            "requested_coverage": group[1],
            "num_seeds": len(rows),
        }

        numeric_keys: list[str] = []
        for row in rows:
            for key, value in row.items():
                if key in {"score_name", "requested_coverage"}:
                    continue
                if _to_float(value) is not None and key not in numeric_keys:
                    numeric_keys.append(key)

        for key in numeric_keys:
            values = []
            for row in rows:
                value = _to_float(row.get(key))
                if value is not None:
                    values.append(value)
            out[f"{key}_mean"] = _safe_mean(values)
            out[f"{key}_std"] = _safe_std(values)

        aggregated.append(out)

    return aggregated


def _aggregate_selective_risk_curve(
    reports: list[dict[str, Any]],
    score_name: str = "least_confidence",
) -> list[dict[str, Any]]:
    return _aggregate_rows(
        reports,
        ["selective_prediction", "curves", score_name],
        ["score_name", "rank"],
    )


def _flatten_summary_row(case_root: Path, split: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_root": str(case_root),
        "split": split,
        "num_seeds": summary["num_seeds"],
        "seeds": " ".join(str(s) for s in summary["seeds"] if s is not None),
    }

    for metric, stats in summary.get("metrics", {}).items():
        row[f"{metric}_mean"] = stats.get("mean")
        row[f"{metric}_std"] = stats.get("std")

    for metric, stats in summary.get("metrics_calibrated", {}).items():
        row[f"{metric}_calibrated_mean"] = stats.get("mean")
        row[f"{metric}_calibrated_std"] = stats.get("std")

    temperature = summary.get("temperature", {})
    if isinstance(temperature, dict) and temperature.get("values"):
        row["temperature_mean"] = temperature.get("mean")
        row["temperature_std"] = temperature.get("std")

    return row


def _write_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    default_fields: list[str] | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow(default_fields or [])
            return

        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_files(case_root: Path, split: str, summary: dict[str, Any]) -> dict[str, str]:
    json_path = case_root / f"{split}_summary.json"
    csv_path = case_root / f"{split}_summary.csv"

    fixed_bins_csv = case_root / f"{split}_calibration_fixed_bins_summary.csv"
    adaptive_bins_csv = case_root / f"{split}_calibration_adaptive_bins_summary.csv"
    confthr_csv = case_root / f"{split}_confidence_threshold_coverage_summary.csv"

    selective_summary_csv = case_root / f"{split}_selective_summary_summary.csv"
    risk_curve_csv = case_root / f"{split}_risk_coverage_curve_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    _write_rows_csv(csv_path, [_flatten_summary_row(case_root, split, summary)])
    _write_rows_csv(fixed_bins_csv, summary.get("calibration", {}).get("fixed_bins", []))
    _write_rows_csv(adaptive_bins_csv, summary.get("calibration", {}).get("adaptive_bins", []))
    _write_rows_csv(
        confthr_csv,
        summary.get("confidence_threshold_coverage", {}).get("rows", []),
    )
    _write_rows_csv(
        selective_summary_csv,
        summary.get("selective_prediction", {}).get("coverage_summary", []),
    )
    _write_rows_csv(
        risk_curve_csv,
        summary.get("selective_prediction", {}).get("risk_coverage_curve", []),
    )

    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "fixed_bins_csv": str(fixed_bins_csv),
        "adaptive_bins_csv": str(adaptive_bins_csv),
        "confidence_threshold_coverage_csv": str(confthr_csv),
        "selective_summary_csv": str(selective_summary_csv),
        "risk_curve_csv": str(risk_curve_csv),
    }


def aggregate_case(
    case_root: Path,
    report_files: list[Path],
    split: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    reports = [_load_json(path) for path in report_files]
    seeds = [_infer_seed(path) for path in report_files]

    metrics = _aggregate_scalar_block(reports, "metrics")
    metrics_calibrated = _aggregate_scalar_block(reports, "metrics_calibrated")

    temperature_values = []
    for report in reports:
        temp_info = report.get("temperature_scaling", {})
        if isinstance(temp_info, dict):
            temp = _to_float(temp_info.get("temperature"))
            if temp is not None:
                temperature_values.append(temp)

    summary = {
        "schema_version": 3,
        "split": split,
        "case_root": str(case_root),
        "num_seeds": len(reports),
        "seeds": seeds,
        "metrics": metrics,
        "metrics_calibrated": metrics_calibrated,
        "temperature": {
            "values": temperature_values,
            "mean": _safe_mean(temperature_values),
            "std": _safe_std(temperature_values),
        },
        "calibration": {
            "fixed_bins": _aggregate_calibration_bins(reports, "fixed_bins"),
            "adaptive_bins": _aggregate_calibration_bins(reports, "adaptive_bins"),
        },
        "confidence_threshold_coverage": {
            "rows": _aggregate_rows(
                reports,
                ["confidence_threshold_coverage", "rows"],
                ["threshold"],
            )
        },
        "selective_prediction": {
            "coverage_summary": _aggregate_selective_coverage_summary(reports),
            "risk_coverage_curve": _aggregate_selective_risk_curve(
                reports,
                score_name="least_confidence",
            ),
        },
        "ood": {},
    }

    saved = _write_summary_files(case_root, split, summary)
    return summary, saved


def aggregate_directory(root_dir: str, split: str = "test") -> list[dict[str, Any]]:
    if split == "ood":
        return aggregate_ood_directory(root_dir)

    root = Path(root_dir)
    report_files = _discover_report_files(root, split)

    if not report_files:
        print(f"No {split}_report.json found under {root}")
        return []

    grouped: dict[Path, list[Path]] = {}
    for path in report_files:
        grouped.setdefault(_infer_case_root(path), []).append(path)

    global_rows: list[dict[str, Any]] = []

    for case_root in sorted(grouped.keys()):
        summary, saved = aggregate_case(case_root, grouped[case_root], split)
        row = _flatten_summary_row(case_root, split, summary)
        row["case_root"] = str(case_root.relative_to(root)) if case_root != root else "."
        global_rows.append(row)

        print(f"[OK] {case_root}")
        for path in saved.values():
            print(f"  saved: {path}")

    global_csv = root / f"aggregated_{split}_summary.csv"
    _write_rows_csv(global_csv, global_rows)
    print(f"[OK] saved global summary: {global_csv}")

    return global_rows


# ---------------------------------------------------------------------------
# OOD aggregation
# ---------------------------------------------------------------------------

def _discover_ood_result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("ood_eval/ood_results.csv"))


def _infer_ood_seed(path: Path):
    # Expected path:
    #   <case_root>/seed1/ood_eval/ood_results.csv
    if path.parent.name != "ood_eval":
        return None

    seed_dir = path.parent.parent
    match = SEED_DIR_PATTERN.fullmatch(seed_dir.name)
    if match:
        return int(match.group(1))

    return None


def _infer_ood_case_root(path: Path) -> Path:
    # Expected path:
    #   <case_root>/seed1/ood_eval/ood_results.csv
    if path.parent.name == "ood_eval":
        seed_dir = path.parent.parent
        if SEED_DIR_PATTERN.fullmatch(seed_dir.name):
            return seed_dir.parent

    return path.parent


def _load_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aggregate_ood_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    registry_values = [
        row.get("registry_dataset")
        for row in rows
        if row.get("registry_dataset") not in {None, ""}
    ]
    out["registry_dataset"] = registry_values[0] if registry_values else ""

    for key in OOD_COUNT_KEYS + OOD_METRIC_KEYS:
        values: list[float] = []
        for row in rows:
            value = _to_float(row.get(key))
            if value is not None:
                values.append(value)

        out[f"{key}_mean"] = _safe_mean(values)
        out[f"{key}_std"] = _safe_std(values)

    return out


def _flatten_ood_summary_rows(
    case_root: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for dataset, stats in summary.get("datasets", {}).items():
        row: dict[str, Any] = {
            "case_root": str(case_root),
            "split": "ood",
            "dataset": dataset,
            "registry_dataset": stats.get("registry_dataset", ""),
            "num_seeds": stats.get("num_seeds", summary.get("num_seeds")),
            "seeds": " ".join(
                str(seed)
                for seed in stats.get("seeds", summary.get("seeds", []))
                if seed is not None
            ),
        }

        for key in OOD_COUNT_KEYS + OOD_METRIC_KEYS:
            row[f"{key}_mean"] = stats.get(f"{key}_mean")
            row[f"{key}_std"] = stats.get(f"{key}_std")

        rows.append(row)

    return rows


def _write_ood_summary_files(
    case_root: Path,
    summary: dict[str, Any],
) -> dict[str, str]:
    json_path = case_root / "ood_summary.json"
    csv_path = case_root / "ood_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    _write_rows_csv(csv_path, _flatten_ood_summary_rows(case_root, summary))

    return {
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
    }


def aggregate_ood_case(
    case_root: Path,
    result_files: list[Path],
) -> tuple[dict[str, Any], dict[str, str]]:
    seed_to_rows: dict[int | None, list[dict[str, Any]]] = {}

    for path in result_files:
        seed = _infer_ood_seed(path)
        seed_to_rows[seed] = _load_csv_dicts(path)

    seeds = sorted(seed for seed in seed_to_rows.keys() if seed is not None)

    grouped_by_dataset: dict[str, list[tuple[int | None, dict[str, Any]]]] = {}

    for seed, rows in seed_to_rows.items():
        for row in rows:
            dataset = row.get("dataset")
            if not dataset:
                continue
            grouped_by_dataset.setdefault(dataset, []).append((seed, row))

    datasets: dict[str, Any] = {}

    for dataset, seed_rows in sorted(grouped_by_dataset.items()):
        rows = [row for _seed, row in seed_rows]
        dataset_seeds = sorted(
            seed
            for seed, _row in seed_rows
            if seed is not None
        )

        stats = _aggregate_ood_rows(rows)
        stats["num_seeds"] = len(set(dataset_seeds))
        stats["seeds"] = sorted(set(dataset_seeds))
        datasets[dataset] = stats

    summary = {
        "schema_version": 1,
        "split": "ood",
        "case_root": str(case_root),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "datasets": datasets,
    }

    saved = _write_ood_summary_files(case_root, summary)
    return summary, saved


def aggregate_ood_directory(root_dir: str) -> list[dict[str, Any]]:
    root = Path(root_dir)
    result_files = _discover_ood_result_files(root)

    if not result_files:
        print(f"No ood_eval/ood_results.csv found under {root}")
        return []

    grouped: dict[Path, list[Path]] = {}

    for path in result_files:
        grouped.setdefault(_infer_ood_case_root(path), []).append(path)

    global_rows: list[dict[str, Any]] = []

    for case_root in sorted(grouped.keys()):
        summary, saved = aggregate_ood_case(case_root, grouped[case_root])
        rows = _flatten_ood_summary_rows(case_root, summary)

        for row in rows:
            row["case_root"] = str(case_root.relative_to(root)) if case_root != root else "."

        global_rows.extend(rows)

        print(f"[OK] {case_root}")
        for path in saved.values():
            print(f"  saved: {path}")

    global_csv = root / "aggregated_ood_summary.csv"
    _write_rows_csv(global_csv, global_rows)
    print(f"[OK] saved global OOD summary: {global_csv}")

    return global_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    aggregate_directory(args.directory, split=args.split)


if __name__ == "__main__":
    main()