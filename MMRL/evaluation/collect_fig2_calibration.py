from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ROOTS = {
    #"MMRL": Path("output_refactor/MMRL/FS"),
    "CrossModal": Path("output_refactor/ClipAdapters/CrossModal/FS"),
    "TR": Path("output_refactor/ClipAdapters/TR/FS"),
    "BayesAdapter": Path("output_refactor/ClipAdapters/BAYES_ADAPTER_无验证集训练/FS"),
    "DetBayesRTMMRL": Path("output_refactor/DetBayesRTMMRL_验证集找参数结果/FS"),
}


def to_float(x, default=0.0):
    if x is None or x == "":
        return default
    return float(x)


def read_bin_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "bin_index": int(float(row["bin_index"])),
                "range_left": to_float(row.get("range_left")),
                "range_right": to_float(row.get("range_right")),
                "count": to_float(row.get("count_mean")),
                "avg_confidence": to_float(row.get("avg_confidence_mean")),
                "avg_accuracy": to_float(row.get("avg_accuracy_mean")),
            })
    return rows


def compute_case_metrics(rows):
    """
    对单个 dataset-shot-seed 或 dataset-shot summary 文件计算指标。

    注意：
    ECE 必须在单个 case 内部先计算：
        ECE_case = sum_b q_b * |acc_b - conf_b|

    不能先跨 case 合并 acc/conf 再取绝对值。
    """
    total_count = sum(r["count"] for r in rows)

    if total_count <= 0:
        return {
            "total_count": 0.0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "ece": 0.0,
        }

    accuracy = sum(r["count"] * r["avg_accuracy"] for r in rows) / total_count
    avg_confidence = sum(r["count"] * r["avg_confidence"] for r in rows) / total_count

    ece = 0.0
    for r in rows:
        q_b = r["count"] / total_count
        gap_b = abs(r["avg_accuracy"] - r["avg_confidence"])
        ece += q_b * gap_b

    return {
        "total_count": total_count,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "ece": ece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output_refactor/fig2_calibration_corrected")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Optional substrings. If any substring appears in a file path, that file is skipped.",
    )
    args = parser.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    exclude_tokens = [x.lower() for x in args.exclude]

    case_rows = []
    bin_macro_stats = defaultdict(lambda: {
        "n_cases": 0,
        "sample_pct_sum": 0.0,
        "avg_confidence_sum": 0.0,
        "avg_accuracy_sum": 0.0,
        "gap_sum": 0.0,
        "ece_contrib_sum": 0.0,
        "range_left": None,
        "range_right": None,
    })

    for method, root in METHOD_ROOTS.items():
        files = sorted(root.rglob("test_calibration_fixed_bins_summary.csv"))

        if not files:
            print(f"[WARN] no bin summary files found for {method}: {root}")
            continue

        for path in files:
            path_str_lower = str(path).lower()
            if any(tok in path_str_lower for tok in exclude_tokens):
                continue

            rows = read_bin_summary(path)
            if not rows:
                continue

            case_metrics = compute_case_metrics(rows)
            total_count = case_metrics["total_count"]

            if total_count <= 0:
                continue

            case_id = str(path.relative_to(root))

            case_rows.append({
                "method": method,
                "case_id": case_id,
                "path": str(path),
                "total_count": total_count,
                "accuracy": case_metrics["accuracy"],
                "avg_confidence": case_metrics["avg_confidence"],
                "ece": case_metrics["ece"],
            })

            for r in rows:
                bin_index = r["bin_index"]
                q_b = r["count"] / total_count if total_count > 0 else 0.0
                sample_pct = 100.0 * q_b
                gap = abs(r["avg_accuracy"] - r["avg_confidence"])
                ece_contrib = q_b * gap

                key = (method, bin_index)
                stat = bin_macro_stats[key]

                stat["n_cases"] += 1
                stat["sample_pct_sum"] += sample_pct
                stat["avg_confidence_sum"] += r["avg_confidence"]
                stat["avg_accuracy_sum"] += r["avg_accuracy"]
                stat["gap_sum"] += gap
                stat["ece_contrib_sum"] += ece_contrib
                stat["range_left"] = r["range_left"]
                stat["range_right"] = r["range_right"]

    # 1. 输出每个 case 的指标
    case_metrics_path = outdir / "case_metrics.csv"
    with case_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "case_id",
            "path",
            "total_count",
            "accuracy",
            "avg_confidence",
            "ece",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    # 2. BA-style 方法汇总：对 case 等权平均
    method_group = defaultdict(list)
    for row in case_rows:
        method_group[row["method"]].append(row)

    summary_rows = []
    for method, rows in sorted(method_group.items()):
        n = len(rows)

        summary_rows.append({
            "method": method,
            "n_cases": n,
            "avg_accuracy": sum(r["accuracy"] for r in rows) / n if n else 0.0,
            "avg_confidence": sum(r["avg_confidence"] for r in rows) / n if n else 0.0,
            "ece": sum(r["ece"] for r in rows) / n if n else 0.0,
            "total_count_sum": sum(r["total_count"] for r in rows),
        })

    summary_path = outdir / "method_summary_macro.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "n_cases",
            "avg_accuracy",
            "avg_confidence",
            "ece",
            "total_count_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3. Figure 2 用的 macro calibration bins
    # 注意：
    # avg_confidence / avg_accuracy 是跨 case 的 bin 均值；
    # gap 是 mean(|acc-conf|)，不是 |mean(acc)-mean(conf)|。
    bin_rows = []
    for (method, bin_index), stat in sorted(bin_macro_stats.items()):
        n = stat["n_cases"]
        if n <= 0:
            continue

        avg_conf = stat["avg_confidence_sum"] / n
        avg_acc = stat["avg_accuracy_sum"] / n

        bin_rows.append({
            "method": method,
            "bin_index": bin_index,
            "range_left": stat["range_left"],
            "range_right": stat["range_right"],
            "sample_pct": stat["sample_pct_sum"] / n,
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc,

            # 用这个作为校准误差的 bin gap，避免抵消
            "gap": stat["gap_sum"] / n,

            # 这个只是画出来的两条均值曲线之间的距离，不用于 ECE
            "curve_gap": abs(avg_acc - avg_conf),

            # 每个 case 内 q_b * gap_b，再跨 case 平均
            "ece_contribution": stat["ece_contrib_sum"] / n,

            "n_cases": n,
        })

    bins_path = outdir / "fig2_calibration_bins_macro.csv"
    with bins_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "bin_index",
            "range_left",
            "range_right",
            "sample_pct",
            "avg_confidence",
            "avg_accuracy",
            "gap",
            "curve_gap",
            "ece_contribution",
            "n_cases",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bin_rows)

    # 4. 用 bin 的 ece_contribution 反查一次 ECE，方便核对
    ece_check = defaultdict(float)
    for row in bin_rows:
        ece_check[row["method"]] += row["ece_contribution"]

    ece_check_rows = []
    for method in sorted(ece_check):
        ece_check_rows.append({
            "method": method,
            "ece_from_bin_contributions": ece_check[method],
        })

    ece_check_path = outdir / "ece_check_from_macro_bins.csv"
    with ece_check_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "ece_from_bin_contributions"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ece_check_rows)

    print(f"[OK] saved {case_metrics_path}")
    print(f"[OK] saved {summary_path}")
    print(f"[OK] saved {bins_path}")
    print(f"[OK] saved {ece_check_path}")


if __name__ == "__main__":
    main()