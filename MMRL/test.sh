OUT=output_sweeps/vcrm_bayes_joint
mkdir -p "${OUT}/inspect"
cp "${OUT}/run_manifest.csv" "${OUT}/inspect/run_manifest.snapshot.csv"

cat > "${OUT}/inspect/select_bayes_best_acc_drop_0p2_ece.py" <<'PY'
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

OUT = Path("output_sweeps/vcrm_bayes_joint")
MANIFEST = OUT / "inspect" / "run_manifest.snapshot.csv"
BEST_CSV = OUT / "inspect" / "bayes_best_acc_drop_0p2_ece.csv"

# Accuracy is reported in percentage points, e.g. 96.91.
# Allow a 0.2 percentage-point drop from the best accuracy.
ACC_DROP = 0.2

rows = []
seen = set()

with MANIFEST.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("stage") != "tune":
            continue
        if row.get("method") != "BayesMMRL":
            continue
        if row.get("protocol") != "FS":
            continue
        if row.get("shot") != "16" or row.get("seed") != "1":
            continue

        outdir = row.get("outdir", "")
        if not outdir or outdir in seen:
            continue
        seen.add(outdir)

        metrics_path = Path(outdir) / "test_metrics.json"
        if not metrics_path.exists():
            continue

        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {}) or {}
            acc = float(metrics.get("accuracy", "nan"))
            ece = float(metrics.get("ece", "nan"))
        except Exception:
            continue

        if not math.isfinite(acc):
            continue
        if not math.isfinite(ece):
            ece = float("inf")

        rows.append({
            "method": row["method"],
            "dataset": row["dataset"],
            "tag": row["tag"],
            "accuracy": acc,
            "ece": ece,
            "outdir": outdir,
            "metrics_path": str(metrics_path),
        })

groups = defaultdict(list)
for r in rows:
    groups[r["dataset"]].append(r)

best_rows = []

print()
print("[BayesMMRL best: acc within 0.2 percentage points, then lowest ECE]")

for_dataset = sorted(groups.items())
for dataset, group in for_dataset:
    best_acc = max(r["accuracy"] for r in group)
    acc_threshold = best_acc - ACC_DROP

    candidates = [
        r for r in group
        if r["accuracy"] >= acc_threshold
    ]

    candidates = sorted(
        candidates,
        key=lambda r: (
            r["ece"],
            -r["accuracy"],
            r["tag"],
        ),
    )

    best = candidates[0]
    best_rows.append({
        **best,
        "best_acc": best_acc,
        "acc_threshold": acc_threshold,
        "acc_drop_allowed": ACC_DROP,
        "num_candidates": len(candidates),
        "num_finished": len(group),
    })

    print(
        f"{dataset:16s} "
        f"best_acc={best_acc:8.4f} "
        f"threshold={acc_threshold:8.4f} "
        f"chosen_acc={best['accuracy']:8.4f} "
        f"chosen_ece={best['ece']:8.4f} "
        f"candidates={len(candidates):3d}/{len(group):3d} "
        f"tag={best['tag']}"
    )

BEST_CSV.parent.mkdir(parents=True, exist_ok=True)
with BEST_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method", "dataset", "tag",
            "accuracy", "ece",
            "best_acc", "acc_threshold",
            "acc_drop_allowed",
            "num_candidates", "num_finished",
            "outdir", "metrics_path",
        ],
    )
    writer.writeheader()
    writer.writerows(best_rows)

print()
print(f"[completed Bayes tune rows] {len(rows)}")
print(f"[completed datasets] {len(best_rows)}")
print(f"[wrote] {BEST_CSV}")
PY

python "${OUT}/inspect/select_bayes_best_acc_drop_0p2_ece.py"