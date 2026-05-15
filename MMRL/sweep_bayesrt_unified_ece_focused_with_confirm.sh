#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Focused BayesRTMMRL unified-hyperparameter sweep for ECE.
#
# Purpose:
#   Run a focused BayesRTMMRL sweep designed to find ONE shared
#   hyperparameter tag across all datasets, prioritizing macro mean ECE.
#
# This is a wrapper around sweep_vcrm_bayes_baselines_newrules.sh.
# It uses that script's KEY=VALUE override interface, disables all
# non-BayesRT methods, and DOES run FS confirm + B2N confirm using the
# same selected unified BayesRT tag for all datasets.
#
# Flow:
#   1) Tune BayesRTMMRL only on FS 16-shot seed1.
#   2) Select ONE unified BayesRT tag across all tune datasets.
#   3) Patch a temporary copy of the base launcher so confirm/B2N use
#      that unified tag instead of per-dataset best tags.
#   4) Run FS confirm and B2N confirm.
#
# Outputs:
#   - ${OUTPUT_ROOT}/unified_bayesrt_config.csv
#   - ${OUTPUT_ROOT}/unified_bayesrt_config.env
#   - ${OUTPUT_ROOT}/unified_bayesrt_top20.csv
#   - normal confirm/B2N summaries from the base script
#
# Example:
#   bash sweep_bayesrt_unified_ece_focused_with_confirm.sh \
#     PROJECT_DIR="$PWD" \
#     DATA_ROOT=DATASETS \
#     OUTPUT_ROOT=output_sweeps/bayesrt_unified_ece_focused \
#     RESET_MANIFEST=1 SKIP_EXISTING=1 \
#     GPU_IDS="0 1 2 3 4 5" JOBS_PER_GPU=2
#
# To skip confirm/B2N and only do tune + unified selection:
#   RUN_CONFIRM=0 bash sweep_bayesrt_unified_ece_focused_with_confirm.sh ...
# ============================================================

apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      printf -v "${key}" '%s' "${val}"
      export "${key}"
    else
      echo "[warn] non KEY=VALUE ignored: ${arg}" >&2
    fi
  done
}

apply_kv_args "$@"

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
BASE_SCRIPT=${BASE_SCRIPT:-${PROJECT_DIR}/sweep_vcrm_bayes_baselines_newrules.sh}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayesrt_unified_ece_focused}
DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft oxford_pets stanford_cars ucf101"}
TUNE_DATASETS=${TUNE_DATASETS:-${DATASETS}}
TUNE_SHOTS=${TUNE_SHOTS:-"16"}
TUNE_SEEDS=${TUNE_SEEDS:-"1"}

CONFIRM_DATASETS=${CONFIRM_DATASETS:-${DATASETS}}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-"1 2 4 8 16 32"}
B2N_SHOTS=${B2N_SHOTS:-"16"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"1 2 3"}
RUN_CONFIRM=${RUN_CONFIRM:-1}

# Selection constraints for ONE shared tag.
# The selector first filters by these accuracy-drop constraints, then chooses
# the lowest macro mean ECE. If no tag satisfies constraints, it falls back to
# the global lowest macro mean ECE.
UNIFIED_MAX_MEAN_ACC_DROP=${UNIFIED_MAX_MEAN_ACC_DROP:-1.2}
UNIFIED_MAX_DATASET_ACC_DROP=${UNIFIED_MAX_DATASET_ACC_DROP:-2.5}
TARGET_MEAN_ECE=${TARGET_MEAN_ECE:-2.0}
UNIFIED_TOPK=${UNIFIED_TOPK:-20}

# Focused BayesRT grid.
BAYESRT_R_PRIOR_STD_LIST=${BAYESRT_R_PRIOR_STD_LIST:-"0.03 0.05 0.07 0.1 0.15"}
BAYESRT_R_KL_WEIGHT_LIST=${BAYESRT_R_KL_WEIGHT_LIST:-"5e-6 1e-5 1e-4"}
BAYESRT_T_PRIOR_STD_LIST=${BAYESRT_T_PRIOR_STD_LIST:-"0.001 0.002 0.003 0.005"}
BAYESRT_T_KL_WEIGHT_LIST=${BAYESRT_T_KL_WEIGHT_LIST:-"1e-4 1e-2 5e-2"}
BAYESRT_EVAL_FUSION_VARIANT_LIST=${BAYESRT_EVAL_FUSION_VARIANT_LIST:-"static_prob"}

# Optional wider mode. Use EXTENDED=1 if the first focused sweep still cannot
# reach the desired macro ECE.
if [[ "${EXTENDED:-0}" == "1" ]]; then
  BAYESRT_R_PRIOR_STD_LIST=${BAYESRT_R_PRIOR_STD_LIST_EXT:-"0.03 0.05 0.07 0.1 0.15 0.2 0.3"}
  BAYESRT_R_KL_WEIGHT_LIST=${BAYESRT_R_KL_WEIGHT_LIST_EXT:-"1e-6 5e-6 1e-5 5e-5 1e-4"}
  BAYESRT_T_PRIOR_STD_LIST=${BAYESRT_T_PRIOR_STD_LIST_EXT:-"0.001 0.002 0.003 0.005"}
  BAYESRT_T_KL_WEIGHT_LIST=${BAYESRT_T_KL_WEIGHT_LIST_EXT:-"1e-4 1e-3 1e-2 5e-2"}
  BAYESRT_EVAL_FUSION_VARIANT_LIST=${BAYESRT_EVAL_FUSION_VARIANT_LIST_EXT:-"static_prob"}
fi

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[error] BASE_SCRIPT not found: ${BASE_SCRIPT}" >&2
  echo "        Set BASE_SCRIPT=/path/to/sweep_vcrm_bayes_baselines_newrules.sh" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[config] PROJECT_DIR=${PROJECT_DIR}"
echo "[config] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] TUNE_DATASETS=${TUNE_DATASETS}"
echo "[config] CONFIRM_DATASETS=${CONFIRM_DATASETS}"
echo "[config] CONFIRM_SHOTS=${CONFIRM_SHOTS}"
echo "[config] B2N_SHOTS=${B2N_SHOTS}"
echo "[config] CONFIRM_SEEDS=${CONFIRM_SEEDS}"
echo "[config] RUN_CONFIRM=${RUN_CONFIRM}"
echo "[config] BayesRT cases per dataset = $(python - <<PY
r='${BAYESRT_R_PRIOR_STD_LIST}'.split()
rkl='${BAYESRT_R_KL_WEIGHT_LIST}'.split()
t='${BAYESRT_T_PRIOR_STD_LIST}'.split()
tkl='${BAYESRT_T_KL_WEIGHT_LIST}'.split()
f='${BAYESRT_EVAL_FUSION_VARIANT_LIST}'.split()
print(len(r)*len(rkl)*len(t)*len(tkl)*len(f))
PY
)"
echo "[config] target macro mean ECE < ${TARGET_MEAN_ECE}"
echo "[config] selection constraints: mean_acc_drop<=${UNIFIED_MAX_MEAN_ACC_DROP}, max_acc_drop<=${UNIFIED_MAX_DATASET_ACC_DROP}"

# Phase 1: focused tuning. Confirm is disabled here because the unified tag
# must be selected before confirm/B2N can be forced to use it.
bash "${BASE_SCRIPT}" \
  PROJECT_DIR="${PROJECT_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  DATASETS="${DATASETS}" \
  TUNE_DATASETS="${TUNE_DATASETS}" \
  TUNE_SHOTS="${TUNE_SHOTS}" \
  TUNE_SEEDS="${TUNE_SEEDS}" \
  CONFIRM_DATASETS="${CONFIRM_DATASETS}" \
  CONFIRM_SHOTS="${CONFIRM_SHOTS}" \
  B2N_SHOTS="${B2N_SHOTS}" \
  CONFIRM_SEEDS="${CONFIRM_SEEDS}" \
  RUN_BAYESRT=1 \
  RUN_BAYES=0 \
  RUN_MNDL=0 \
  RUN_VCRM=0 \
  RUN_MMRL=0 \
  RUN_BAYES_ADAPTER=0 \
  AUTO_TUNE="${AUTO_TUNE:-1}" \
  AUTO_CONFIRM_FS=0 \
  AUTO_CONFIRM_B2N=0 \
  RESET_MANIFEST="${RESET_MANIFEST:-0}" \
  SKIP_EXISTING="${SKIP_EXISTING:-1}" \
  GPU_IDS="${GPU_IDS:-}" \
  NGPU="${NGPU:-}" \
  JOBS_PER_GPU="${JOBS_PER_GPU:-2}" \
  DATA_ROOT="${DATA_ROOT:-${ROOT:-DATASETS}}" \
  BACKBONE="${BACKBONE:-ViT-B/16}" \
  EXEC_MODE="${EXEC_MODE:-online}" \
  DELETE_CKPT_AFTER_TEST="${DELETE_CKPT_AFTER_TEST:-1}" \
  SUMMARY_ONLY="${SUMMARY_ONLY:-0}" \
  BAYESRT_R_PRIOR_STD_LIST="${BAYESRT_R_PRIOR_STD_LIST}" \
  BAYESRT_R_KL_WEIGHT_LIST="${BAYESRT_R_KL_WEIGHT_LIST}" \
  BAYESRT_T_PRIOR_STD_LIST="${BAYESRT_T_PRIOR_STD_LIST}" \
  BAYESRT_T_KL_WEIGHT_LIST="${BAYESRT_T_KL_WEIGHT_LIST}" \
  BAYESRT_EVAL_FUSION_VARIANT_LIST="${BAYESRT_EVAL_FUSION_VARIANT_LIST}" \
  ACC_DROP="${ACC_DROP:-0.8}"

# Select one shared BayesRT tag from tune_summary.csv.
python - <<PY
import csv
import math
from collections import defaultdict
from pathlib import Path

output_root = Path(r"${OUTPUT_ROOT}")
summary_path = output_root / "tune_summary.csv"
selected_csv = output_root / "unified_bayesrt_config.csv"
top_csv = output_root / "unified_bayesrt_top${UNIFIED_TOPK}.csv"
env_path = output_root / "unified_bayesrt_config.env"

max_mean_drop = float("${UNIFIED_MAX_MEAN_ACC_DROP}")
max_dataset_drop = float("${UNIFIED_MAX_DATASET_ACC_DROP}")
target_ece = float("${TARGET_MEAN_ECE}")
topk = int("${UNIFIED_TOPK}")

if not summary_path.exists():
    raise SystemExit(f"[error] missing {summary_path}")

rows = []
with summary_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("stage") != "tune":
            continue
        if row.get("method") != "BayesRTMMRL":
            continue
        if row.get("protocol") != "FS":
            continue
        if row.get("shot") != "16" or row.get("seed") != "1":
            continue
        if row.get("status") != "ok":
            continue
        try:
            row["accuracy"] = float(row["accuracy"])
            row["ece"] = float(row["ece"])
            row["nll"] = float(row["nll"])
            row["brier"] = float(row["brier"])
        except Exception:
            continue
        try:
            row["num_samples"] = int(float(row.get("num_samples") or 0))
        except Exception:
            row["num_samples"] = 0
        rows.append(row)

if not rows:
    raise SystemExit(f"[error] no valid BayesRTMMRL tune rows found in {summary_path}")

best_acc_by_dataset = {}
for r in rows:
    ds = r["dataset"]
    best_acc_by_dataset[ds] = max(best_acc_by_dataset.get(ds, -math.inf), r["accuracy"])

by_tag = defaultdict(list)
for r in rows:
    by_tag[r["tag"]].append(r)

all_datasets = sorted(best_acc_by_dataset)
needed = set(all_datasets)
records = []
for tag, group in by_tag.items():
    present = {r["dataset"] for r in group}
    if present != needed:
        continue
    n = len(group)
    total_samples = sum(r["num_samples"] for r in group)
    eces = [r["ece"] for r in group]
    accs = [r["accuracy"] for r in group]
    nlls = [r["nll"] for r in group]
    briers = [r["brier"] for r in group]
    drops = [best_acc_by_dataset[r["dataset"]] - r["accuracy"] for r in group]
    weighted_ece = sum(r["ece"] * r["num_samples"] for r in group) / total_samples if total_samples else float("nan")
    weighted_acc = sum(r["accuracy"] * r["num_samples"] for r in group) / total_samples if total_samples else float("nan")
    records.append({
        "tag": tag,
        "num_datasets": n,
        "mean_ece": sum(eces) / n,
        "max_ece": max(eces),
        "weighted_ece": weighted_ece,
        "mean_accuracy": sum(accs) / n,
        "weighted_accuracy": weighted_acc,
        "mean_acc_drop": sum(drops) / n,
        "max_acc_drop": max(drops),
        "mean_nll": sum(nlls) / n,
        "mean_brier": sum(briers) / n,
        "below_target_mean_ece": sum(eces) / n < target_ece,
        "satisfies_acc_constraints": (sum(drops) / n <= max_mean_drop and max(drops) <= max_dataset_drop),
    })

if not records:
    raise SystemExit("[error] no tag has complete coverage over all datasets")

records.sort(key=lambda r: (r["mean_ece"], r["max_ece"], r["mean_acc_drop"], r["tag"]))
constrained = [r for r in records if r["satisfies_acc_constraints"]]
selected = (constrained or records)[0]
selection_mode = "constrained_lowest_mean_ece" if constrained else "global_lowest_mean_ece_fallback"

fieldnames = [
    "selection_mode", "tag", "num_datasets", "mean_ece", "max_ece", "weighted_ece",
    "mean_accuracy", "weighted_accuracy", "mean_acc_drop", "max_acc_drop",
    "mean_nll", "mean_brier", "below_target_mean_ece", "satisfies_acc_constraints",
]
selected_csv.parent.mkdir(parents=True, exist_ok=True)
with selected_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow({**selected, "selection_mode": selection_mode})

with top_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in records[:topk]:
        w.writerow({**r, "selection_mode": "ranked_by_macro_mean_ece"})

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

env_lines = [
    f"UNIFIED_BAYESRT_TAG={shq(selected['tag'])}",
    f"UNIFIED_BAYESRT_MEAN_ECE={selected['mean_ece']:.6f}",
    f"UNIFIED_BAYESRT_MAX_ECE={selected['max_ece']:.6f}",
    f"UNIFIED_BAYESRT_WEIGHTED_ECE={selected['weighted_ece']:.6f}",
    f"UNIFIED_BAYESRT_MEAN_ACC_DROP={selected['mean_acc_drop']:.6f}",
    f"UNIFIED_BAYESRT_MAX_ACC_DROP={selected['max_acc_drop']:.6f}",
]
env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

print("[unified] selection_mode=", selection_mode)
print("[unified] tag=", selected["tag"])
print("[unified] mean_ece=", f"{selected['mean_ece']:.6f}")
print("[unified] max_ece=", f"{selected['max_ece']:.6f}")
print("[unified] weighted_ece=", f"{selected['weighted_ece']:.6f}")
print("[unified] mean_acc_drop=", f"{selected['mean_acc_drop']:.6f}")
print("[unified] max_acc_drop=", f"{selected['max_acc_drop']:.6f}")
print("[unified] tags_below_target_mean_ece=", sum(1 for r in records if r["below_target_mean_ece"]))
print("[unified] wrote", selected_csv)
print("[unified] wrote", env_path)
print("[unified] wrote", top_csv)
PY

# Phase 2: run FS confirm and B2N confirm with the same unified BayesRT tag.
if [[ "${RUN_CONFIRM}" == "1" && "${SUMMARY_ONLY:-0}" != "1" ]]; then
  UNIFIED_ENV="${OUTPUT_ROOT}/unified_bayesrt_config.env"
  if [[ ! -f "${UNIFIED_ENV}" ]]; then
    echo "[error] missing ${UNIFIED_ENV}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${UNIFIED_ENV}"

  if [[ -z "${UNIFIED_BAYESRT_TAG:-}" ]]; then
    echo "[error] UNIFIED_BAYESRT_TAG is empty in ${UNIFIED_ENV}" >&2
    exit 1
  fi

  PATCHED_BASE="${OUTPUT_ROOT}/.sweep_vcrm_bayes_baselines_newrules_force_unified_bayesrt.sh"
  python - <<PY
from pathlib import Path
src = Path(r"${BASE_SCRIPT}")
dst = Path(r"${PATCHED_BASE}")
text = src.read_text(encoding="utf-8")
old = '''require_best_tag() {
  local method=$1 dataset=$2 tag
  tag="$(get_best_tag "${method}" "${dataset}")"
  if [[ -z "${tag}" ]]; then
    echo "[error] no best tag found for method=${method} dataset=${dataset}. Check ${BEST_SUMMARY}" >&2
    exit 1
  fi
  echo "${tag}"
}
'''
new = '''require_best_tag() {
  local method=$1 dataset=$2 tag
  if [[ "${method}" == "BayesRTMMRL" && -n "${FORCE_UNIFIED_BAYESRT_TAG:-}" ]]; then
    echo "${FORCE_UNIFIED_BAYESRT_TAG}"
    return 0
  fi
  tag="$(get_best_tag "${method}" "${dataset}")"
  if [[ -z "${tag}" ]]; then
    echo "[error] no best tag found for method=${method} dataset=${dataset}. Check ${BEST_SUMMARY}" >&2
    exit 1
  fi
  echo "${tag}"
}
'''
if old not in text:
    raise SystemExit("[error] could not patch require_best_tag() in base script")
dst.write_text(text.replace(old, new), encoding="utf-8")
dst.chmod(0o755)
print(f"[patch] wrote {dst}")
PY

  echo "[confirm] forcing BayesRTMMRL unified tag: ${UNIFIED_BAYESRT_TAG}"
  FORCE_UNIFIED_BAYESRT_TAG="${UNIFIED_BAYESRT_TAG}" bash "${PATCHED_BASE}" \
    PROJECT_DIR="${PROJECT_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATASETS="${DATASETS}" \
    TUNE_DATASETS="${TUNE_DATASETS}" \
    TUNE_SHOTS="${TUNE_SHOTS}" \
    TUNE_SEEDS="${TUNE_SEEDS}" \
    CONFIRM_DATASETS="${CONFIRM_DATASETS}" \
    CONFIRM_SHOTS="${CONFIRM_SHOTS}" \
    B2N_SHOTS="${B2N_SHOTS}" \
    CONFIRM_SEEDS="${CONFIRM_SEEDS}" \
    RUN_BAYESRT=1 \
    RUN_BAYES=0 \
    RUN_MNDL=0 \
    RUN_VCRM=0 \
    RUN_MMRL=0 \
    RUN_BAYES_ADAPTER=0 \
    AUTO_TUNE=0 \
    AUTO_CONFIRM_FS="${AUTO_CONFIRM_FS:-1}" \
    AUTO_CONFIRM_B2N="${AUTO_CONFIRM_B2N:-1}" \
    RESET_MANIFEST=0 \
    SKIP_EXISTING="${SKIP_EXISTING:-1}" \
    GPU_IDS="${GPU_IDS:-}" \
    NGPU="${NGPU:-}" \
    JOBS_PER_GPU="${JOBS_PER_GPU:-2}" \
    DATA_ROOT="${DATA_ROOT:-${ROOT:-DATASETS}}" \
    BACKBONE="${BACKBONE:-ViT-B/16}" \
    EXEC_MODE="${EXEC_MODE:-online}" \
    DELETE_CKPT_AFTER_TEST="${DELETE_CKPT_AFTER_TEST:-1}" \
    SUMMARY_ONLY=0 \
    BAYESRT_R_PRIOR_STD_LIST="${BAYESRT_R_PRIOR_STD_LIST}" \
    BAYESRT_R_KL_WEIGHT_LIST="${BAYESRT_R_KL_WEIGHT_LIST}" \
    BAYESRT_T_PRIOR_STD_LIST="${BAYESRT_T_PRIOR_STD_LIST}" \
    BAYESRT_T_KL_WEIGHT_LIST="${BAYESRT_T_KL_WEIGHT_LIST}" \
    BAYESRT_EVAL_FUSION_VARIANT_LIST="${BAYESRT_EVAL_FUSION_VARIANT_LIST}" \
    ACC_DROP="${ACC_DROP:-0.8}"
else
  echo "[confirm] skipped because RUN_CONFIRM=${RUN_CONFIRM} or SUMMARY_ONLY=${SUMMARY_ONLY:-0}"
fi

echo "[done] focused unified BayesRT sweep + unified-tag confirm/B2N finished."
