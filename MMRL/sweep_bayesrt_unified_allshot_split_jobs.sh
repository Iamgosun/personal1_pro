#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BayesRTMMRL strict-ECE all-shot unified hyperparameter sweep.
#
# Purpose:
#   Find ONE BayesRTMMRL hyperparameter tag shared across all datasets,
#   using a selection objective that is aware of low-shot behavior.
#
# Why this version exists:
#   Selecting the unified tag only from 16-shot seed1 can overfit to the
#   16-shot calibration regime and produce poor ECE for 1/2/4-shot confirm.
#   This wrapper tunes on multiple shots by default: 1, 2, 4, and 16.
#
# Flow:
#   1) Run BayesRTMMRL-only tuning with a low-shot-stable focused grid.
#   2) Select ONE unified tag over all tuned dataset x shot pairs.
#   3) Patch a temporary copy of the base launcher so FS confirm and B2N
#      confirm use that unified tag instead of per-dataset best tags.
#   4) Run FS confirm and B2N confirm.
#   5) Delete checkpoint/model weights after successful completion.
#
# Base launcher expected:
#   sweep_vcrm_bayes_baselines_newrules.sh
#
# Example:
#   bash sweep_bayesrt_unified_allshot_robust_with_confirm_delete_weights.sh \
#     PROJECT_DIR="$PWD" \
#     DATA_ROOT=DATASETS \
#     OUTPUT_ROOT=output_sweeps/bayesrt_unified_allshot_strict_ece \
#     RESET_MANIFEST=1 SKIP_EXISTING=1 \
#     GPU_IDS="0 1 2 3 4 5" JOBS_PER_GPU=1
#
# Notes:
#   - Use JOBS_PER_GPU=1 if disk space is tight. Checkpoints are deleted
#     only after a run has produced metrics, so peak storage still depends
#     on concurrency.
#   - Use RUN_CONFIRM=0 to do tune + unified selection only.
#   - By default, tuning is split by shot concurrency:
#       LOW_SHOT_TUNE_SHOTS="1 2 4" uses LOW_SHOT_JOBS_PER_GPU=6.
#       SHOT16_TUNE_SHOTS="16" uses SHOT16_JOBS_PER_GPU=3.
#     Set SPLIT_TUNE_BY_SHOT_JOBS=0 to use the old single JOBS_PER_GPU path.
#   - Use SUMMARY_ONLY=1 only to rebuild summaries/selection from manifest.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
BASE_SCRIPT=${BASE_SCRIPT:-${PROJECT_DIR}/sweep_vcrm_bayes_baselines_newrules.sh}
if [[ ! -f "${BASE_SCRIPT}" && -f "${SCRIPT_DIR}/sweep_vcrm_bayes_baselines_newrules.sh" ]]; then
  BASE_SCRIPT="${SCRIPT_DIR}/sweep_vcrm_bayes_baselines_newrules.sh"
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayesrt_unified_allshot_strict_ece}
DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft oxford_pets stanford_cars ucf101"}
TUNE_DATASETS=${TUNE_DATASETS:-${DATASETS}}

# All-shot-aware tuning proxy. This is the key change from the prior script.
TUNE_SHOTS=${TUNE_SHOTS:-"1 2 4 16"}
TUNE_SEEDS=${TUNE_SEEDS:-"1"}

# Per-shot tuning concurrency. By default:
#   - 1/2/4-shot tune jobs use 6 jobs per GPU
#   - 16-shot tune jobs use 3 jobs per GPU
# Set SPLIT_TUNE_BY_SHOT_JOBS=0 to restore the old single JOBS_PER_GPU behavior.
SPLIT_TUNE_BY_SHOT_JOBS=${SPLIT_TUNE_BY_SHOT_JOBS:-1}
LOW_SHOT_TUNE_SHOTS=${LOW_SHOT_TUNE_SHOTS:-"1 2 4"}
LOW_SHOT_JOBS_PER_GPU=${LOW_SHOT_JOBS_PER_GPU:-5}
SHOT16_TUNE_SHOTS=${SHOT16_TUNE_SHOTS:-"16"}
SHOT16_JOBS_PER_GPU=${SHOT16_JOBS_PER_GPU:-3}
OTHER_SHOT_JOBS_PER_GPU=${OTHER_SHOT_JOBS_PER_GPU:-${JOBS_PER_GPU:-1}}

CONFIRM_DATASETS=${CONFIRM_DATASETS:-${DATASETS}}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-"1 2 4 8 16 32"}
B2N_SHOTS=${B2N_SHOTS:-"16"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"1 2 3"}
RUN_CONFIRM=${RUN_CONFIRM:-1}

# Keep disk usage under control. The base launcher deletes after metrics exist;
# this wrapper also does a final cleanup pass after the whole run succeeds.
DELETE_CKPT_AFTER_TEST=1
export DELETE_CKPT_AFTER_TEST
CLEAN_WEIGHTS_AFTER_RUN=${CLEAN_WEIGHTS_AFTER_RUN:-1}
CLEAN_INCOMPLETE_BEFORE_RUN=${CLEAN_INCOMPLETE_BEFORE_RUN:-1}
CLEAN_PARTIAL_B2N_BEFORE_RUN=${CLEAN_PARTIAL_B2N_BEFORE_RUN:-1}

# Low-shot-stable BayesRT grid. This deliberately avoids rpstd=0.1 as the
# default center because it looked too 16-shot-specific in confirm.
BAYESRT_R_PRIOR_STD_LIST=${BAYESRT_R_PRIOR_STD_LIST:-"0.001 0.005 0.01 0.02 0.03 "}
BAYESRT_R_KL_WEIGHT_LIST=${BAYESRT_R_KL_WEIGHT_LIST:-"1e-6 1e-5 1e-4"}
BAYESRT_T_PRIOR_STD_LIST=${BAYESRT_T_PRIOR_STD_LIST:-"0.001 0.003 0.005 0.01"}
BAYESRT_T_KL_WEIGHT_LIST=${BAYESRT_T_KL_WEIGHT_LIST:-"1e-4 1e-2 5e-2 1e-1"}
BAYESRT_EVAL_FUSION_VARIANT_LIST=${BAYESRT_EVAL_FUSION_VARIANT_LIST:-"static_prob static_logit"}



# Optional wider mode, still biased toward low-shot robustness.
if [[ "${EXTENDED:-0}" == "1" ]]; then
  BAYESRT_R_PRIOR_STD_LIST=${BAYESRT_R_PRIOR_STD_LIST_EXT:-"0.002 0.003 0.005 0.007 0.01 0.02 0.03 0.05"}
  BAYESRT_R_KL_WEIGHT_LIST=${BAYESRT_R_KL_WEIGHT_LIST_EXT:-"1e-6 5e-6 1e-5 5e-5 1e-4"}
  BAYESRT_T_PRIOR_STD_LIST=${BAYESRT_T_PRIOR_STD_LIST_EXT:-"0.001 0.002 0.003 0.005 0.007"}
  BAYESRT_T_KL_WEIGHT_LIST=${BAYESRT_T_KL_WEIGHT_LIST_EXT:-"1e-3 1e-2 5e-2 1e-1"}
  BAYESRT_EVAL_FUSION_VARIANT_LIST=${BAYESRT_EVAL_FUSION_VARIANT_LIST_EXT:-"static_prob"}
fi

# Unified selection controls.
# The selector first filters with these constraints when possible, then chooses
# the lowest robust score. If no tag satisfies constraints, it falls back to
# all tags but still uses the same score.
UNIFIED_TOPK=${UNIFIED_TOPK:-30}
TARGET_MEAN_ECE=${TARGET_MEAN_ECE:-2.0}
UNIFIED_MAX_MEAN_ACC_DROP=${UNIFIED_MAX_MEAN_ACC_DROP:-2.0}
UNIFIED_MAX_PAIR_ACC_DROP=${UNIFIED_MAX_PAIR_ACC_DROP:-6.0}
UNIFIED_MAX_SHOT_MEAN_ECE=${UNIFIED_MAX_SHOT_MEAN_ECE:-8.0}
UNIFIED_MAX_1SHOT_MEAN_ECE=${UNIFIED_MAX_1SHOT_MEAN_ECE:-5.5}
UNIFIED_MAX_2SHOT_MEAN_ECE=${UNIFIED_MAX_2SHOT_MEAN_ECE:-5.0}
UNIFIED_MAX_4SHOT_MEAN_ECE=${UNIFIED_MAX_4SHOT_MEAN_ECE:-4.0}
UNIFIED_MAX_16SHOT_MEAN_ECE=${UNIFIED_MAX_16SHOT_MEAN_ECE:-3.0}

# Robust score = all-pair mean ECE + penalties for worst shot, low shot, and ACC drop.
# Set weights to 0 if you want pure macro mean ECE.
UNIFIED_SCORE_MAX_SHOT_WEIGHT=${UNIFIED_SCORE_MAX_SHOT_WEIGHT:-0.25}
UNIFIED_SCORE_LOW_SHOT_WEIGHT=${UNIFIED_SCORE_LOW_SHOT_WEIGHT:-0.15}
UNIFIED_SCORE_ACC_DROP_WEIGHT=${UNIFIED_SCORE_ACC_DROP_WEIGHT:-0.05}
LOW_SHOTS_FOR_SCORE=${LOW_SHOTS_FOR_SCORE:-"1 2 4"}

cleanup_weight_files() {
  if [[ "${CLEAN_WEIGHTS_AFTER_RUN:-1}" != "1" ]]; then
    echo "[cleanup] skipped checkpoint cleanup because CLEAN_WEIGHTS_AFTER_RUN=${CLEAN_WEIGHTS_AFTER_RUN:-0}"
    return 0
  fi
  if [[ -z "${OUTPUT_ROOT:-}" || ! -d "${OUTPUT_ROOT}" ]]; then
    return 0
  fi

  echo "[cleanup] deleting checkpoint/model weight files under ${OUTPUT_ROOT}"
  find "${OUTPUT_ROOT}" -type f \( \
      -name "*.pth" -o \
      -name "*.pth.tar" -o \
      -name "*.pt" -o \
      -name "*.ckpt" -o \
      -name "*.bin" -o \
      -name "*.safetensors" -o \
      -name "checkpoint*" -o \
      -name "model-best*" -o \
      -name "model_best*" \
    \) -print -delete 2>/dev/null || true

  echo "[cleanup] deleting refactor_model/tensorboard dirs under ${OUTPUT_ROOT}"
  find "${OUTPUT_ROOT}" -type d \( \
      -name "refactor_model" -o \
      -name "tensorboard" \
    \) -prune -print -exec rm -rf {} + 2>/dev/null || true
}

clean_incomplete_seed_dirs() {
  if [[ "${CLEAN_INCOMPLETE_BEFORE_RUN:-1}" != "1" ]]; then
    echo "[preclean] skipped incomplete-dir cleanup"
    return 0
  fi
  if [[ -z "${OUTPUT_ROOT:-}" || ! -d "${OUTPUT_ROOT}" ]]; then
    return 0
  fi

  echo "[preclean] removing incomplete seed dirs under ${OUTPUT_ROOT}"
  echo "[preclean] rule: delete seed* dirs without test_metrics.json"
  while IFS= read -r d; do
    if [[ -d "$d" && ! -f "$d/test_metrics.json" ]]; then
      echo "[preclean] rm -rf $d"
      rm -rf "$d"
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -name 'seed*' 2>/dev/null || true)
}

clean_partial_b2n_dirs() {
  if [[ "${CLEAN_PARTIAL_B2N_BEFORE_RUN:-1}" != "1" ]]; then
    echo "[preclean] skipped partial-B2N cleanup"
    return 0
  fi
  if [[ -z "${OUTPUT_ROOT:-}" || ! -d "${OUTPUT_ROOT}/b2n/BayesRTMMRL/B2N/train_base" ]]; then
    return 0
  fi

  echo "[preclean] checking partial B2N train_base dirs without paired test_new metrics"
  while IFS= read -r train_seed_dir; do
    rel="${train_seed_dir#${OUTPUT_ROOT}/b2n/BayesRTMMRL/B2N/train_base/}"
    eval_seed_dir="${OUTPUT_ROOT}/b2n/BayesRTMMRL/B2N/test_new/${rel}"
    if [[ ! -f "${eval_seed_dir}/test_metrics.json" ]]; then
      echo "[preclean] rm -rf partial B2N train/eval: ${train_seed_dir} ; ${eval_seed_dir}"
      rm -rf "${train_seed_dir}" "${eval_seed_dir}"
    fi
  done < <(find "${OUTPUT_ROOT}/b2n/BayesRTMMRL/B2N/train_base" -type d -name 'seed*' 2>/dev/null || true)
}

clean_incomplete_seed_dirs
clean_partial_b2n_dirs

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[error] BASE_SCRIPT not found: ${BASE_SCRIPT}" >&2
  echo "        Put sweep_vcrm_bayes_baselines_newrules.sh in PROJECT_DIR or set BASE_SCRIPT=/path/to/it" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

cases_per_dataset=$(python - <<PY
r='${BAYESRT_R_PRIOR_STD_LIST}'.split()
rkl='${BAYESRT_R_KL_WEIGHT_LIST}'.split()
t='${BAYESRT_T_PRIOR_STD_LIST}'.split()
tkl='${BAYESRT_T_KL_WEIGHT_LIST}'.split()
f='${BAYESRT_EVAL_FUSION_VARIANT_LIST}'.split()
print(len(r)*len(rkl)*len(t)*len(tkl)*len(f))
PY
)
num_tune_shots=$(python - <<PY
print(len('${TUNE_SHOTS}'.split()))
PY
)
num_tune_datasets=$(python - <<PY
print(len('${TUNE_DATASETS}'.split()))
PY
)

echo "[config] PROJECT_DIR=${PROJECT_DIR}"
echo "[config] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] TUNE_DATASETS=${TUNE_DATASETS}"
echo "[config] TUNE_SHOTS=${TUNE_SHOTS}"
echo "[config] TUNE_SEEDS=${TUNE_SEEDS}"
echo "[config] SPLIT_TUNE_BY_SHOT_JOBS=${SPLIT_TUNE_BY_SHOT_JOBS}"
echo "[config] LOW_SHOT_TUNE_SHOTS=${LOW_SHOT_TUNE_SHOTS}"
echo "[config] LOW_SHOT_JOBS_PER_GPU=${LOW_SHOT_JOBS_PER_GPU}"
echo "[config] SHOT16_TUNE_SHOTS=${SHOT16_TUNE_SHOTS}"
echo "[config] SHOT16_JOBS_PER_GPU=${SHOT16_JOBS_PER_GPU}"
echo "[config] OTHER_SHOT_JOBS_PER_GPU=${OTHER_SHOT_JOBS_PER_GPU}"
echo "[config] CONFIRM_DATASETS=${CONFIRM_DATASETS}"
echo "[config] CONFIRM_SHOTS=${CONFIRM_SHOTS}"
echo "[config] B2N_SHOTS=${B2N_SHOTS}"
echo "[config] CONFIRM_SEEDS=${CONFIRM_SEEDS}"
echo "[config] RUN_CONFIRM=${RUN_CONFIRM}"
echo "[config] DELETE_CKPT_AFTER_TEST=1"
echo "[config] BayesRT cases per dataset-shot = ${cases_per_dataset}"
echo "[config] estimated tune runs = $((cases_per_dataset * num_tune_datasets * num_tune_shots))"
echo "[config] robust selection target macro mean ECE < ${TARGET_MEAN_ECE}"
echo "[config] shot ECE constraints: 1-shot<=${UNIFIED_MAX_1SHOT_MEAN_ECE}, 2-shot<=${UNIFIED_MAX_2SHOT_MEAN_ECE}, 4-shot<=${UNIFIED_MAX_4SHOT_MEAN_ECE}, 16-shot<=${UNIFIED_MAX_16SHOT_MEAN_ECE}"
echo "[config] score weights: max_shot=${UNIFIED_SCORE_MAX_SHOT_WEIGHT}, low_shot=${UNIFIED_SCORE_LOW_SHOT_WEIGHT}, acc_drop=${UNIFIED_SCORE_ACC_DROP_WEIGHT}"

# Phase 1: tuning only. Confirm is disabled here because we first need to
# select the unified tag.
run_bayesrt_tune_phase() {
  local phase_name=$1
  local tune_shots=$2
  local jobs_per_gpu=$3
  local reset_manifest=$4

  if [[ -z "${tune_shots// }" ]]; then
    echo "[tune:${phase_name}] skipped because no shots matched TUNE_SHOTS=${TUNE_SHOTS}"
    return 0
  fi

  echo "[tune:${phase_name}] TUNE_SHOTS=${tune_shots} JOBS_PER_GPU=${jobs_per_gpu} RESET_MANIFEST=${reset_manifest}"

  bash "${BASE_SCRIPT}" \
    PROJECT_DIR="${PROJECT_DIR}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATASETS="${DATASETS}" \
    TUNE_DATASETS="${TUNE_DATASETS}" \
    TUNE_SHOTS="${tune_shots}" \
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
    RESET_MANIFEST="${reset_manifest}" \
    SKIP_EXISTING="${SKIP_EXISTING:-1}" \
    GPU_IDS="${GPU_IDS:-}" \
    NGPU="${NGPU:-}" \
    JOBS_PER_GPU="${jobs_per_gpu}" \
    DATA_ROOT="${DATA_ROOT:-${ROOT:-DATASETS}}" \
    BACKBONE="${BACKBONE:-ViT-B/16}" \
    EXEC_MODE="${EXEC_MODE:-online}" \
    DELETE_CKPT_AFTER_TEST=1 \
    SUMMARY_ONLY="${SUMMARY_ONLY:-0}" \
    BAYESRT_R_PRIOR_STD_LIST="${BAYESRT_R_PRIOR_STD_LIST}" \
    BAYESRT_R_KL_WEIGHT_LIST="${BAYESRT_R_KL_WEIGHT_LIST}" \
    BAYESRT_T_PRIOR_STD_LIST="${BAYESRT_T_PRIOR_STD_LIST}" \
    BAYESRT_T_KL_WEIGHT_LIST="${BAYESRT_T_KL_WEIGHT_LIST}" \
    BAYESRT_EVAL_FUSION_VARIANT_LIST="${BAYESRT_EVAL_FUSION_VARIANT_LIST}" \
    ACC_DROP="${ACC_DROP:-0.8}"
}

if [[ "${SPLIT_TUNE_BY_SHOT_JOBS}" == "1" && "${SUMMARY_ONLY:-0}" != "1" ]]; then
  LOW_TUNE_SHOTS_EFFECTIVE=$(python - <<PY
tune = "${TUNE_SHOTS}".split()
low = set("${LOW_SHOT_TUNE_SHOTS}".split())
print(" ".join([s for s in tune if s in low]))
PY
)
  SHOT16_TUNE_SHOTS_EFFECTIVE=$(python - <<PY
tune = "${TUNE_SHOTS}".split()
shot16 = set("${SHOT16_TUNE_SHOTS}".split())
print(" ".join([s for s in tune if s in shot16]))
PY
)
  OTHER_TUNE_SHOTS_EFFECTIVE=$(python - <<PY
tune = "${TUNE_SHOTS}".split()
low = set("${LOW_SHOT_TUNE_SHOTS}".split())
shot16 = set("${SHOT16_TUNE_SHOTS}".split())
print(" ".join([s for s in tune if s not in low and s not in shot16]))
PY
)

  TUNE_RESET_FOR_NEXT="${RESET_MANIFEST:-0}"

  if [[ -n "${LOW_TUNE_SHOTS_EFFECTIVE// }" ]]; then
    run_bayesrt_tune_phase "low-shot" "${LOW_TUNE_SHOTS_EFFECTIVE}" "${LOW_SHOT_JOBS_PER_GPU}" "${TUNE_RESET_FOR_NEXT}"
    TUNE_RESET_FOR_NEXT=0
  fi

  if [[ -n "${SHOT16_TUNE_SHOTS_EFFECTIVE// }" ]]; then
    run_bayesrt_tune_phase "16-shot" "${SHOT16_TUNE_SHOTS_EFFECTIVE}" "${SHOT16_JOBS_PER_GPU}" "${TUNE_RESET_FOR_NEXT}"
    TUNE_RESET_FOR_NEXT=0
  fi

  if [[ -n "${OTHER_TUNE_SHOTS_EFFECTIVE// }" ]]; then
    run_bayesrt_tune_phase "other-shot" "${OTHER_TUNE_SHOTS_EFFECTIVE}" "${OTHER_SHOT_JOBS_PER_GPU}" "${TUNE_RESET_FOR_NEXT}"
    TUNE_RESET_FOR_NEXT=0
  fi
else
  echo "[tune:single] SPLIT_TUNE_BY_SHOT_JOBS=${SPLIT_TUNE_BY_SHOT_JOBS}; using one JOBS_PER_GPU=${JOBS_PER_GPU:-1} for TUNE_SHOTS=${TUNE_SHOTS}"
  run_bayesrt_tune_phase "single" "${TUNE_SHOTS}" "${JOBS_PER_GPU:-1}" "${RESET_MANIFEST:-0}"
fi

# Phase 2: select one unified BayesRT tag from all tuned dataset x shot pairs.
python - <<PY
import csv
import math
from collections import defaultdict
from pathlib import Path

output_root = Path(r"${OUTPUT_ROOT}")
summary_path = output_root / "tune_summary.csv"
selected_csv = output_root / "unified_bayesrt_allshot_config.csv"
selected_breakdown_csv = output_root / "unified_bayesrt_allshot_selected_breakdown.csv"
top_csv = output_root / "unified_bayesrt_allshot_top${UNIFIED_TOPK}.csv"
env_path = output_root / "unified_bayesrt_allshot_config.env"

tune_shots = set("${TUNE_SHOTS}".split())
tune_seeds = set("${TUNE_SEEDS}".split())
low_shots = set("${LOW_SHOTS_FOR_SCORE}".split())

topk = int("${UNIFIED_TOPK}")
target_ece = float("${TARGET_MEAN_ECE}")
max_mean_acc_drop = float("${UNIFIED_MAX_MEAN_ACC_DROP}")
max_pair_acc_drop = float("${UNIFIED_MAX_PAIR_ACC_DROP}")
max_shot_mean_ece = float("${UNIFIED_MAX_SHOT_MEAN_ECE}")
max_1shot_mean_ece = float("${UNIFIED_MAX_1SHOT_MEAN_ECE}")
max_2shot_mean_ece = float("${UNIFIED_MAX_2SHOT_MEAN_ECE}")
max_4shot_mean_ece = float("${UNIFIED_MAX_4SHOT_MEAN_ECE}")
max_16shot_mean_ece = float("${UNIFIED_MAX_16SHOT_MEAN_ECE}")
w_max_shot = float("${UNIFIED_SCORE_MAX_SHOT_WEIGHT}")
w_low_shot = float("${UNIFIED_SCORE_LOW_SHOT_WEIGHT}")
w_acc_drop = float("${UNIFIED_SCORE_ACC_DROP_WEIGHT}")

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
        if row.get("shot") not in tune_shots:
            continue
        if row.get("seed") not in tune_seeds:
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

# Best ACC baseline per dataset x shot x seed among the searched tags.
best_acc_by_pair = {}
for r in rows:
    key = (r["dataset"], r["shot"], r["seed"])
    best_acc_by_pair[key] = max(best_acc_by_pair.get(key, -math.inf), r["accuracy"])

needed_pairs = set(best_acc_by_pair)
by_tag = defaultdict(list)
for r in rows:
    by_tag[r["tag"]].append(r)

records = []
for tag, group in by_tag.items():
    present = {(r["dataset"], r["shot"], r["seed"]) for r in group}
    if present != needed_pairs:
        continue

    n = len(group)
    total_samples = sum(r["num_samples"] for r in group)
    eces = [r["ece"] for r in group]
    accs = [r["accuracy"] for r in group]
    nlls = [r["nll"] for r in group]
    briers = [r["brier"] for r in group]
    drops = [best_acc_by_pair[(r["dataset"], r["shot"], r["seed"])] - r["accuracy"] for r in group]

    by_shot = defaultdict(list)
    by_dataset = defaultdict(list)
    for r in group:
        by_shot[r["shot"]].append(r["ece"])
        by_dataset[r["dataset"]].append(r["ece"])

    shot_mean_ece = {s: sum(vals) / len(vals) for s, vals in by_shot.items()}
    dataset_mean_ece = {d: sum(vals) / len(vals) for d, vals in by_dataset.items()}
    low_vals = [r["ece"] for r in group if r["shot"] in low_shots]
    low_shot_mean_ece = sum(low_vals) / len(low_vals) if low_vals else sum(eces) / n

    weighted_ece = sum(r["ece"] * r["num_samples"] for r in group) / total_samples if total_samples else float("nan")
    weighted_acc = sum(r["accuracy"] * r["num_samples"] for r in group) / total_samples if total_samples else float("nan")

    mean_ece = sum(eces) / n
    max_shot_ece = max(shot_mean_ece.values())
    mean_acc_drop = sum(drops) / n
    max_acc_drop = max(drops)
    robust_score = mean_ece + w_max_shot * max_shot_ece + w_low_shot * low_shot_mean_ece + w_acc_drop * mean_acc_drop

    constraints_ok = True
    constraints_ok = constraints_ok and mean_acc_drop <= max_mean_acc_drop
    constraints_ok = constraints_ok and max_acc_drop <= max_pair_acc_drop
    constraints_ok = constraints_ok and max_shot_ece <= max_shot_mean_ece
    if "1" in shot_mean_ece:
        constraints_ok = constraints_ok and shot_mean_ece["1"] <= max_1shot_mean_ece
    if "2" in shot_mean_ece:
        constraints_ok = constraints_ok and shot_mean_ece["2"] <= max_2shot_mean_ece
    if "4" in shot_mean_ece:
        constraints_ok = constraints_ok and shot_mean_ece["4"] <= max_4shot_mean_ece
    if "16" in shot_mean_ece:
        constraints_ok = constraints_ok and shot_mean_ece["16"] <= max_16shot_mean_ece

    rec = {
        "tag": tag,
        "num_pairs": n,
        "mean_ece": mean_ece,
        "max_ece": max(eces),
        "max_shot_mean_ece": max_shot_ece,
        "low_shot_mean_ece": low_shot_mean_ece,
        "weighted_ece": weighted_ece,
        "mean_accuracy": sum(accs) / n,
        "weighted_accuracy": weighted_acc,
        "mean_acc_drop": mean_acc_drop,
        "max_acc_drop": max_acc_drop,
        "mean_nll": sum(nlls) / n,
        "mean_brier": sum(briers) / n,
        "robust_score": robust_score,
        "below_target_mean_ece": mean_ece < target_ece,
        "satisfies_constraints": constraints_ok,
    }

    for s in sorted(shot_mean_ece, key=lambda x: float(x)):
        rec[f"shot{s}_mean_ece"] = shot_mean_ece[s]
    for d in sorted(dataset_mean_ece):
        rec[f"dataset_{d}_mean_ece"] = dataset_mean_ece[d]

    records.append(rec)

if not records:
    raise SystemExit("[error] no tag has complete coverage over all tuned dataset x shot x seed pairs")

# Prefer constrained tags. Within that set, minimize robust score.
# Tie-break toward lower all-pair mean ECE, then lower worst-shot ECE, then lower ACC drop.
records.sort(key=lambda r: (r["robust_score"], r["mean_ece"], r["max_shot_mean_ece"], r["mean_acc_drop"], r["tag"]))
constrained = [r for r in records if r["satisfies_constraints"]]
selected = (constrained or records)[0]
selection_mode = "constrained_lowest_robust_score" if constrained else "global_lowest_robust_score_fallback"

fieldnames = [
    "selection_mode", "tag", "num_pairs", "robust_score", "mean_ece", "max_ece",
    "max_shot_mean_ece", "low_shot_mean_ece", "weighted_ece", "mean_accuracy",
    "weighted_accuracy", "mean_acc_drop", "max_acc_drop", "mean_nll", "mean_brier",
    "below_target_mean_ece", "satisfies_constraints",
]
extra_fields = sorted({k for r in records[:topk] + [selected] for k in r if k.startswith("shot") or k.startswith("dataset_")})
fieldnames = fieldnames + extra_fields

selected_csv.parent.mkdir(parents=True, exist_ok=True)
with selected_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    row = {k: selected.get(k, "") for k in fieldnames}
    row["selection_mode"] = selection_mode
    w.writerow(row)

with top_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in records[:topk]:
        row = {k: r.get(k, "") for k in fieldnames}
        row["selection_mode"] = "ranked_by_robust_score"
        w.writerow(row)

selected_group = [r for r in by_tag[selected["tag"]] if (r["dataset"], r["shot"], r["seed"]) in needed_pairs]
breakdown_fields = ["tag", "dataset", "shot", "seed", "accuracy", "best_acc", "acc_drop", "ece", "nll", "brier", "num_samples", "outdir", "metrics_path"]
with selected_breakdown_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=breakdown_fields)
    w.writeheader()
    for r in sorted(selected_group, key=lambda x: (float(x["shot"]), x["dataset"], x["seed"])):
        pair = (r["dataset"], r["shot"], r["seed"])
        w.writerow({
            "tag": selected["tag"],
            "dataset": r["dataset"],
            "shot": r["shot"],
            "seed": r["seed"],
            "accuracy": r["accuracy"],
            "best_acc": best_acc_by_pair[pair],
            "acc_drop": best_acc_by_pair[pair] - r["accuracy"],
            "ece": r["ece"],
            "nll": r["nll"],
            "brier": r["brier"],
            "num_samples": r.get("num_samples", ""),
            "outdir": r.get("outdir", ""),
            "metrics_path": r.get("metrics_path", ""),
        })

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

env_lines = [
    f"UNIFIED_BAYESRT_TAG={shq(selected['tag'])}",
    f"UNIFIED_BAYESRT_SELECTION_MODE={shq(selection_mode)}",
    f"UNIFIED_BAYESRT_ROBUST_SCORE={selected['robust_score']:.6f}",
    f"UNIFIED_BAYESRT_MEAN_ECE={selected['mean_ece']:.6f}",
    f"UNIFIED_BAYESRT_MAX_ECE={selected['max_ece']:.6f}",
    f"UNIFIED_BAYESRT_MAX_SHOT_MEAN_ECE={selected['max_shot_mean_ece']:.6f}",
    f"UNIFIED_BAYESRT_LOW_SHOT_MEAN_ECE={selected['low_shot_mean_ece']:.6f}",
    f"UNIFIED_BAYESRT_WEIGHTED_ECE={selected['weighted_ece']:.6f}",
    f"UNIFIED_BAYESRT_MEAN_ACC_DROP={selected['mean_acc_drop']:.6f}",
    f"UNIFIED_BAYESRT_MAX_ACC_DROP={selected['max_acc_drop']:.6f}",
]
env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

print("[unified] selection_mode=", selection_mode)
print("[unified] tag=", selected["tag"])
print("[unified] robust_score=", f"{selected['robust_score']:.6f}")
print("[unified] mean_ece=", f"{selected['mean_ece']:.6f}")
print("[unified] max_shot_mean_ece=", f"{selected['max_shot_mean_ece']:.6f}")
print("[unified] low_shot_mean_ece=", f"{selected['low_shot_mean_ece']:.6f}")
print("[unified] weighted_ece=", f"{selected['weighted_ece']:.6f}")
print("[unified] mean_acc_drop=", f"{selected['mean_acc_drop']:.6f}")
print("[unified] max_acc_drop=", f"{selected['max_acc_drop']:.6f}")
print("[unified] tags_below_target_mean_ece=", sum(1 for r in records if r["below_target_mean_ece"]))
print("[unified] wrote", selected_csv)
print("[unified] wrote", env_path)
print("[unified] wrote", top_csv)
print("[unified] wrote", selected_breakdown_csv)
PY

# Phase 3: run FS confirm and B2N confirm with the same selected tag.
if [[ "${RUN_CONFIRM}" == "1" && "${SUMMARY_ONLY:-0}" != "1" ]]; then
  UNIFIED_ENV="${OUTPUT_ROOT}/unified_bayesrt_allshot_config.env"
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
    JOBS_PER_GPU="${JOBS_PER_GPU:-1}" \
    DATA_ROOT="${DATA_ROOT:-${ROOT:-DATASETS}}" \
    BACKBONE="${BACKBONE:-ViT-B/16}" \
    EXEC_MODE="${EXEC_MODE:-online}" \
    DELETE_CKPT_AFTER_TEST=1 \
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

cleanup_weight_files
echo "[done] strict-ECE all-shot unified BayesRT sweep + confirm/B2N finished; checkpoint/model weights deleted."
