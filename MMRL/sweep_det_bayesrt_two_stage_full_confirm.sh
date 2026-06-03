#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DetBayesRTMMRL two-stage hyperparameter sweep.
#
# Stage 1:
#   Use default DetBayesRTMMRL model hyperparameters and sweep OPTIM.MAX_EPOCH.
#   Default grid: 50 80 100.
#
# Stage 2:
#   Fix the selected MAX_EPOCH and sweep DetBayesRTMMRL core hyperparameters:
#     BAYESRT_MMRL.R_PRIOR_STD
#     BAYESRT_MMRL.R_KL_WEIGHT
#     BAYESRT_MMRL.T_PRIOR_STD
#     BAYESRT_MMRL.T_KL_WEIGHT
#     BAYESRT_MMRL.EVAL_FUSION_VARIANT
#
# Important:
#   - HPO.ENABLED is forced to false for every launched run, so the YAML HPO
#     grid does not start and each external sweep candidate has its own
#     test_report.json/test_metrics.json result file.
#   - After test_report.json or test_metrics.json exists, generated
#     refactor_model and tensorboard directories are deleted immediately.
#   - Core/epoch selection follows the existing sweep convention:
#     keep candidates with mean ACC >= best mean ACC - ACC_DROP, then
#     choose the lowest mean ECE; tie-break by higher ACC and lower NLL.
#   - After the best core hyperparameters are selected, run full few-shot
#     confirmation on CONFIRM_SHOTS="1 2 4 8 16 32" and
#     CONFIRM_SEEDS="1 2 3" by default.
#   - For B2N-style train/eval, use a separate wrapper. This file is FS tuning
#     oriented, matching your request to inspect each candidate's test result.
# ============================================================

apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      if [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        printf -v "${key}" '%s' "${val}"
        export "${key}"
      else
        echo "[warn] invalid KEY ignored: ${key}" >&2
      fi
    else
      echo "[warn] non KEY=VALUE ignored: ${arg}" >&2
    fi
  done
}

apply_kv_args "$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=${PROJECT_DIR:-${SCRIPT_DIR}}
ROOT=${ROOT:-${DATA_ROOT:-DATASETS}}
DATA_ROOT=${DATA_ROOT:-${ROOT}}

METHOD=${METHOD:-DetBayesRTMMRL}
METHOD_CFG=${METHOD_CFG:-configs/methods/det_bayesrt_mmrl.yaml}
RUNTIME_CFG=${RUNTIME_CFG:-configs/runtime/mmrl_family.yaml}
PROTOCOL_CFG=${PROTOCOL_CFG:-configs/protocols/fs.yaml}
PROTOCOL=${PROTOCOL:-FS}
PHASE=${PHASE:-fewshot_train}
SUBSAMPLE=${SUBSAMPLE:-all}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}

OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/det_bayesrt_two_stage_60}
EPOCH_ROOT="${OUTPUT_ROOT}/stage1_max_epoch"
CORE_ROOT="${OUTPUT_ROOT}/stage2_core_hparams"
CONFIRM_ROOT="${OUTPUT_ROOT}/stage3_confirm_fullshot"

DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft stanford_cars "}
TUNE_DATASETS=${TUNE_DATASETS:-${DATASETS}}
TUNE_SHOTS=${TUNE_SHOTS:-"1  4 8 16"}
TUNE_SEEDS=${TUNE_SEEDS:-"1"}

MAX_EPOCH_LIST=${MAX_EPOCH_LIST:-" 60  "}

DET_R_PRIOR_STD_LIST=${DET_R_PRIOR_STD_LIST:-"0.1 0.01 0.001"}
DET_R_KL_WEIGHT_LIST=${DET_R_KL_WEIGHT_LIST:-"1.0e-5 1.0e-4 1.0e-3"}
DET_T_PRIOR_STD_LIST=${DET_T_PRIOR_STD_LIST:-"0.1 0.01 0.001"}
DET_T_KL_WEIGHT_LIST=${DET_T_KL_WEIGHT_LIST:-"5.0e-2 1.0e-2 1.0e-3"}
DET_EVAL_FUSION_VARIANT_LIST=${DET_EVAL_FUSION_VARIANT_LIST:-" static_logit"}



GPU_IDS=${GPU_IDS:-}
NGPU=${NGPU:-}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}
SLEEP_SEC=${SLEEP_SEC:-2}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RESET_MANIFEST=${RESET_MANIFEST:-0}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}
RUN_EPOCH_SWEEP=${RUN_EPOCH_SWEEP:-1}
RUN_CORE_SWEEP=${RUN_CORE_SWEEP:-1}
RUN_CONFIRM_FULL=${RUN_CONFIRM_FULL:-1}
STRICT_COMPLETE_SELECTION=${STRICT_COMPLETE_SELECTION:-1}
ACC_DROP=${ACC_DROP:-0.8}

CONFIRM_DATASETS=${CONFIRM_DATASETS:-${TUNE_DATASETS}}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-"1 2 4 8 16 32"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"1 2 3"}

# global: use the single best config in core_best_global.csv for every dataset.
# per_dataset: use core_best_per_dataset.csv when available, falling back to global.
CONFIRM_BEST_MODE=${CONFIRM_BEST_MODE:-global}

# Correct cleanup target for this project:
# refactor_model and tensorboard are generated after training.
DELETE_ARTIFACTS_AFTER_TEST=${DELETE_ARTIFACTS_AFTER_TEST:-1}

# Kept as a safety net only. Your current runner mainly writes refactor_model.
DELETE_WEIGHT_FILES_AFTER_TEST=${DELETE_WEIGHT_FILES_AFTER_TEST:-1}

MANIFEST="${OUTPUT_ROOT}/run_manifest.csv"
EPOCH_SUMMARY="${OUTPUT_ROOT}/epoch_sweep_summary.csv"
EPOCH_SELECTION="${OUTPUT_ROOT}/epoch_selection.csv"
BEST_MAX_EPOCH_ENV="${OUTPUT_ROOT}/best_max_epoch.env"
CORE_SUMMARY="${OUTPUT_ROOT}/core_sweep_summary.csv"
CORE_RANKED="${OUTPUT_ROOT}/core_candidates_ranked.csv"
CORE_BEST_GLOBAL="${OUTPUT_ROOT}/core_best_global.csv"
CORE_BEST_PER_DATASET="${OUTPUT_ROOT}/core_best_per_dataset.csv"
CONFIRM_SUMMARY="${OUTPUT_ROOT}/confirm_fullshot_summary.csv"
ALL_SUMMARY="${OUTPUT_ROOT}/all_summary.csv"
BEST_CORE_ENV="${OUTPUT_ROOT}/best_core_config.env"

FAILED_JOBS=0
READY_SLOT=""
READY_GPU=""

declare -ga PHYSICAL_GPUS
declare -ga RUNNING_PIDS
declare -ga SLOT_GPU
declare -ga SLOT_DESC
declare -ga SLOT_LOG
declare -gA GPU_USED

sanitize() {
  local s="$1"
  s="${s//\//-}"
  s="${s// /_}"
  s="${s//,/}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s//:/-}"
  s="${s//|/-}"
  echo "${s}"
}

backbone_tag() {
  echo "${BACKBONE//\//-}"
}

build_outdir() {
  local base_root=$1 stage=$2 dataset=$3 shot=$4 seed=$5 tag=$6
  echo "${base_root}/${METHOD}/${stage}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/$(backbone_tag)/${tag}/seed${seed}"
}

ensure_manifest_header() {
  mkdir -p "$(dirname "${MANIFEST}")"
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "stage,dataset,shot,seed,tag,max_epoch,r_prior_std,r_kl_weight,t_prior_std,t_kl_weight,fusion,selected_from_tag,outdir" > "${MANIFEST}"
  fi
}

append_manifest() {
  local stage=$1 dataset=$2 shot=$3 seed=$4 tag=$5 max_epoch=$6 r_prior=$7 r_kl=$8 t_prior=$9 t_kl=${10} fusion=${11} selected_from_tag=${12} outdir=${13}
  ensure_manifest_header
  echo "${stage},${dataset},${shot},${seed},${tag},${max_epoch},${r_prior},${r_kl},${t_prior},${t_kl},${fusion},${selected_from_tag},${outdir}" >> "${MANIFEST}"
}

reset_index_files_if_requested() {
  if [[ "${RESET_MANIFEST}" != "1" || "${SUMMARY_ONLY}" == "1" ]]; then
    return 0
  fi

  echo "[reset] removing manifest and summary/index files only; experiment outputs are preserved"
  rm -f "${MANIFEST}" "${EPOCH_SUMMARY}" "${EPOCH_SELECTION}" "${BEST_MAX_EPOCH_ENV}" \
        "${CORE_SUMMARY}" "${CORE_RANKED}" "${CORE_BEST_GLOBAL}" \
        "${CORE_BEST_PER_DATASET}" "${BEST_CORE_ENV}" "${CONFIRM_SUMMARY}" "${ALL_SUMMARY}"
}

cleanup_run_artifacts() {
  local outdir=$1

  if [[ "${DELETE_ARTIFACTS_AFTER_TEST}" != "1" ]]; then
    return 0
  fi
  if [[ -z "$(case_result_json "${outdir}")" ]]; then
    return 0
  fi

  echo "[cleanup] ${outdir}: deleting refactor_model/tensorboard after test_report.json/test_metrics.json"
  find "${outdir}" -type d \( \
      -name "refactor_model" -o \
      -name "tensorboard" \
    \) -prune -exec rm -rf {} + 2>/dev/null || true

  if [[ "${DELETE_WEIGHT_FILES_AFTER_TEST}" == "1" ]]; then
    find "${outdir}" -type f \( \
        -name "*.pth" -o \
        -name "*.pth-*" -o \
        -name "*.pth.tar" -o \
        -name "*.pth.tar-*" -o \
        -name "*.pt" -o \
        -name "*.pt-*" -o \
        -name "*.ckpt" -o \
        -name "*.ckpt-*" -o \
        -name "*.bin" -o \
        -name "*.safetensors" -o \
        -name "checkpoint*" -o \
        -name "model-best*" -o \
        -name "model_best*" \
      \) -delete 2>/dev/null || true
  fi
}

case_result_json() {
  local outdir=$1
  if [[ -f "${outdir}/test_metrics.json" ]]; then
    echo "${outdir}/test_metrics.json"
  elif [[ -f "${outdir}/test_report.json" ]]; then
    echo "${outdir}/test_report.json"
  else
    echo ""
  fi
}

case_is_complete() {
  local outdir=$1
  [[ -n "$(case_result_json "${outdir}")" ]]
}

init_gpu_list() {
  PHYSICAL_GPUS=()

  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"
  elif [[ -n "${NGPU}" ]]; then
    local i
    for ((i=0; i<NGPU; i++)); do
      PHYSICAL_GPUS+=("${i}")
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r idx; do
      [[ -n "${idx}" ]] && PHYSICAL_GPUS+=("${idx}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi

  if [[ ${#PHYSICAL_GPUS[@]} -eq 0 ]]; then
    echo "[error] no visible GPU found. Set GPU_IDS or NGPU." >&2
    exit 1
  fi
  if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] invalid JOBS_PER_GPU=${JOBS_PER_GPU}" >&2
    exit 1
  fi

  GPU_USED=()
  local gpu
  for gpu in "${PHYSICAL_GPUS[@]}"; do
    GPU_USED["${gpu}"]=0
  done

  echo "[GPU] physical: ${PHYSICAL_GPUS[*]}"
  echo "[GPU] jobs_per_gpu: ${JOBS_PER_GPU}"
}

init_slots() {
  local nslots=$((${#PHYSICAL_GPUS[@]} * JOBS_PER_GPU))
  local i

  RUNNING_PIDS=()
  SLOT_GPU=()
  SLOT_DESC=()
  SLOT_LOG=()

  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_DESC[$i]=""
    SLOT_LOG[$i]=""
  done
}

cleanup_children() {
  local p
  for p in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${p:-}" ]] && kill -0 "${p}" 2>/dev/null; then
      kill "${p}" 2>/dev/null || true
    fi
  done
}

_find_empty_process_slot() {
  local idx
  for idx in "${!RUNNING_PIDS[@]}"; do
    if [[ -z "${RUNNING_PIDS[$idx]}" ]]; then
      echo "${idx}"
      return 0
    fi
  done
  return 1
}

_reap_finished_jobs() {
  local idx pid rc gpu
  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      gpu="${SLOT_GPU[$idx]}"
      if [[ -n "${gpu}" ]]; then
        GPU_USED["${gpu}"]=$(( ${GPU_USED["${gpu}"]:-0} - 1 ))
        if [[ ${GPU_USED["${gpu}"]} -lt 0 ]]; then
          GPU_USED["${gpu}"]=0
        fi
      fi

      if [[ "${rc}" -eq 0 ]]; then
        echo "[done] ${SLOT_DESC[$idx]}"
      else
        echo "[failed] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_DESC[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}

wait_for_slot() {
  local gpu used slot
  READY_SLOT=""
  READY_GPU=""

  while true; do
    _reap_finished_jobs

    for gpu in "${PHYSICAL_GPUS[@]}"; do
      used="${GPU_USED["${gpu}"]:-0}"
      if (( used < JOBS_PER_GPU )); then
        slot="$(_find_empty_process_slot || true)"
        if [[ -n "${slot}" ]]; then
          READY_SLOT="${slot}"
          READY_GPU="${gpu}"
          return 0
        fi
      fi
    done

    sleep "${SLEEP_SEC}"
  done
}

wait_all_jobs() {
  local idx pid rc gpu
  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    if [[ -n "${pid}" ]]; then
      rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      gpu="${SLOT_GPU[$idx]}"
      if [[ -n "${gpu}" ]]; then
        GPU_USED["${gpu}"]=$(( ${GPU_USED["${gpu}"]:-0} - 1 ))
        if [[ ${GPU_USED["${gpu}"]} -lt 0 ]]; then
          GPU_USED["${gpu}"]=0
        fi
      fi

      if [[ "${rc}" -eq 0 ]]; then
        echo "[done] ${SLOT_DESC[$idx]}"
      else
        echo "[failed] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_DESC[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}

run_case() {
  local stage=$1 dataset=$2 shot=$3 seed=$4 tag=$5 gpu=$6 max_epoch=$7 r_prior=$8 r_kl=$9 t_prior=${10} t_kl=${11} fusion=${12} selected_from_tag=${13}
  shift 13
  local opts=("$@")

  local base_root outdir logfile statusfile
  if [[ "${stage}" == "epoch" ]]; then
    base_root="${EPOCH_ROOT}"
  elif [[ "${stage}" == "confirm" ]]; then
    base_root="${CONFIRM_ROOT}"
  else
    base_root="${CORE_ROOT}"
  fi

  outdir="$(build_outdir "${base_root}" "${stage}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  logfile="${outdir}/run.log"
  statusfile="${outdir}/job_status.txt"

  mkdir -p "${outdir}"
  append_manifest "${stage}" "${dataset}" "${shot}" "${seed}" "${tag}" "${max_epoch}" "${r_prior}" "${r_kl}" "${t_prior}" "${t_kl}" "${fusion}" "${selected_from_tag}" "${outdir}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && case_is_complete "${outdir}"; then
    echo "SKIP" > "${statusfile}"
    cleanup_run_artifacts "${outdir}"
    echo "[skip] stage=${stage} dataset=${dataset} shot=${shot} seed=${seed} tag=${tag}"
    return 0
  fi

  : > "${logfile}"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: ${stage}"
    echo "GPU: ${gpu}"
    echo "METHOD: ${METHOD}"
    echo "RUN_TAG: ${tag}"
    echo "DATASET: ${dataset}"
    echo "SHOT: ${shot}"
    echo "SEED: ${seed}"
    echo "MAX_EPOCH: ${max_epoch}"
    echo "R_PRIOR_STD: ${r_prior}"
    echo "R_KL_WEIGHT: ${r_kl}"
    echo "T_PRIOR_STD: ${t_prior}"
    echo "T_KL_WEIGHT: ${t_kl}"
    echo "EVAL_FUSION_VARIANT: ${fusion}"
    echo "METHOD_CONFIG: ${METHOD_CFG}"
    echo "RUNTIME_CONFIG: ${RUNTIME_CFG}"
    echo "PROTOCOL_CONFIG: ${PROTOCOL_CFG}"
    echo "EXTRA_OPTS: ${opts[*]:-}"
    echo "============================================================"
  } >> "${logfile}"

  if CUDA_VISIBLE_DEVICES="${gpu}" python run.py \
      --root "${DATA_ROOT}" \
      --dataset-config-file "configs/datasets/${dataset}.yaml" \
      --method-config-file "${METHOD_CFG}" \
      --protocol-config-file "${PROTOCOL_CFG}" \
      --runtime-config-file "${RUNTIME_CFG}" \
      --output-dir "${outdir}" \
      --method "${METHOD}" \
      --protocol "${PROTOCOL}" \
      --exec-mode "${EXEC_MODE}" \
      --seed "${seed}" \
      DATASET.NUM_SHOTS "${shot}" \
      DATASET.SUBSAMPLE_CLASSES "${SUBSAMPLE}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      METHOD.TAG "${tag}" \
      OPTIM.MAX_EPOCH "${max_epoch}" \
      "${opts[@]}" \
      >> "${logfile}" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "============================================================"
    } >> "${logfile}"

    echo "SUCCESS" > "${statusfile}"
    cleanup_run_artifacts "${outdir}"
    return 0
  else
    local rc=$?
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: FAILED"
      echo "EXIT_CODE: ${rc}"
      echo "============================================================"
    } >> "${logfile}"

    echo "FAILED(${rc})" > "${statusfile}"
    return "${rc}"
  fi
}

submit_case() {
  local stage=$1 dataset=$2 shot=$3 seed=$4 tag=$5 max_epoch=$6 r_prior=$7 r_kl=$8 t_prior=$9 t_kl=${10} fusion=${11} selected_from_tag=${12}
  shift 12
  local opts=("$@")

  wait_for_slot
  local slot="${READY_SLOT}"
  local gpu="${READY_GPU}"
  local base_root outdir logfile desc

  if [[ "${stage}" == "epoch" ]]; then
    base_root="${EPOCH_ROOT}"
  elif [[ "${stage}" == "confirm" ]]; then
    base_root="${CONFIRM_ROOT}"
  else
    base_root="${CORE_ROOT}"
  fi

  outdir="$(build_outdir "${base_root}" "${stage}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  logfile="${outdir}/run.log"
  mkdir -p "${outdir}"
  desc="stage=${stage} dataset=${dataset} shot=${shot} seed=${seed} tag=${tag} gpu=${gpu}"

  (
    run_case "${stage}" "${dataset}" "${shot}" "${seed}" "${tag}" "${gpu}" "${max_epoch}" "${r_prior}" "${r_kl}" "${t_prior}" "${t_kl}" "${fusion}" "${selected_from_tag}" "${opts[@]}"
  ) >> "${logfile}" 2>&1 &

  RUNNING_PIDS[$slot]=$!
  SLOT_GPU[$slot]="${gpu}"
  SLOT_DESC[$slot]="${desc}"
  SLOT_LOG[$slot]="${logfile}"
  GPU_USED["${gpu}"]=$(( ${GPU_USED["${gpu}"]:-0} + 1 ))

  echo "[launch] ${desc}"
}

summarize_manifest() {
  local stage_filter=$1 summary_csv=$2

  python - <<PY
import csv
import json
from pathlib import Path

manifest = Path(r"${MANIFEST}")
out = Path(r"${summary_csv}")
stage_filter = "${stage_filter}"
out.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "stage", "dataset", "shot", "seed", "tag", "max_epoch",
    "r_prior_std", "r_kl_weight", "t_prior_std", "t_kl_weight", "fusion",
    "selected_from_tag", "outdir", "status", "num_samples", "accuracy", "error", "macro_f1",
    "ece", "nll", "brier", "metrics_path",
]

rows = []
seen = set()

if manifest.exists():
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage") != stage_filter:
                continue

            key = tuple(row.get(k, "") for k in ["stage", "dataset", "shot", "seed", "tag", "outdir"])
            if key in seen:
                continue
            seen.add(key)

            outdir = Path(row.get("outdir", ""))
            metrics_path = outdir / "test_metrics.json"
            if not metrics_path.exists():
                alt = outdir / "test_report.json"
                if alt.exists():
                    metrics_path = alt

            status = "ok" if metrics_path.exists() else "missing"
            metrics = {}
            num_samples = ""

            if metrics_path.exists():
                try:
                    data = json.loads(metrics_path.read_text(encoding="utf-8"))
                    num_samples = data.get("num_samples", "")
                    metrics = data.get("metrics", {}) or {}
                    if not metrics:
                        metrics = data
                except Exception as exc:
                    status = f"error:{type(exc).__name__}"

            rows.append({
                **row,
                "status": status,
                "num_samples": num_samples,
                "accuracy": metrics.get("accuracy", ""),
                "error": metrics.get("error", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "ece": metrics.get("ece", ""),
                "nll": metrics.get("nll", ""),
                "brier": metrics.get("brier", ""),
                "metrics_path": str(metrics_path),
            })

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print(f"[summary] wrote {out} rows={len(rows)}")
PY
}

select_max_epoch() {
  python - <<PY
import csv
import math
from collections import defaultdict
from pathlib import Path

summary = Path(r"${EPOCH_SUMMARY}")
out_csv = Path(r"${EPOCH_SELECTION}")
env_path = Path(r"${BEST_MAX_EPOCH_ENV}")
strict = "${STRICT_COMPLETE_SELECTION}" == "1"
acc_drop = float("${ACC_DROP}")

if not summary.exists():
    raise SystemExit(f"[error] missing {summary}")

rows = []
with summary.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "ok":
            continue
        try:
            row["_acc"] = float(row.get("accuracy", "nan"))
            row["_ece"] = float(row.get("ece", "nan"))
            row["_nll"] = float(row.get("nll", "nan"))
            row["_brier"] = float(row.get("brier", "nan"))
        except Exception:
            continue
        if math.isfinite(row["_acc"]):
            rows.append(row)

if not rows:
    raise SystemExit(f"[error] no successful epoch-sweep rows in {summary}")

needed = {(r["dataset"], r["shot"], r["seed"]) for r in rows}
by_epoch = defaultdict(list)
for r in rows:
    by_epoch[r["max_epoch"]].append(r)

def finite_mean(group, key):
    vals = [r[key] for r in group if math.isfinite(r[key])]
    return sum(vals) / len(vals) if vals else float("inf")

records = []
for epoch, group in by_epoch.items():
    present = {(r["dataset"], r["shot"], r["seed"]) for r in group}
    complete = present == needed
    n = len(group)
    records.append({
        "max_epoch": epoch,
        "num_results": n,
        "complete": complete,
        "mean_accuracy": sum(r["_acc"] for r in group) / n,
        "mean_ece": finite_mean(group, "_ece"),
        "mean_nll": finite_mean(group, "_nll"),
        "mean_brier": finite_mean(group, "_brier"),
    })

eligible = [r for r in records if r["complete"]] if strict else records
coverage_mode = "complete_only" if eligible else "incomplete_fallback"
if not eligible:
    eligible = records

best_acc = max(r["mean_accuracy"] for r in eligible)
acc_threshold = best_acc - acc_drop
window = [r for r in eligible if r["mean_accuracy"] >= acc_threshold]
if not window:
    window = eligible

window.sort(key=lambda r: (r["mean_ece"], -r["mean_accuracy"], r["mean_nll"], int(float(r["max_epoch"]))))
selected = window[0]

records.sort(key=lambda r: (
    0 if r in window else 1,
    r["mean_ece"] if r in window else float("inf"),
    -r["mean_accuracy"],
    r["mean_nll"],
    int(float(r["max_epoch"])),
))

fieldnames = [
    "selection_mode", "max_epoch", "num_results", "complete",
    "mean_accuracy", "best_accuracy", "acc_threshold", "acc_drop",
    "mean_ece", "mean_nll", "mean_brier",
]
out_csv.parent.mkdir(parents=True, exist_ok=True)

with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in records:
        row = {k: r.get(k, "") for k in fieldnames}
        row["selection_mode"] = (
            f"{coverage_mode}_acc_window_lowest_ece" if r is selected
            else ("acc_window_ranked" if r in window else "outside_acc_window")
        )
        row["best_accuracy"] = best_acc
        row["acc_threshold"] = acc_threshold
        row["acc_drop"] = acc_drop
        w.writerow(row)

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

env_path.write_text(
    "BEST_MAX_EPOCH=" + shq(selected["max_epoch"]) + "\n" +
    "BEST_MAX_EPOCH_SELECTION_MODE=" + shq(f"{coverage_mode}_acc_window_lowest_ece") + "\n" +
    f"BEST_MAX_EPOCH_BEST_ACC={best_acc:.6f}\n" +
    f"BEST_MAX_EPOCH_ACC_THRESHOLD={acc_threshold:.6f}\n" +
    f"BEST_MAX_EPOCH_ACC_DROP={acc_drop:.6f}\n" +
    f"BEST_MAX_EPOCH_MEAN_ACC={selected['mean_accuracy']:.6f}\n" +
    f"BEST_MAX_EPOCH_MEAN_ECE={selected['mean_ece']:.6f}\n",
    encoding="utf-8",
)

print(
    f"[select:max_epoch] mode={coverage_mode}_acc_window_lowest_ece "
    f"best={selected['max_epoch']} selected_acc={selected['mean_accuracy']:.4f} "
    f"selected_ece={selected['mean_ece']:.4f} best_acc={best_acc:.4f} "
    f"threshold={acc_threshold:.4f} acc_drop={acc_drop:.4f}"
)
print(f"[select:max_epoch] wrote {out_csv}")
print(f"[select:max_epoch] wrote {env_path}")
PY
}


select_or_write_best_epoch_env() {
  if [[ -n "${BEST_MAX_EPOCH:-}" ]]; then
    mkdir -p "$(dirname "${BEST_MAX_EPOCH_ENV}")"
    {
      printf "BEST_MAX_EPOCH='%s'\n" "${BEST_MAX_EPOCH}"
      printf "BEST_MAX_EPOCH_SELECTION_MODE='provided_by_user'\n"
    } > "${BEST_MAX_EPOCH_ENV}"
    echo "[select:max_epoch] using user-provided BEST_MAX_EPOCH=${BEST_MAX_EPOCH}"
  else
    select_max_epoch
  fi
}

load_best_epoch() {
  if [[ ! -f "${BEST_MAX_EPOCH_ENV}" ]]; then
    echo "[error] missing ${BEST_MAX_EPOCH_ENV}; run epoch sweep/selection first or pass BEST_MAX_EPOCH=..." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${BEST_MAX_EPOCH_ENV}"

  if [[ -z "${BEST_MAX_EPOCH:-}" ]]; then
    echo "[error] BEST_MAX_EPOCH is empty in ${BEST_MAX_EPOCH_ENV}" >&2
    exit 1
  fi
}

select_core_best() {
  python - <<PY
import csv
import math
from collections import defaultdict
from pathlib import Path

summary = Path(r"${CORE_SUMMARY}")
ranked_csv = Path(r"${CORE_RANKED}")
best_global_csv = Path(r"${CORE_BEST_GLOBAL}")
best_per_dataset_csv = Path(r"${CORE_BEST_PER_DATASET}")
env_path = Path(r"${BEST_CORE_ENV}")
strict = "${STRICT_COMPLETE_SELECTION}" == "1"
acc_drop = float("${ACC_DROP}")

if not summary.exists():
    raise SystemExit(f"[error] missing {summary}")

rows = []
with summary.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "ok":
            continue
        try:
            row["_acc"] = float(row.get("accuracy", "nan"))
            row["_ece"] = float(row.get("ece", "nan"))
            row["_nll"] = float(row.get("nll", "nan"))
            row["_brier"] = float(row.get("brier", "nan"))
        except Exception:
            continue
        if math.isfinite(row["_acc"]):
            rows.append(row)

if not rows:
    raise SystemExit(f"[error] no successful core-sweep rows in {summary}")

needed = {(r["dataset"], r["shot"], r["seed"]) for r in rows}
by_tag = defaultdict(list)
for r in rows:
    by_tag[r["tag"]].append(r)

def safe_mean(group, key):
    vals = [r[key] for r in group if math.isfinite(r[key])]
    return sum(vals) / len(vals) if vals else float("inf")

def make_record(tag, group, needed_pairs, dataset_name=""):
    present = {(r["dataset"], r["shot"], r["seed"]) for r in group}
    complete = present == needed_pairs
    n = len(group)
    first = group[0]
    rec = {
        "tag": tag,
        "max_epoch": first.get("max_epoch", ""),
        "r_prior_std": first.get("r_prior_std", ""),
        "r_kl_weight": first.get("r_kl_weight", ""),
        "t_prior_std": first.get("t_prior_std", ""),
        "t_kl_weight": first.get("t_kl_weight", ""),
        "fusion": first.get("fusion", ""),
        "num_results": n,
        "complete": complete,
        "mean_accuracy": safe_mean(group, "_acc"),
        "mean_ece": safe_mean(group, "_ece"),
        "mean_nll": safe_mean(group, "_nll"),
        "mean_brier": safe_mean(group, "_brier"),
    }
    if dataset_name:
        rec["dataset"] = dataset_name
    return rec

def select_acc_window_lowest_ece(records):
    eligible = [r for r in records if r["complete"]] if strict else records
    coverage_mode = "complete_only" if eligible else "incomplete_fallback"
    if not eligible:
        eligible = records

    best_acc = max(r["mean_accuracy"] for r in eligible)
    acc_threshold = best_acc - acc_drop
    window = [r for r in eligible if r["mean_accuracy"] >= acc_threshold]
    if not window:
        window = eligible

    window.sort(key=lambda r: (r["mean_ece"], -r["mean_accuracy"], r["mean_nll"], r["tag"]))
    selected = window[0]

    ranked = list(records)
    ranked.sort(key=lambda r: (
        0 if r in window else 1,
        r["mean_ece"] if r in window else float("inf"),
        -r["mean_accuracy"],
        r["mean_nll"],
        r["tag"],
    ))

    for r in ranked:
        r["best_accuracy"] = best_acc
        r["acc_threshold"] = acc_threshold
        r["acc_drop"] = acc_drop
        r["in_acc_window"] = r in window
        r["selection_mode"] = (
            f"{coverage_mode}_acc_window_lowest_ece" if r is selected
            else ("acc_window_ranked" if r in window else "outside_acc_window")
        )
    return selected, ranked, coverage_mode, best_acc, acc_threshold

records = [make_record(tag, group, needed) for tag, group in by_tag.items()]
selected, ranked_records, selection_mode, best_acc, acc_threshold = select_acc_window_lowest_ece(records)

fieldnames = [
    "selection_mode", "tag", "max_epoch", "r_prior_std", "r_kl_weight",
    "t_prior_std", "t_kl_weight", "fusion", "num_results", "complete",
    "mean_accuracy", "best_accuracy", "acc_threshold", "acc_drop",
    "in_acc_window", "mean_ece", "mean_nll", "mean_brier",
]

ranked_csv.parent.mkdir(parents=True, exist_ok=True)
with ranked_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in ranked_records:
        w.writerow({k: r.get(k, "") for k in fieldnames})

with best_global_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow({k: selected.get(k, "") for k in fieldnames})

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

env_lines = [
    f"BEST_CORE_TAG={shq(selected.get('tag', ''))}",
    f"BEST_CORE_MAX_EPOCH={shq(selected.get('max_epoch', ''))}",
    f"BEST_CORE_R_PRIOR_STD={shq(selected.get('r_prior_std', ''))}",
    f"BEST_CORE_R_KL_WEIGHT={shq(selected.get('r_kl_weight', ''))}",
    f"BEST_CORE_T_PRIOR_STD={shq(selected.get('t_prior_std', ''))}",
    f"BEST_CORE_T_KL_WEIGHT={shq(selected.get('t_kl_weight', ''))}",
    f"BEST_CORE_FUSION={shq(selected.get('fusion', ''))}",
    f"BEST_CORE_SELECTION_MODE={shq(selection_mode + '_acc_window_lowest_ece')}",
    f"BEST_CORE_BEST_ACC={best_acc:.6f}",
    f"BEST_CORE_ACC_THRESHOLD={acc_threshold:.6f}",
    f"BEST_CORE_ACC_DROP={acc_drop:.6f}",
    f"BEST_CORE_MEAN_ACC={selected['mean_accuracy']:.6f}",
    f"BEST_CORE_MEAN_ECE={selected['mean_ece']:.6f}",
]
env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

per_dataset_rows = []
by_dataset = defaultdict(list)
for r in rows:
    by_dataset[r["dataset"]].append(r)

per_dataset_fieldnames = ["dataset"] + fieldnames
for dataset, dataset_rows in sorted(by_dataset.items()):
    needed_ds = {(dataset, r["shot"], r["seed"]) for r in dataset_rows}
    by_tag_ds = defaultdict(list)
    for r in dataset_rows:
        by_tag_ds[r["tag"]].append(r)

    ds_records = [
        make_record(tag, group, needed_ds, dataset_name=dataset)
        for tag, group in by_tag_ds.items()
    ]
    if not ds_records:
        continue

    ds_selected, _ds_ranked, ds_mode, ds_best_acc, ds_acc_threshold = select_acc_window_lowest_ece(ds_records)
    ds_selected["dataset"] = dataset
    ds_selected["selection_mode"] = f"dataset_{ds_mode}_acc_window_lowest_ece"
    ds_selected["best_accuracy"] = ds_best_acc
    ds_selected["acc_threshold"] = ds_acc_threshold
    ds_selected["acc_drop"] = acc_drop
    ds_selected["in_acc_window"] = True
    per_dataset_rows.append(ds_selected)

with best_per_dataset_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=per_dataset_fieldnames)
    w.writeheader()
    for r in per_dataset_rows:
        w.writerow({k: r.get(k, "") for k in per_dataset_fieldnames})

print(
    f"[select:core] mode={selection_mode}_acc_window_lowest_ece "
    f"best_tag={selected['tag']} selected_acc={selected['mean_accuracy']:.4f} "
    f"selected_ece={selected['mean_ece']:.4f} best_acc={best_acc:.4f} "
    f"threshold={acc_threshold:.4f} acc_drop={acc_drop:.4f}"
)
print(f"[select:core] wrote {ranked_csv}")
print(f"[select:core] wrote {best_global_csv}")
print(f"[select:core] wrote {best_per_dataset_csv}")
print(f"[select:core] wrote {env_path}")
PY
}


load_best_core_global() {
  if [[ ! -f "${BEST_CORE_ENV}" ]]; then
    echo "[error] missing ${BEST_CORE_ENV}; run core sweep/selection first or set RUN_CONFIRM_FULL=0" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${BEST_CORE_ENV}"

  if [[ -z "${BEST_CORE_MAX_EPOCH:-}" || -z "${BEST_CORE_TAG:-}" ]]; then
    echo "[error] BEST_CORE_ENV is incomplete: ${BEST_CORE_ENV}" >&2
    exit 1
  fi
}

get_best_core_for_dataset() {
  local dataset=$1
  python - "${dataset}" <<PY
import csv
import sys
from pathlib import Path

dataset = sys.argv[1]
mode = "${CONFIRM_BEST_MODE}"
global_csv = Path(r"${CORE_BEST_GLOBAL}")
per_dataset_csv = Path(r"${CORE_BEST_PER_DATASET}")

def read_first(path, predicate=lambda r: True):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if predicate(row):
                return row
    return None

row = None
if mode == "per_dataset":
    row = read_first(per_dataset_csv, lambda r: r.get("dataset") == dataset)
if row is None:
    row = read_first(global_csv)

if row is None:
    raise SystemExit(f"[error] no selected core config found for dataset={dataset}")

fields = [
    "tag",
    "max_epoch",
    "r_prior_std",
    "r_kl_weight",
    "t_prior_std",
    "t_kl_weight",
    "fusion",
]
for field in fields:
    print(str(row.get(field, "")))
PY
}

run_confirm_stage() {
  local dataset shot seed best_tag best_epoch r_prior r_kl t_prior t_kl fusion tag
  local best_fields

  echo "[stage3] full confirm with selected core hyperparameters"
  echo "[stage3] CONFIRM_DATASETS=${CONFIRM_DATASETS}"
  echo "[stage3] CONFIRM_SHOTS=${CONFIRM_SHOTS}"
  echo "[stage3] CONFIRM_SEEDS=${CONFIRM_SEEDS}"
  echo "[stage3] CONFIRM_BEST_MODE=${CONFIRM_BEST_MODE}"

  load_best_core_global

  for dataset in ${CONFIRM_DATASETS}; do
    mapfile -t best_fields < <(get_best_core_for_dataset "${dataset}")
    best_tag="${best_fields[0]}"
    best_epoch="${best_fields[1]}"
    r_prior="${best_fields[2]}"
    r_kl="${best_fields[3]}"
    t_prior="${best_fields[4]}"
    t_kl="${best_fields[5]}"
    fusion="${best_fields[6]}"

    if [[ -z "${best_epoch}" || -z "${r_prior}" || -z "${r_kl}" || -z "${t_prior}" || -z "${t_kl}" || -z "${fusion}" ]]; then
      echo "[error] incomplete selected config for dataset=${dataset}: tag=${best_tag}" >&2
      exit 1
    fi

    echo "[stage3] dataset=${dataset} selected_tag=${best_tag} epoch=${best_epoch} r_prior=${r_prior} r_kl=${r_kl} t_prior=${t_prior} t_kl=${t_kl} fusion=${fusion}"

    for shot in ${CONFIRM_SHOTS}; do
      for seed in ${CONFIRM_SEEDS}; do
        tag="$(sanitize "detbayesrt_confirm_epoch-${best_epoch}_rpstd-${r_prior}_rkl-${r_kl}_tpstd-${t_prior}_tkl-${t_kl}_fusion-${fusion}")"
        submit_case "confirm" "${dataset}" "${shot}" "${seed}" "${tag}" "${best_epoch}" "${r_prior}" "${r_kl}" "${t_prior}" "${t_kl}" "${fusion}" "${best_tag}" \
          BAYESRT_MMRL.R_PRIOR_STD "${r_prior}" \
          BAYESRT_MMRL.R_KL_WEIGHT "${r_kl}" \
          BAYESRT_MMRL.T_PRIOR_STD "${t_prior}" \
          BAYESRT_MMRL.T_KL_WEIGHT "${t_kl}" \
          BAYESRT_MMRL.EVAL_FUSION_VARIANT "${fusion}"
      done
    done
  done

  wait_all_jobs
}

combine_summaries() {
  python - <<PY
import csv
from pathlib import Path

paths = [Path(r"${EPOCH_SUMMARY}"), Path(r"${CORE_SUMMARY}"), Path(r"${CONFIRM_SUMMARY}")]
out = Path(r"${ALL_SUMMARY}")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
fieldnames = None

for path in paths:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)

if fieldnames is None:
    fieldnames = [
        "stage", "dataset", "shot", "seed", "tag", "max_epoch",
        "r_prior_std", "r_kl_weight", "t_prior_std", "t_kl_weight", "fusion",
        "selected_from_tag", "outdir", "status", "num_samples", "accuracy", "error", "macro_f1",
        "ece", "nll", "brier", "metrics_path",
    ]

with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in fieldnames})

print(f"[summary] wrote {out} rows={len(rows)}")
PY
}

run_epoch_stage() {
  local dataset shot seed epoch tag

  echo "[stage1] sweep MAX_EPOCH with default DetBayesRTMMRL hyperparameters: ${MAX_EPOCH_LIST}"

  for dataset in ${TUNE_DATASETS}; do
    for shot in ${TUNE_SHOTS}; do
      for seed in ${TUNE_SEEDS}; do
        for epoch in ${MAX_EPOCH_LIST}; do
          tag="$(sanitize "detbayesrt_epoch-${epoch}_defaults")"
          submit_case "epoch" "${dataset}" "${shot}" "${seed}" "${tag}" "${epoch}" "" "" "" "" "" ""
        done
      done
    done
  done

  wait_all_jobs
}

run_core_stage() {
  load_best_epoch
  local dataset shot seed r_prior r_kl t_prior t_kl fusion tag

  echo "[stage2] sweep core DetBayesRTMMRL hyperparameters with BEST_MAX_EPOCH=${BEST_MAX_EPOCH}"

  for dataset in ${TUNE_DATASETS}; do
    for shot in ${TUNE_SHOTS}; do
      for seed in ${TUNE_SEEDS}; do
        for r_prior in ${DET_R_PRIOR_STD_LIST}; do
          for r_kl in ${DET_R_KL_WEIGHT_LIST}; do
            for t_prior in ${DET_T_PRIOR_STD_LIST}; do
              for t_kl in ${DET_T_KL_WEIGHT_LIST}; do
                for fusion in ${DET_EVAL_FUSION_VARIANT_LIST}; do
                  tag="$(sanitize "detbayesrt_epoch-${BEST_MAX_EPOCH}_rpstd-${r_prior}_rkl-${r_kl}_tpstd-${t_prior}_tkl-${t_kl}_fusion-${fusion}")"
                  submit_case "core" "${dataset}" "${shot}" "${seed}" "${tag}" "${BEST_MAX_EPOCH}" "${r_prior}" "${r_kl}" "${t_prior}" "${t_kl}" "${fusion}" "" \
                    BAYESRT_MMRL.R_PRIOR_STD "${r_prior}" \
                    BAYESRT_MMRL.R_KL_WEIGHT "${r_kl}" \
                    BAYESRT_MMRL.T_PRIOR_STD "${t_prior}" \
                    BAYESRT_MMRL.T_KL_WEIGHT "${t_kl}" \
                    BAYESRT_MMRL.EVAL_FUSION_VARIANT "${fusion}"
                done
              done
            done
          done
        done
      done
    done
  done

  wait_all_jobs
}

print_config() {
  local epoch_cases core_cases datasets_n shots_n seeds_n
  epoch_cases=$(python - <<PY
print(len("${MAX_EPOCH_LIST}".split()))
PY
)
  core_cases=$(python - <<PY
print(len("${DET_R_PRIOR_STD_LIST}".split()) * len("${DET_R_KL_WEIGHT_LIST}".split()) * len("${DET_T_PRIOR_STD_LIST}".split()) * len("${DET_T_KL_WEIGHT_LIST}".split()) * len("${DET_EVAL_FUSION_VARIANT_LIST}".split()))
PY
)
  datasets_n=$(python - <<PY
print(len("${TUNE_DATASETS}".split()))
PY
)
  shots_n=$(python - <<PY
print(len("${TUNE_SHOTS}".split()))
PY
)
  seeds_n=$(python - <<PY
print(len("${TUNE_SEEDS}".split()))
PY
)

  echo "[config] PROJECT_DIR=${PROJECT_DIR}"
  echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[config] METHOD=${METHOD} METHOD_CFG=${METHOD_CFG}"
  echo "[config] RUNTIME_CFG=${RUNTIME_CFG} PROTOCOL_CFG=${PROTOCOL_CFG}"
  echo "[config] TUNE_DATASETS=${TUNE_DATASETS}"
  echo "[config] TUNE_SHOTS=${TUNE_SHOTS} TUNE_SEEDS=${TUNE_SEEDS}"
  echo "[config] stage1 epoch candidates per dataset/shot/seed=${epoch_cases}"
  echo "[config] stage2 core candidates per dataset/shot/seed=${core_cases}"
  local confirm_datasets_n confirm_shots_n confirm_seeds_n
  confirm_datasets_n=$(python - <<PY
print(len("${CONFIRM_DATASETS}".split()))
PY
)
  confirm_shots_n=$(python - <<PY
print(len("${CONFIRM_SHOTS}".split()))
PY
)
  confirm_seeds_n=$(python - <<PY
print(len("${CONFIRM_SEEDS}".split()))
PY
)

  echo "[config] confirm datasets=${CONFIRM_DATASETS}"
  echo "[config] confirm shots=${CONFIRM_SHOTS} confirm seeds=${CONFIRM_SEEDS} mode=${CONFIRM_BEST_MODE}"
  echo "[config] estimated sweep jobs=$((datasets_n * shots_n * seeds_n * (epoch_cases + core_cases)))"
  echo "[config] estimated confirm jobs=$((confirm_datasets_n * confirm_shots_n * confirm_seeds_n))"
  echo "[config] estimated total jobs=$((datasets_n * shots_n * seeds_n * (epoch_cases + core_cases) + confirm_datasets_n * confirm_shots_n * confirm_seeds_n))"
  echo "[config] ACC_DROP=${ACC_DROP} (selection: candidates within best_acc-ACC_DROP, then lowest ECE)"
echo "[config] HPO.ENABLED is forced false for every run"
  echo "[config] cleanup after result json: refactor_model/tensorboard=${DELETE_ARTIFACTS_AFTER_TEST}, weight_files=${DELETE_WEIGHT_FILES_AFTER_TEST}"
}

summary_only_main() {
  summarize_manifest "epoch" "${EPOCH_SUMMARY}"

  if [[ -n "${BEST_MAX_EPOCH:-}" || -s "${EPOCH_SUMMARY}" ]]; then
    select_or_write_best_epoch_env
  fi

  summarize_manifest "core" "${CORE_SUMMARY}"

  if [[ -s "${CORE_SUMMARY}" ]]; then
    select_core_best || true
  fi

  summarize_manifest "confirm" "${CONFIRM_SUMMARY}"
  combine_summaries
}

main() {
  cd "${PROJECT_DIR}"

  if [[ ! -f "run.py" ]]; then
    echo "[error] run.py not found in PROJECT_DIR=${PROJECT_DIR}. Put this script in MMRL/ or set PROJECT_DIR=/path/to/MMRL." >&2
    exit 1
  fi

  if [[ ! -f "${METHOD_CFG}" ]]; then
    echo "[error] method config not found: ${METHOD_CFG}" >&2
    exit 1
  fi

  mkdir -p "${OUTPUT_ROOT}"
  reset_index_files_if_requested
  print_config

  if [[ "${SUMMARY_ONLY}" == "1" ]]; then
    summary_only_main
    exit 0
  fi

  init_gpu_list
  init_slots
  trap 'echo "[INTERRUPT] stopping child jobs..."; cleanup_children; exit 130' INT TERM

  if [[ "${RUN_EPOCH_SWEEP}" == "1" ]]; then
    run_epoch_stage
  else
    echo "[stage1] skipped because RUN_EPOCH_SWEEP=${RUN_EPOCH_SWEEP}"
  fi

  summarize_manifest "epoch" "${EPOCH_SUMMARY}"
  select_or_write_best_epoch_env

  if [[ "${RUN_CORE_SWEEP}" == "1" ]]; then
    run_core_stage
  else
    echo "[stage2] skipped because RUN_CORE_SWEEP=${RUN_CORE_SWEEP}"
  fi

  summarize_manifest "core" "${CORE_SUMMARY}"

  if [[ "${RUN_CORE_SWEEP}" == "1" || -s "${CORE_SUMMARY}" ]]; then
    select_core_best
  fi

  if [[ "${RUN_CONFIRM_FULL}" == "1" ]]; then
    run_confirm_stage
  else
    echo "[stage3] skipped because RUN_CONFIRM_FULL=${RUN_CONFIRM_FULL}"
  fi

  summarize_manifest "confirm" "${CONFIRM_SUMMARY}"
  combine_summaries

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)." >&2
    exit 1
  fi

  echo "[DONE] DetBayesRTMMRL two-stage sweep finished."
}

main "$@"
