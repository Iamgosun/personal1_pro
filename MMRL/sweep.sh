#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-DATASETS}
DATASET=${DATASET:-caltech101}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}
SHOTS=${SHOTS:-1 4 16}
SEEDS=${SEEDS:-"1 "}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
TAG_PREFIX=${TAG_PREFIX:-bayes_sweep}

# 参考 run_plan.sh 的 GPU 调度方式
NGPU=${NGPU:-2}
GPU_IDS=${GPU_IDS:-"0 1"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

DATASET_CFG="configs/datasets/${DATASET}.yaml"
METHOD_CFG="configs/methods/bayesmmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

resolve_phase_semantics() {
  case "$1" in
    B2N) echo "train_base base configs/protocols/b2n.yaml" ;;
    FS)  echo "fewshot_train all configs/protocols/fs.yaml" ;;
    CD)  echo "cross_train all configs/protocols/cd.yaml" ;;
    *) echo "Unknown PROTOCOL=$1" >&2; exit 1 ;;
  esac
}

read -r PHASE SUBSAMPLE PROTOCOL_CFG <<< "$(resolve_phase_semantics "$PROTOCOL")"

BACKBONE_TAG="${BACKBONE//\//-}"
METHOD_ROOT="${OUTPUT_ROOT}/BayesMMRL/${PROTOCOL}/${PHASE}/${DATASET}/shots_${SHOTS}/${BACKBONE_TAG}"
GLOBAL_SUMMARY="${METHOD_ROOT}/sweep_summary.csv"

mkdir -p "${METHOD_ROOT}"

init_gpu_list() {
  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a GPU_LIST <<< "${GPU_IDS}"
  else
    GPU_LIST=()
    local i
    for ((i=0; i<NGPU; i++)); do
      GPU_LIST+=("$i")
    done
  fi

  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "No GPU ids resolved. Set NGPU or GPU_IDS." >&2
    exit 1
  fi
}

build_outdir() {
  local tag=$1
  local seed=$2
  echo "${METHOD_ROOT}/${tag}/seed${seed}"
}

write_log_header() {
  local logfile=$1
  local gpu_id=$2
  local seed=$3
  local alpha=$4
  local kl=$5
  local prior=$6
  local tag=$7

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "GPU: ${gpu_id}"
    echo "METHOD: BayesMMRL"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${DATASET}"
    echo "SHOTS: ${SHOTS}"
    echo "SEED: ${seed}"
    echo "ALPHA: ${alpha}"
    echo "KL_WEIGHT: ${kl}"
    echo "PRIOR_STD: ${prior}"
    echo "TAG: ${tag}"
    echo "DATA_ROOT: ${ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "============================================================"
  } >> "${logfile}"
}

launch_one_case() {
  local gpu_id=$1
  local seed=$2
  local alpha=$3
  local kl=$4
  local prior=$5
  local tag=$6

  local outdir logfile statusfile
  outdir="$(build_outdir "${tag}" "${seed}")"
  logfile="${outdir}/run.log"
  statusfile="${outdir}/job_status.txt"

  mkdir -p "${outdir}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
    echo "SKIP" > "${statusfile}"
    return 0
  fi

  : > "${logfile}"
  write_log_header "${logfile}" "${gpu_id}" "${seed}" "${alpha}" "${kl}" "${prior}" "${tag}"

  if CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
      --root "${ROOT}" \
      --dataset-config-file "${DATASET_CFG}" \
      --method-config-file "${METHOD_CFG}" \
      --protocol-config-file "${PROTOCOL_CFG}" \
      --runtime-config-file "${RUNTIME_CFG}" \
      --output-dir "${outdir}" \
      --method BayesMMRL \
      --protocol "${PROTOCOL}" \
      --exec-mode "${EXEC_MODE}" \
      --seed "${seed}" \
      DATASET.NUM_SHOTS "${SHOTS}" \
      DATASET.SUBSAMPLE_CLASSES "${SUBSAMPLE}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      BAYES_MMRL.ALPHA "${alpha}" \
      BAYES_MMRL.KL_WEIGHT "${kl}" \
      BAYES_MMRL.PRIOR_STD "${prior}" \
      >> "${logfile}" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "============================================================"
    } >> "${logfile}"
    echo "SUCCESS" > "${statusfile}"
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

append_summary_to_global() {
  local tag=$1
  local alpha=$2
  local kl=$3
  local prior=$4
  local summary_csv=$5

  local header
  header="$(head -n 1 "${summary_csv}")"

  if [[ ! -f "${GLOBAL_SUMMARY}" ]]; then
    echo "tag,alpha,kl_weight,prior_std,${header}" > "${GLOBAL_SUMMARY}"
  fi

  tail -n +2 "${summary_csv}" | while IFS= read -r line; do
    echo "${tag},${alpha},${kl},${prior},${line}" >> "${GLOBAL_SUMMARY}"
  done
}

summarize_tag() {
  local tag=$1
  local alpha=$2
  local kl=$3
  local prior=$4
  local tag_root="${METHOD_ROOT}/${tag}"

  python evaluation/result_parser.py "${tag_root}" --split test || true

  local summary_csv="${tag_root}/test_summary.csv"
  if [[ ! -f "${summary_csv}" ]]; then
    echo "[WARN] summary file not found: ${summary_csv}"
    return
  fi

  append_summary_to_global "${tag}" "${alpha}" "${kl}" "${prior}" "${summary_csv}"
}

cleanup_children() {
  local p
  for p in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${p:-}" ]] && kill -0 "${p}" 2>/dev/null; then
      kill "${p}" 2>/dev/null || true
    fi
  done
}

print_finish_status() {
  local rc=$1
  local gpu_id=$2
  local seed=$3
  local alpha=$4
  local kl=$5
  local prior=$6
  local logfile=$7

  if [[ "${rc}" -eq 0 ]]; then
    echo "[OK]   gpu=${gpu_id} seed=${seed} alpha=${alpha} kl=${kl} prior=${prior}" >&2
  else
    echo "[FAIL] gpu=${gpu_id} seed=${seed} alpha=${alpha} kl=${kl} prior=${prior} log=${logfile}" >&2
  fi
}

READY_SLOT=""

wait_for_any_slot() {
  READY_SLOT=""

  while true; do
    local idx
    for idx in "${!RUNNING_PIDS[@]}"; do
      local pid="${RUNNING_PIDS[$idx]}"

      if [[ -z "${pid}" ]]; then
        READY_SLOT="${idx}"
        return 0
      fi

      if ! kill -0 "${pid}" 2>/dev/null; then
        local rc=0
        if wait "${pid}"; then
          rc=0
        else
          rc=$?
        fi

        local gpu_id="${SLOT_GPU[$idx]}"
        local seed="${SLOT_SEED[$idx]}"
        local alpha="${SLOT_ALPHA[$idx]}"
        local kl="${SLOT_KL[$idx]}"
        local prior="${SLOT_PRIOR[$idx]}"
        local logfile="${SLOT_LOG[$idx]}"

        print_finish_status "${rc}" "${gpu_id}" "${seed}" "${alpha}" "${kl}" "${prior}" "${logfile}"

        if [[ "${rc}" -ne 0 ]]; then
          FAILED_JOBS=$((FAILED_JOBS + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_GPU[$idx]=""
        SLOT_SEED[$idx]=""
        SLOT_ALPHA[$idx]=""
        SLOT_KL[$idx]=""
        SLOT_PRIOR[$idx]=""
        SLOT_LOG[$idx]=""

        READY_SLOT="${idx}"
        return 0
      fi
    done

    sleep "${SLEEP_SEC}"
  done
}

wait_all_jobs() {
  local idx
  for idx in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$idx]}"
    if [[ -n "${pid}" ]]; then
      local rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      local gpu_id="${SLOT_GPU[$idx]}"
      local seed="${SLOT_SEED[$idx]}"
      local alpha="${SLOT_ALPHA[$idx]}"
      local kl="${SLOT_KL[$idx]}"
      local prior="${SLOT_PRIOR[$idx]}"
      local logfile="${SLOT_LOG[$idx]}"

      print_finish_status "${rc}" "${gpu_id}" "${seed}" "${alpha}" "${kl}" "${prior}" "${logfile}"

      if [[ "${rc}" -ne 0 ]]; then
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_SEED[$idx]=""
      SLOT_ALPHA[$idx]=""
      SLOT_KL[$idx]=""
      SLOT_PRIOR[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}

declare -a TAG_LIST
declare -A TAG_ALPHA
declare -A TAG_KL
declare -A TAG_PRIOR

register_tag() {
  local alpha=$1
  local kl=$2
  local prior=$3
  local tag="${TAG_PREFIX}_alpha_${alpha}_kl_${kl}_prior_${prior}"

  if [[ -z "${TAG_ALPHA[$tag]+x}" ]]; then
    TAG_LIST+=("${tag}")
    TAG_ALPHA["$tag"]="${alpha}"
    TAG_KL["$tag"]="${kl}"
    TAG_PRIOR["$tag"]="${prior}"
  fi
}

enqueue_rounds() {
  BASE_KL=1e-5
  BASE_PRIOR=0.1

  for alpha in 0.0, 0.3, 0.5, 0.7, 1.0; do
    register_tag "${alpha}" "${BASE_KL}" "${BASE_PRIOR}"
  done

  BEST_ALPHA=${BEST_ALPHA:-0.6}
  for kl in 5e-6 1e-5 5e-5 1e-4 5e-4; do
    register_tag "${BEST_ALPHA}" "${kl}" "${BASE_PRIOR}"
  done

  BEST_KL=${BEST_KL:-5e-5}
  for prior in 0.02 0.05 0.1 0.2; do
    register_tag "${BEST_ALPHA}" "${BEST_KL}" "${prior}"
  done
}

print_global_summary() {
  echo "========================================"
  echo "Global sweep summary saved to:"
  echo "${GLOBAL_SUMMARY}"
  echo "========================================"

  if [[ -f "${GLOBAL_SUMMARY}" ]]; then
    python - <<PY
import pandas as pd
from pathlib import Path

path = Path(r"${GLOBAL_SUMMARY}")
df = pd.read_csv(path)

preferred_cols = [
    "tag",
    "alpha",
    "kl_weight",
    "prior_std",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "ece_mean",
    "ece_std",
    "brier_mean",
    "brier_std",
]

keep = [c for c in preferred_cols if c in df.columns]
if keep:
    sort_col = keep[4] if len(keep) > 4 else keep[0]
    print(df[keep].sort_values(by=sort_col, ascending=False).to_string(index=False))
else:
    print(df.to_string(index=False))
PY
  fi
}

main() {
  init_gpu_list
  enqueue_rounds

  declare -ga RUNNING_PIDS
  declare -ga SLOT_GPU
  declare -ga SLOT_SEED
  declare -ga SLOT_ALPHA
  declare -ga SLOT_KL
  declare -ga SLOT_PRIOR
  declare -ga SLOT_LOG

  FAILED_JOBS=0

  local nslots=${#GPU_LIST[@]}
  local i
  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_SEED[$i]=""
    SLOT_ALPHA[$i]=""
    SLOT_KL[$i]=""
    SLOT_PRIOR[$i]=""
    SLOT_LOG[$i]=""
  done

  trap 'echo "[INTERRUPT] stopping child jobs..."; cleanup_children; exit 130' INT TERM

  local tag seed outdir logfile statusfile slot gpu_id alpha kl prior
  for tag in "${TAG_LIST[@]}"; do
    alpha="${TAG_ALPHA[$tag]}"
    kl="${TAG_KL[$tag]}"
    prior="${TAG_PRIOR[$tag]}"

    for seed in ${SEEDS}; do
      outdir="$(build_outdir "${tag}" "${seed}")"
      logfile="${outdir}/run.log"
      statusfile="${outdir}/job_status.txt"

      if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
        mkdir -p "${outdir}"
        echo "SKIP" > "${statusfile}"
        echo "[SKIP] seed=${seed} alpha=${alpha} kl=${kl} prior=${prior}"
        continue
      fi

      wait_for_any_slot
      slot="${READY_SLOT}"
      gpu_id="${GPU_LIST[$slot]}"

      (
        launch_one_case "${gpu_id}" "${seed}" "${alpha}" "${kl}" "${prior}" "${tag}"
      ) &
      RUNNING_PIDS[$slot]=$!
      SLOT_GPU[$slot]="${gpu_id}"
      SLOT_SEED[$slot]="${seed}"
      SLOT_ALPHA[$slot]="${alpha}"
      SLOT_KL[$slot]="${kl}"
      SLOT_PRIOR[$slot]="${prior}"
      SLOT_LOG[$slot]="${logfile}"
    done
  done

  wait_all_jobs

  # 串行汇总，避免并发写 GLOBAL_SUMMARY
  rm -f "${GLOBAL_SUMMARY}"
  for tag in "${TAG_LIST[@]}"; do
    summarize_tag "${tag}" "${TAG_ALPHA[$tag]}" "${TAG_KL[$tag]}" "${TAG_PRIOR[$tag]}"
  done

  print_global_summary

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)."
    exit 1
  fi

  echo "[DONE] all jobs finished successfully."
}

main "$@"