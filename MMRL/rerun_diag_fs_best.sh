#!/bin/bash
set -euo pipefail

# ============================================================
# Rerun BayesMMRL rep_tokens Diag mode on FS three datasets
# using best_config_per_dataset_sigma.csv.
#
# Important:
#   - This script DOES NOT delete checkpoints / weights.
#   - No rm -rf refactor_model.
#   - Default output path is a new rerun stage to avoid overwriting
#     existing confirm_stage results.
#
# Usage:
#   cd MMRL
#   bash rerun_diag_fs_best.sh GPU_IDS="0 1 2 3" JOBS_PER_GPU=2
#
# Optional:
#   DATA_ROOT=DATASETS
#   OUTPUT_ROOT=output_sweeps/bayes_mmrl_rep_tokens/coarse3/diag_rerun_stage
#   SHOTS="1 2 4 8 16 32"
#   SEEDS="1 2 3"
#   SKIP_EXISTING=1
# ============================================================

apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      case "${key}" in
        DATA_ROOT|OUTPUT_ROOT|GPU_IDS|JOBS_PER_GPU|SHOTS|SEEDS|SKIP_EXISTING|SLEEP_SEC|BACKBONE|EXEC_MODE)
          printf -v "${key}" '%s' "${val}"
          export "${key}"
          ;;
        *)
          echo "[warn] unknown KEY=VALUE ignored: ${arg}" >&2
          ;;
      esac
    fi
  done
}

apply_kv_args "$@"

DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor/diag_rerun_stage}
GPU_IDS=${GPU_IDS:-0 1 2}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
SHOTS=${SHOTS:-"1 2 4 8 16 32"}
SEEDS=${SEEDS:-"1 2 3"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

BACKBONE=${BACKBONE:-ViT-B/16}
EXEC_MODE=${EXEC_MODE:-online}

METHOD=BayesMMRL
PROTOCOL=FS
PHASE=fewshot_train
SUBSAMPLE=all

METHOD_CFG=configs/methods/bayesmmrl.yaml
PROTOCOL_CFG=configs/protocols/fs.yaml
RUNTIME_CFG=configs/runtime/default.yaml

DATASETS="caltech101 oxford_pets ucf101"

# Common BayesMMRL settings used in the previous confirm-stage logs.
COMMON_OPTS=(
  BAYES_MMRL.ALPHA 0.7
  BAYES_MMRL.REG_WEIGHT 0.5
  BAYES_MMRL.N_REP_TOKENS 5
  BAYES_MMRL.REP_LAYERS "[6,7,8,9,10,11,12]"
  BAYES_MMRL.REP_DIM 512
  BAYES_MMRL.N_MC_TRAIN 3
  BAYES_MMRL.N_MC_TEST 10
  BAYES_MMRL.EVAL_MODE mc_predictive
  BAYES_MMRL.EVAL_USE_POSTERIOR_MEAN False
  BAYES_MMRL.EVAL_AGGREGATION logit_mean
  BAYES_MMRL.KL_WARMUP_EPOCHS 6
  BAYES_MMRL.BAYES_TARGET rep_tokens
  BAYES_MMRL.REP_PRIOR_MODE zero
  BAYES_MMRL.REP_SIGMA_MODE diagonal
)

get_diag_config() {
  local dataset="$1"

  case "${dataset}" in
    caltech101)
      echo "rep_zero_sig-diagonal_pstd-1.0_kl-5e-2 1.0 5e-2"
      ;;
    oxford_pets)
      echo "rep_zero_sig-diagonal_pstd-0.1_kl-5e-2 0.1 5e-2"
      ;;
    ucf101)
      echo "rep_zero_sig-diagonal_pstd-0.5_kl-5e-2 0.5 5e-2"
      ;;
    *)
      echo "unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
}

sanitize_backbone() {
  local s="$1"
  s="${s//\//-}"
  echo "${s}"
}

build_outdir() {
  local dataset="$1"
  local shot="$2"
  local tag="$3"
  local seed="$4"
  local backbone_tag
  backbone_tag="$(sanitize_backbone "${BACKBONE}")"

  echo "${OUTPUT_ROOT}/${METHOD}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/${backbone_tag}/${tag}/seed${seed}"
}

write_log_header() {
  local logfile="$1"
  local gpu="$2"
  local dataset="$3"
  local shot="$4"
  local seed="$5"
  local tag="$6"
  local prior_std="$7"
  local kl="$8"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: diag_rerun_best_from_csv"
    echo "GPU: ${gpu}"
    echo "METHOD: ${METHOD}"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "PHASE: ${PHASE}"
    echo "SUBSAMPLE: ${SUBSAMPLE}"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "REP_SIGMA_MODE: diagonal"
    echo "REP_PRIOR_STD: ${prior_std}"
    echo "REP_KL_WEIGHT: ${kl}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "IMPORTANT: checkpoints are kept; this script never deletes refactor_model"
    echo "============================================================"
  } >> "${logfile}"
}

launch_one() {
  local gpu="$1"
  local dataset="$2"
  local shot="$3"
  local seed="$4"

  local tag prior_std kl
  read -r tag prior_std kl <<< "$(get_diag_config "${dataset}")"

  local outdir logfile statusfile
  outdir="$(build_outdir "${dataset}" "${shot}" "${tag}" "${seed}")"
  logfile="${outdir}/run.log"
  statusfile="${outdir}/job_status.txt"

  mkdir -p "${outdir}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
    echo "[skip] dataset=${dataset} shot=${shot} seed=${seed} tag=${tag}"
    echo "SKIP" > "${statusfile}"
    return 0
  fi

  : > "${logfile}"
  write_log_header "${logfile}" "${gpu}" "${dataset}" "${shot}" "${seed}" "${tag}" "${prior_std}" "${kl}"

  echo "[run] gpu=${gpu} dataset=${dataset} shot=${shot} seed=${seed} tag=${tag}"
  echo "      log=${logfile}"

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
      "${COMMON_OPTS[@]}" \
      BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
      BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
      >> "${logfile}" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "CHECKPOINT_POLICY: kept"
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
      echo "CHECKPOINT_POLICY: kept"
      echo "============================================================"
    } >> "${logfile}"
    echo "FAILED(${rc})" > "${statusfile}"
    return "${rc}"
  fi
}

# Build GPU slots, e.g. GPU_IDS="0 1", JOBS_PER_GPU=2 -> slots: 0 0 1 1
GPU_LIST=()
read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"

for gpu in "${PHYSICAL_GPUS[@]}"; do
  for ((i=0; i<JOBS_PER_GPU; i++)); do
    GPU_LIST+=("${gpu}")
  done
done

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPU slot resolved. Set GPU_IDS." >&2
  exit 1
fi

echo "[info] GPU_IDS=${GPU_IDS}"
echo "[info] JOBS_PER_GPU=${JOBS_PER_GPU}"
echo "[info] slots=${GPU_LIST[*]}"
echo "[info] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[info] SHOTS=${SHOTS}"
echo "[info] SEEDS=${SEEDS}"
echo "[info] SKIP_EXISTING=${SKIP_EXISTING}"
echo "[info] checkpoint deletion: disabled"

RUNNING_PIDS=()
SLOT_DESC=()
SLOT_LOG=()
FAILED=0

cleanup_children() {
  for pid in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

trap 'echo "[interrupt] stopping child jobs..."; cleanup_children; exit 130' INT TERM

READY_SLOT=""

wait_for_free_slot() {
  READY_SLOT=""

  while true; do
    for idx in "${!GPU_LIST[@]}"; do
      local pid="${RUNNING_PIDS[$idx]:-}"

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

        if [[ "${rc}" -eq 0 ]]; then
          echo "[done] ${SLOT_DESC[$idx]}"
        else
          echo "[fail] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
          FAILED=$((FAILED + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_DESC[$idx]=""
        SLOT_LOG[$idx]=""

        READY_SLOT="${idx}"
        return 0
      fi
    done

    sleep "${SLEEP_SEC}"
  done
}


for idx in "${!GPU_LIST[@]}"; do
  RUNNING_PIDS[$idx]=""
  SLOT_DESC[$idx]=""
  SLOT_LOG[$idx]=""
done

for dataset in ${DATASETS}; do
  for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
      wait_for_free_slot
      slot="${READY_SLOT}"
      gpu="${GPU_LIST[$slot]}"

      read -r tag prior_std kl <<< "$(get_diag_config "${dataset}")"
      outdir="$(build_outdir "${dataset}" "${shot}" "${tag}" "${seed}")"

      (
        launch_one "${gpu}" "${dataset}" "${shot}" "${seed}"
      ) &

      RUNNING_PIDS[$slot]=$!
      SLOT_DESC[$slot]="dataset=${dataset} shot=${shot} seed=${seed} tag=${tag} gpu=${gpu}"
      SLOT_LOG[$slot]="${outdir}/run.log"
    done
  done
done

for idx in "${!RUNNING_PIDS[@]}"; do
  pid="${RUNNING_PIDS[$idx]}"
  if [[ -n "${pid}" ]]; then
    rc=0
    if wait "${pid}"; then
      rc=0
    else
      rc=$?
    fi

    if [[ "${rc}" -eq 0 ]]; then
      echo "[done] ${SLOT_DESC[$idx]}"
    else
      echo "[fail] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
      FAILED=$((FAILED + 1))
    fi
  fi
done

echo
echo "[summary] parsing results if available..."
python evaluation/result_parser.py "${OUTPUT_ROOT}/${METHOD}/${PROTOCOL}" --split test || true

if [[ "${FAILED}" -gt 0 ]]; then
  echo "[DONE] finished with ${FAILED} failed job(s)."
  exit 1
fi

echo "[DONE] all Diag FS rerun jobs finished successfully."