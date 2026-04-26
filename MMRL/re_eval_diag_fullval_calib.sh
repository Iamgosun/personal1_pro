#!/bin/bash
set -euo pipefail

# ============================================================
# Eval-only rerun for BayesMMRL Diag FS runs.
#
# Purpose:
#   1) Do NOT retrain.
#   2) Load existing checkpoints from SRC_ROOT.
#   3) Use full validation set to fit temperature.
#   4) Re-predict test set and save raw + calibrated metrics.
#   5) Keep all checkpoints untouched.
#
# Usage:
#   cd MMRL
#   bash re_eval_diag_fullval_calib.sh GPU_IDS="0 1 2" JOBS_PER_GPU=1
#
# Optional env:
#   DATA_ROOT=DATASETS
#   SRC_ROOT=output_refactor/diag_rerun_stage
#   DST_ROOT=output_refactor/diag_rerun_stage_fullval_calib
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
        DATA_ROOT|SRC_ROOT|DST_ROOT|GPU_IDS|JOBS_PER_GPU|SHOTS|SEEDS|SKIP_EXISTING|SLEEP_SEC|BACKBONE|EXEC_MODE)
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
SRC_ROOT=${SRC_ROOT:-output_refactor/diag_rerun_stage}
DST_ROOT=${DST_ROOT:-output_refactor/diag_rerun_stage_fullval_calib}

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

bb_tag() {
  local s="$1"
  echo "${s//\//-}"
}

build_dir() {
  local root="$1"
  local dataset="$2"
  local shot="$3"
  local tag="$4"
  local seed="$5"

  echo "${root}/${METHOD}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/$(bb_tag "${BACKBONE}")/${tag}/seed${seed}"
}

run_one() {
  local gpu="$1"
  local dataset="$2"
  local shot="$3"
  local seed="$4"

  local tag prior_std kl
  read -r tag prior_std kl <<< "$(get_diag_config "${dataset}")"

  local model_dir out_dir log_file status_file
  model_dir="$(build_dir "${SRC_ROOT}" "${dataset}" "${shot}" "${tag}" "${seed}")"
  out_dir="$(build_dir "${DST_ROOT}" "${dataset}" "${shot}" "${tag}" "${seed}")"
  log_file="${out_dir}/eval_only.log"
  status_file="${out_dir}/job_status.txt"

  mkdir -p "${out_dir}"

  if [[ ! -d "${model_dir}/refactor_model" ]]; then
    {
      echo "============================================================"
      echo "STATUS: MISSING_CKPT"
      echo "MODEL_DIR: ${model_dir}"
      echo "Expected checkpoint dir: ${model_dir}/refactor_model"
      echo "============================================================"
    } | tee -a "${log_file}"

    echo "MISSING_CKPT" > "${status_file}"
    return 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/test_metrics.json" ]]; then
    echo "[skip] ${dataset} shot=${shot} seed=${seed}"
    echo "SKIP" > "${status_file}"
    return 0
  fi

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "MODE: eval-only full-val temperature calibration"
    echo "GPU: ${gpu}"
    echo "DATASET: ${dataset}"
    echo "SHOT: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "REP_SIGMA_MODE: diagonal"
    echo "REP_PRIOR_STD: ${prior_std}"
    echo "REP_KL_WEIGHT: ${kl}"
    echo "MODEL_DIR: ${model_dir}"
    echo "OUT_DIR: ${out_dir}"
    echo "IMPORTANT: no training, checkpoint kept"
    echo "============================================================"
  } > "${log_file}"

  echo "[eval-only] gpu=${gpu} dataset=${dataset} shot=${shot} seed=${seed} tag=${tag}"
  echo "            model_dir=${model_dir}"
  echo "            out_dir=${out_dir}"

  if CUDA_VISIBLE_DEVICES="${gpu}" python run.py \
      --root "${DATA_ROOT}" \
      --dataset-config-file "configs/datasets/${dataset}.yaml" \
      --method-config-file "${METHOD_CFG}" \
      --protocol-config-file "${PROTOCOL_CFG}" \
      --runtime-config-file "${RUNTIME_CFG}" \
      --output-dir "${out_dir}" \
      --model-dir "${model_dir}" \
      --method "${METHOD}" \
      --protocol "${PROTOCOL}" \
      --exec-mode "${EXEC_MODE}" \
      --seed "${seed}" \
      --eval-only \
      DATASET.NUM_SHOTS "${shot}" \
      DATASET.SUBSAMPLE_CLASSES "${SUBSAMPLE}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      CALIBRATION.USE_FULL_VAL True \
      "${COMMON_OPTS[@]}" \
      BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
      BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
      >> "${log_file}" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "CHECKPOINT_POLICY: kept"
      echo "============================================================"
    } >> "${log_file}"

    echo "SUCCESS" > "${status_file}"
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
    } >> "${log_file}"

    echo "FAILED(${rc})" > "${status_file}"
    return "${rc}"
  fi
}

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

RUNNING_PIDS=()
SLOT_DESC=()
SLOT_LOG=()
FAILED=0
READY_SLOT=""

for idx in "${!GPU_LIST[@]}"; do
  RUNNING_PIDS[$idx]=""
  SLOT_DESC[$idx]=""
  SLOT_LOG[$idx]=""
done

cleanup_children() {
  local pid

  for pid in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

trap 'echo "[interrupt] stopping child jobs..."; cleanup_children; exit 130' INT TERM

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

echo "[info] SRC_ROOT=${SRC_ROOT}"
echo "[info] DST_ROOT=${DST_ROOT}"
echo "[info] DATA_ROOT=${DATA_ROOT}"
echo "[info] GPU_IDS=${GPU_IDS}"
echo "[info] JOBS_PER_GPU=${JOBS_PER_GPU}"
echo "[info] GPU slots=${GPU_LIST[*]}"
echo "[info] SHOTS=${SHOTS}"
echo "[info] SEEDS=${SEEDS}"
echo "[info] SKIP_EXISTING=${SKIP_EXISTING}"
echo "[info] eval-only=true"
echo "[info] full-val calibration=true"
echo "[info] checkpoint deletion=disabled"

for dataset in ${DATASETS}; do
  for shot in ${SHOTS}; do
    for seed in ${SEEDS}; do
      wait_for_free_slot
      slot="${READY_SLOT}"
      gpu="${GPU_LIST[$slot]}"

      read -r tag prior_std kl <<< "$(get_diag_config "${dataset}")"
      out_dir="$(build_dir "${DST_ROOT}" "${dataset}" "${shot}" "${tag}" "${seed}")"

      (
        run_one "${gpu}" "${dataset}" "${shot}" "${seed}"
      ) &

      RUNNING_PIDS[$slot]=$!
      SLOT_DESC[$slot]="dataset=${dataset} shot=${shot} seed=${seed} tag=${tag} gpu=${gpu}"
      SLOT_LOG[$slot]="${out_dir}/eval_only.log"
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
echo "[summary] parsing results..."
python evaluation/result_parser.py "${DST_ROOT}/${METHOD}/${PROTOCOL}" --split test || true

if [[ "${FAILED}" -gt 0 ]]; then
  echo "[DONE] finished with ${FAILED} failed job(s)."
  exit 1
fi

echo "[DONE] all eval-only full-val calibration jobs finished."