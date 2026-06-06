#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# MMRL experiment runner
#
# FS/CD:
#   Keep the normal single-stage execution path.
#
# B2N:
#   Run train_base first.
#   Then run test_new with:
#     - best HPO opts from train_base/hpo_best_opts.json if present
#     - best model dir resolved from train_base/hpo_best_opts.json if present
#     - HPO forced off for eval-only
#
# Usage:
#   GPU_IDS="0 1" bash run_plan.sh B2N "DetBayesRTMMRL" online "caltech101 dtd" "1 2 4" "1 2 3"
#   GPU_IDS="0 1" bash run_plan.sh FS  "DetBayesRTMMRL" online "caltech101 dtd" "1 2 4" "1 2 3"
# ============================================================

PROTOCOL=${1:-B2N}
METHODS_ARG=${2:-"DetBayesRTMMRL MMRL BayesAdapter"}
EXEC_MODE=${3:-online}
DATASETS_ARG=${4:-"caltech101 oxford_pets dtd  food101 eurosat imagenet  oxford_flowers  sun397 fgvc_aircraft stanford_cars ucf101  "}
SHOTS_ARG=${5:-"16 "}
SEEDS_ARG=${6:-${SEEDS:-"1 2 3 "}}

DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
BACKBONE=${BACKBONE:-ViT-B/16}
TAG=${TAG:-}

GPU_IDS=${GPU_IDS:-"0 1 2"}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# Non-B2N eval-only mode is kept for compatibility.
# For B2N, eval-only is controlled internally only for test_new.
EVAL_ONLY=${EVAL_ONLY:-0}

# Summary is intentionally optional and conservative here.
# Set SUMMARY_SCOPE=none to skip.
SUMMARY_SCOPE=${SUMMARY_SCOPE:-current}

normalise_words() {
  tr '\n\t' '  ' <<< "$1" | xargs
}

METHODS_ARG="$(normalise_words "${METHODS_ARG}")"
DATASETS_ARG="$(normalise_words "${DATASETS_ARG}")"
SHOTS_ARG="$(normalise_words "${SHOTS_ARG}")"
SEEDS_ARG="$(normalise_words "${SEEDS_ARG}")"
GPU_IDS="$(normalise_words "${GPU_IDS}")"

read -r -a METHODS <<< "${METHODS_ARG}"
read -r -a DATASET_LIST <<< "${DATASETS_ARG}"
read -r -a SHOT_LIST <<< "${SHOTS_ARG}"
read -r -a SEED_LIST <<< "${SEEDS_ARG}"

method_key() {
  local method=$1
  echo "${method}" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g'
}

is_clip_adapter_cfg() {
  local cfg=$1
  [[ "${cfg}" == configs/methods/clip_adapters*.yaml ]]
}

resolve_method_cfg() {
  local method=$1
  local key
  key="$(method_key "${method}")"

  case "${method}" in
    MMRL)
      echo "configs/methods/mmrl.yaml"
      return 0
      ;;
    MMRLMix)
      echo "configs/methods/mmrl_mix.yaml"
      return 0
      ;;
    BayesMMRL)
      echo "configs/methods/bayesmmrl.yaml"
      return 0
      ;;
    BayesTextMMRL)
      echo "configs/methods/bayes_text_mmrl.yaml"
      return 0
      ;;
    BayesRTMMRL)
      echo "configs/methods/bayesrt_mmrl.yaml"
      return 0
      ;;
    DetBayesRTMMRL)
      echo "configs/methods/det_bayesrt_mmrl.yaml"
      return 0
      ;;
    VCRMMMRL)
      echo "configs/methods/vcrm_mmrl.yaml"
      return 0
      ;;
    MMRLpp|MMRLPP)
      echo "configs/methods/mmrlpp.yaml"
      return 0
      ;;
    ClipAdapters|ClipADAPTER)
      echo "configs/methods/clip_adapters.yaml"
      return 0
      ;;
  esac

  case "${key}" in
    zs)
      echo "configs/methods/clip_adapters_zs.yaml"
      return 0
      ;;
    random)
      echo "configs/methods/clip_adapters_random.yaml"
      return 0
      ;;
    bayesadapter|bayes_adapter)
      echo "configs/methods/clip_adapters_bayes.yaml"
      return 0
      ;;
    bayesadapter_l2|bayes_adapter_l2)
      echo "configs/methods/clip_adapters_bayes_clap.yaml"
      return 0
      ;;
    dream_bayes_adapter|dreambayes|dream_ba)
      echo "configs/methods/clip_adapters_dream_bayes.yaml"
      return 0
      ;;
    taskres|task_res|tr)
      echo "configs/methods/clip_adapters_tr.yaml"
      return 0
      ;;
    taskres_grid|task_res_grid|tr_grid)
      echo "configs/methods/clip_adapters_tr_grid.yaml"
      return 0
      ;;
    clipa|clip_adapter|clipadapter)
      echo "configs/methods/clip_adapters_clipa.yaml"
      return 0
      ;;
    tipa)
      echo "configs/methods/clip_adapters_tipa.yaml"
      return 0
      ;;
    tipa_f|tipa_f_)
      echo "configs/methods/clip_adapters_tipa_f.yaml"
      return 0
      ;;
    tipa_f_grid|tipa_f__grid)
      echo "configs/methods/clip_adapters_tipa_f_grid.yaml"
      return 0
      ;;
    crossmodal|cross_modal)
      echo "configs/methods/clip_adapters_crossmodal.yaml"
      return 0
      ;;
  esac

  local candidates=(
    "configs/methods/${key}.yaml"
    "configs/methods/clip_adapters_${key}.yaml"
  )

  local cfg
  for cfg in "${candidates[@]}"; do
    if [[ -f "${cfg}" ]]; then
      echo "${cfg}"
      return 0
    fi
  done

  echo "[ERROR] Unknown METHOD=${method}; no matching method config found." >&2
  echo "[ERROR] Tried:" >&2
  for cfg in "${candidates[@]}"; do
    echo "  - ${cfg}" >&2
  done
  return 1
}

resolve_protocol_cfg() {
  case "$1" in
    B2N)
      echo "configs/protocols/b2n.yaml"
      ;;
    B2N_TEST_NEW)
      echo "configs/protocols/b2n_test_new.yaml"
      ;;
    FS)
      echo "configs/protocols/fs.yaml"
      ;;
    CD)
      echo "configs/protocols/cd.yaml"
      ;;
    *)
      echo "[ERROR] Unknown PROTOCOL=$1" >&2
      return 1
      ;;
  esac
}

resolve_runtime_cfg() {
  local method=$1
  local method_cfg
  method_cfg="$(resolve_method_cfg "${method}")"

  case "${method}" in
    MMRL|MMRLMix|BayesMMRL|BayesRTMMRL|DetBayesRTMMRL|BayesTextMMRL|VCRMMMRL|MMRLpp|MMRLPP)
      echo "configs/runtime/mmrl_family.yaml"
      return 0
      ;;
  esac

  if is_clip_adapter_cfg "${method_cfg}"; then
    echo "configs/runtime/adapter_family.yaml"
    return 0
  fi

  echo "configs/runtime/default.yaml"
}

resolve_launch_method() {
  local method=$1
  local method_cfg
  method_cfg="$(resolve_method_cfg "${method}")"

  if is_clip_adapter_cfg "${method_cfg}"; then
    echo "ClipAdapters"
  else
    echo "${method}"
  fi
}

resolve_launch_exec_mode() {
  echo "${EXEC_MODE}"
}

resolve_phase_semantics() {
  case "$1" in
    B2N)
      echo "train_base base"
      ;;
    FS)
      echo "fewshot_train all"
      ;;
    CD)
      echo "cross_train all"
      ;;
    *)
      echo "[ERROR] Unknown PROTOCOL=$1" >&2
      return 1
      ;;
  esac
}

resolve_run_tag() {
  local method=$1
  local method_cfg=$2

  if [[ -n "${TAG}" ]]; then
    echo "${TAG}"
    return 0
  fi

  python - "${method_cfg}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    import yaml
except Exception:
    print("default")
    raise SystemExit(0)

try:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    print("default")
    raise SystemExit(0)

method_cfg = cfg.get("METHOD", {}) or {}
cad = cfg.get("CLIP_ADAPTERS", {}) or {}

print(method_cfg.get("TAG") or cad.get("INIT") or "default")
PY
}

init_gpu_list() {
  local -a base_gpus=()
  read -r -a base_gpus <<< "${GPU_IDS}"

  if [[ ${#base_gpus[@]} -eq 0 ]]; then
    echo "[ERROR] no GPU ids resolved. Set GPU_IDS, e.g. GPU_IDS=\"0 1\"." >&2
    exit 1
  fi

  GPU_LIST=()
  local gpu_id rep
  for gpu_id in "${base_gpus[@]}"; do
    for ((rep=0; rep<JOBS_PER_GPU; rep++)); do
      GPU_LIST+=("${gpu_id}")
    done
  done
}

build_outdir() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local run_tag=$5

  local launch_method phase subsample
  launch_method="$(resolve_launch_method "${method}")"
  read -r phase subsample <<< "$(resolve_phase_semantics "${PROTOCOL}")"

  if [[ "${launch_method}" == "ClipAdapters" || "${launch_method}" == "ClipADAPTER" ]]; then
    echo "${OUTPUT_ROOT}/${launch_method}/${run_tag}/${PROTOCOL}/${phase}/${dataset}/shots_${shot}/${BACKBONE//\//-}/seed${seed}"
  else
    echo "${OUTPUT_ROOT}/${launch_method}/${PROTOCOL}/${phase}/${dataset}/shots_${shot}/${BACKBONE//\//-}/${run_tag}/seed${seed}"
  fi
}

build_b2n_new_eval_outdir() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local run_tag=$5

  local launch_method
  launch_method="$(resolve_launch_method "${method}")"

  if [[ "${launch_method}" == "ClipAdapters" || "${launch_method}" == "ClipADAPTER" ]]; then
    echo "${OUTPUT_ROOT}/${launch_method}/${run_tag}/B2N/test_new/${dataset}/shots_${shot}/${BACKBONE//\//-}/seed${seed}"
  else
    echo "${OUTPUT_ROOT}/${launch_method}/B2N/test_new/${dataset}/shots_${shot}/${BACKBONE//\//-}/${run_tag}/seed${seed}"
  fi
}

build_logfile() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local run_tag=$5

  local outdir
  outdir="$(build_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"
  echo "${outdir}/run.log"
}

read_hpo_best_opts() {
  local train_dir=$1
  local best_json="${train_dir}/hpo_best_opts.json"

  if [[ ! -f "${best_json}" ]]; then
    return 0
  fi

  python - "${best_json}" <<'PY'
from __future__ import annotations

import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

opts = data.get("opts", [])

if not isinstance(opts, list) or len(opts) % 2 != 0:
    raise SystemExit(f"[ERROR] invalid opts in {path}: expected an even-length list")

for item in opts:
    print(str(item))
PY
}

resolve_b2n_model_dir() {
  local train_dir=$1
  local best_json="${train_dir}/hpo_best_opts.json"

  # Non-HPO result.
  if [[ ! -f "${best_json}" ]]; then
    echo "${train_dir}"
    return 0
  fi

  # HPO.COPY_BEST_MODEL=True copies best refactor_model to train_dir/refactor_model.
  if [[ -d "${train_dir}/refactor_model" ]]; then
    echo "${train_dir}"
    return 0
  fi

  # Fallback to selected best candidate dir.
  python - "${best_json}" <<'PY'
from __future__ import annotations

import json
import sys

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

best = data.get("best", {})
output_dir = best.get("output_dir")

if not isinstance(output_dir, str) or not output_dir:
    raise SystemExit(f"[ERROR] missing best.output_dir in {path}")

print(output_dir)
PY
}

train_case_is_complete() {
  local train_dir=$1

  [[ -f "${train_dir}/.run_plan_train.done" ]] && return 0
  [[ -f "${train_dir}/test_report.json" ]] && return 0
  [[ -d "${train_dir}/refactor_model" ]] && return 0

  if [[ -f "${train_dir}/hpo_best_opts.json" ]]; then
    local model_dir
    if model_dir="$(resolve_b2n_model_dir "${train_dir}")"; then
      [[ -d "${model_dir}" ]] && return 0
    fi
  fi

  return 1
}

b2n_new_eval_is_complete() {
  local eval_dir=$1
  [[ -f "${eval_dir}/.run_plan_eval.done" ]] && return 0
  [[ -f "${eval_dir}/test_report.json" ]] && return 0
  return 1
}

case_is_complete() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4

  local method_cfg run_tag train_dir eval_dir
  method_cfg="$(resolve_method_cfg "${method}")"
  run_tag="$(resolve_run_tag "${method}" "${method_cfg}")"
  train_dir="$(build_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"

  if ! train_case_is_complete "${train_dir}"; then
    return 1
  fi

  if [[ "${PROTOCOL}" != "B2N" ]]; then
    return 0
  fi

  eval_dir="$(build_b2n_new_eval_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"
  b2n_new_eval_is_complete "${eval_dir}"
}

write_train_log_header() {
  local logfile=$1
  local gpu_id=$2
  local method=$3
  local dataset=$4
  local shot=$5
  local seed=$6

  local method_cfg protocol_cfg runtime_cfg run_tag launch_method launch_exec_mode
  method_cfg="$(resolve_method_cfg "${method}")"
  protocol_cfg="$(resolve_protocol_cfg "${PROTOCOL}")"
  runtime_cfg="$(resolve_runtime_cfg "${method}")"
  run_tag="$(resolve_run_tag "${method}" "${method_cfg}")"
  launch_method="$(resolve_launch_method "${method}")"
  launch_exec_mode="$(resolve_launch_exec_mode)"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: ${PROTOCOL} train"
    echo "GPU: ${gpu_id}"
    echo "REQUESTED_METHOD: ${method}"
    echo "LAUNCH_METHOD: ${launch_method}"
    echo "RUN_TAG: ${run_tag}"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "EXEC_MODE: ${launch_exec_mode}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "METHOD_CONFIG: ${method_cfg}"
    echo "PROTOCOL_CONFIG: ${protocol_cfg}"
    echo "RUNTIME_CONFIG: ${runtime_cfg}"
    echo "============================================================"
  } >> "${logfile}"
}

launch_train_case() {
  local gpu_id=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5

  local method_cfg protocol_cfg runtime_cfg run_tag launch_method launch_exec_mode outdir logfile phase subsample
  method_cfg="$(resolve_method_cfg "${method}")"
  protocol_cfg="$(resolve_protocol_cfg "${PROTOCOL}")"
  runtime_cfg="$(resolve_runtime_cfg "${method}")"
  run_tag="$(resolve_run_tag "${method}" "${method_cfg}")"
  launch_method="$(resolve_launch_method "${method}")"
  launch_exec_mode="$(resolve_launch_exec_mode)"
  outdir="$(build_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"
  logfile="${outdir}/run.log"
  read -r phase subsample <<< "$(resolve_phase_semantics "${PROTOCOL}")"

  if [[ "${SKIP_EXISTING}" == "1" ]] && train_case_is_complete "${outdir}"; then
    echo "[SKIP][TRAIN] ${outdir}"
    return 0
  fi

  mkdir -p "${outdir}"
  write_train_log_header "${logfile}" "${gpu_id}" "${method}" "${dataset}" "${shot}" "${seed}"

  local -a cmd=(
    python run.py
    --root "${DATA_ROOT}"
    --seed "${seed}"
    --trainer RefactorRunner
    --method "${launch_method}"
    --protocol "${PROTOCOL}"
    --exec-mode "${launch_exec_mode}"
    --dataset-config-file "configs/datasets/${dataset}.yaml"
    --method-config-file "${method_cfg}"
    --protocol-config-file "${protocol_cfg}"
    --runtime-config-file "${runtime_cfg}"
    --output-dir "${outdir}"
  )

  if [[ "${EVAL_ONLY}" == "1" && "${PROTOCOL}" != "B2N" ]]; then
    cmd+=(
      --eval-only
      --model-dir "${outdir}"
    )
  fi

  cmd+=(
    DATASET.NUM_SHOTS "${shot}"
    MODEL.BACKBONE.NAME "${BACKBONE}"
    DATASET.SUBSAMPLE_CLASSES "${subsample}"
  )

  if [[ "${EVAL_ONLY}" == "1" && "${PROTOCOL}" != "B2N" ]]; then
    cmd+=(
      HPO.ENABLED False
      TEST.NO_TEST False
    )
  fi

  {
    echo "[CMD] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
  } >> "${logfile}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >> "${logfile}" 2>&1
  local rc=$?
  set -e

  echo "EXIT_CODE: ${rc}" >> "${logfile}"

  if [[ ${rc} -eq 0 ]]; then
    touch "${outdir}/.run_plan_train.done"
  fi

  return "${rc}"
}

launch_b2n_new_eval() {
  local gpu_id=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5

  local method_cfg runtime_cfg run_tag launch_method launch_exec_mode train_dir eval_dir model_dir logfile
  method_cfg="$(resolve_method_cfg "${method}")"
  runtime_cfg="$(resolve_runtime_cfg "${method}")"
  run_tag="$(resolve_run_tag "${method}" "${method_cfg}")"
  launch_method="$(resolve_launch_method "${method}")"
  launch_exec_mode="$(resolve_launch_exec_mode)"
  train_dir="$(build_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"
  eval_dir="$(build_b2n_new_eval_outdir "${method}" "${dataset}" "${shot}" "${seed}" "${run_tag}")"
  logfile="${eval_dir}/run.log"

  if [[ "${SKIP_EXISTING}" == "1" ]] && b2n_new_eval_is_complete "${eval_dir}"; then
    echo "[SKIP][B2N_TEST_NEW] ${eval_dir}"
    return 0
  fi

  if ! train_case_is_complete "${train_dir}"; then
    echo "[ERROR] B2N train_base result is incomplete: ${train_dir}" >&2
    return 1
  fi

  model_dir="$(resolve_b2n_model_dir "${train_dir}")"

  if [[ ! -d "${model_dir}" ]]; then
    echo "[ERROR] resolved model_dir does not exist: ${model_dir}" >&2
    echo "[ERROR] train_dir=${train_dir}" >&2
    return 1
  fi

  local -a best_opts=()
  if [[ -f "${train_dir}/hpo_best_opts.json" ]]; then
    mapfile -t best_opts < <(read_hpo_best_opts "${train_dir}")
  fi

  mkdir -p "${eval_dir}"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: B2N test_new"
    echo "GPU: ${gpu_id}"
    echo "REQUESTED_METHOD: ${method}"
    echo "LAUNCH_METHOD: ${launch_method}"
    echo "RUN_TAG: ${run_tag}"
    echo "PROTOCOL: B2N"
    echo "PHASE: test_new"
    echo "EXEC_MODE: ${launch_exec_mode}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "TRAIN_DIR: ${train_dir}"
    echo "MODEL_DIR: ${model_dir}"
    echo "METHOD_CONFIG: ${method_cfg}"
    echo "PROTOCOL_CONFIG: configs/protocols/b2n_test_new.yaml"
    echo "RUNTIME_CONFIG: ${runtime_cfg}"
    echo "HPO_BEST_JSON: ${train_dir}/hpo_best_opts.json"
    echo "BEST_OPTS: ${best_opts[*]:-<none>}"
    echo "============================================================"
  } >> "${logfile}"

  local -a cmd=(
    python run.py
    --root "${DATA_ROOT}"
    --seed "${seed}"
    --trainer RefactorRunner
    --method "${launch_method}"
    --protocol B2N
    --exec-mode "${launch_exec_mode}"
    --dataset-config-file "configs/datasets/${dataset}.yaml"
    --method-config-file "${method_cfg}"
    --protocol-config-file "configs/protocols/b2n_test_new.yaml"
    --runtime-config-file "${runtime_cfg}"
    --output-dir "${eval_dir}"
    --model-dir "${model_dir}"
    --eval-only
  )

  if [[ ${#best_opts[@]} -gt 0 ]]; then
    cmd+=("${best_opts[@]}")
  fi

  cmd+=(
    HPO.ENABLED False
    TEST.NO_TEST False
    DATASET.NUM_SHOTS "${shot}"
    MODEL.BACKBONE.NAME "${BACKBONE}"
    DATASET.SUBSAMPLE_CLASSES new
  )

  {
    echo "[CMD] CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
  } >> "${logfile}"

  set +e
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >> "${logfile}" 2>&1
  local rc=$?
  set -e

  echo "EXIT_CODE: ${rc}" >> "${logfile}"

  if [[ ${rc} -eq 0 ]]; then
    touch "${eval_dir}/.run_plan_eval.done"
  fi

  return "${rc}"
}

launch_b2n_case() {
  local gpu_id=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5

  launch_train_case "${gpu_id}" "${method}" "${dataset}" "${shot}" "${seed}"
  local train_rc=$?

  if [[ ${train_rc} -ne 0 ]]; then
    return "${train_rc}"
  fi

  launch_b2n_new_eval "${gpu_id}" "${method}" "${dataset}" "${shot}" "${seed}"
}

run_summary_if_requested() {
  if [[ "${SUMMARY_SCOPE}" == "none" ]]; then
    return 0
  fi

  # Keep summary optional. Different branches of this repo have used different
  # summary scripts, so do not fail the experiment if no summary script exists.
  if [[ -f "parse_test_res.py" && "${PROTOCOL}" != "B2N" ]]; then
    echo "[SUMMARY] parse_test_res.py is available, but automatic path inference is skipped."
  fi
}

print_plan_header() {
  echo "============================================================"
  echo "RUN PLAN"
  echo "PROTOCOL=${PROTOCOL}"
  echo "METHODS=${METHODS_ARG}"
  echo "EXEC_MODE=${EXEC_MODE}"
  echo "DATASETS=${DATASETS_ARG}"
  echo "SHOTS=${SHOTS_ARG}"
  echo "SEEDS=${SEEDS_ARG}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "BACKBONE=${BACKBONE}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "JOBS_PER_GPU=${JOBS_PER_GPU}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"
  echo "EVAL_ONLY=${EVAL_ONLY}"
  echo "SUMMARY_SCOPE=${SUMMARY_SCOPE}"
  echo "============================================================"
}

main() {
  case "${PROTOCOL}" in
    B2N|FS|CD)
      ;;
    *)
      echo "[ERROR] Unsupported PROTOCOL=${PROTOCOL}. Use B2N, FS, or CD." >&2
      exit 1
      ;;
  esac

  if [[ "${PROTOCOL}" == "B2N" && "${EVAL_ONLY}" == "1" ]]; then
    echo "[WARN] EVAL_ONLY=1 is ignored for PROTOCOL=B2N."
    echo "[WARN] B2N test_new eval-only is launched automatically after train_base."
  fi

  init_gpu_list
  print_plan_header

  local failures=0
  local slot_idx=0
  local method dataset shot seed gpu_id

  for method in "${METHODS[@]}"; do
    # Resolve once early to fail before launching jobs.
    resolve_method_cfg "${method}" >/dev/null

    for dataset in "${DATASET_LIST[@]}"; do
      for shot in "${SHOT_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
          gpu_id="${GPU_LIST[$((slot_idx % ${#GPU_LIST[@]}))]}"
          slot_idx=$((slot_idx + 1))

          echo "------------------------------------------------------------"
          echo "[CASE] protocol=${PROTOCOL} method=${method} dataset=${dataset} shot=${shot} seed=${seed} gpu=${gpu_id}"

          if [[ "${SKIP_EXISTING}" == "1" ]] && case_is_complete "${method}" "${dataset}" "${shot}" "${seed}"; then
            echo "[SKIP][CASE COMPLETE] method=${method} dataset=${dataset} shot=${shot} seed=${seed}"
            continue
          fi

          set +e
          if [[ "${PROTOCOL}" == "B2N" ]]; then
            launch_b2n_case "${gpu_id}" "${method}" "${dataset}" "${shot}" "${seed}"
          else
            launch_train_case "${gpu_id}" "${method}" "${dataset}" "${shot}" "${seed}"
          fi
          local rc=$?
          set -e

          if [[ ${rc} -ne 0 ]]; then
            failures=$((failures + 1))
            echo "[FAILED] protocol=${PROTOCOL} method=${method} dataset=${dataset} shot=${shot} seed=${seed} rc=${rc}"
          else
            echo "[OK] protocol=${PROTOCOL} method=${method} dataset=${dataset} shot=${shot} seed=${seed}"
          fi

          sleep "${SLEEP_SEC}"
        done
      done
    done
  done

  run_summary_if_requested

  echo "============================================================"
  if [[ ${failures} -ne 0 ]]; then
    echo "DONE WITH FAILURES: ${failures}"
    echo "============================================================"
    return 1
  fi

  echo "DONE"
  echo "============================================================"
  return 0
}

main "$@"
