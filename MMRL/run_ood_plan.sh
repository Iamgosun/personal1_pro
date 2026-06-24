#!/bin/bash
set -euo pipefail

# Usage:
#   GPU_IDS="0 1" bash run_ood_plan.sh FS "BayesAdapter MMRL BayesRTMMRL" auto cifar10 "dtd tinyimagenet lsun" "1 2 4 8 16 32" "1 2 3"
#
# Args:
#   1: PROTOCOL       FS
#   2: METHODS        "BayesAdapter MMRL DetBayesRTMMRL CrossModal"
#   3: EXEC_MODE      auto | online | cache
#   4: ID_DATASET     cifar10
#   5: OOD_DATASETS   "dtd tinyimagenet lsun"
#   6: SHOTS          "1 2 4 8 16 32"
#   7: SEEDS          "1 2 3"

# dtd tinyimagenet oxford_flowers sun397 inaturalist lsun
PROTOCOL=${1:-FS}
METHODS_ARG=${2:-"  DetBayesRTMMRL_CLAMP  "}
EXEC_MODE=${3:-online}
ID_DATASET=${4:-cifar_10}
OOD_DATASETS_ARG=${5:-"    dtd tinyimagenet oxford_flowers sun397 inaturalist lsun  "}
SHOTS_ARG=${6:-"16 "}
SEEDS_ARG=${7:-${SEEDS:-"1 2 3"}}

DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
BACKBONE=${BACKBONE:-ViT-B/16}
TAG=${TAG:-}

NGPU=${NGPU:-1}
GPU_IDS=${GPU_IDS:-0  }
JOBS_PER_GPU=${JOBS_PER_GPU:-6}

SKIP_EXISTING=${SKIP_EXISTING:-0}
TRAIN_IF_MISSING=${TRAIN_IF_MISSING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

OOD_BATCH_SIZE=${OOD_BATCH_SIZE:-250}
OOD_NUM_WORKERS=${OOD_NUM_WORKERS:-4}

read -r -a METHODS <<< "$METHODS_ARG"
read -r -a OOD_DATASET_LIST <<< "$OOD_DATASETS_ARG"
read -r -a SHOT_LIST <<< "$SHOTS_ARG"
read -r -a SEED_LIST <<< "$SEEDS_ARG"


resolve_phase_semantics() {
  case "$1" in
    B2N) echo "train_base base" ;;
    FS)  echo "fewshot_train all" ;;
    CD)  echo "cross_train all" ;;
    *) echo "Unknown PROTOCOL=$1" >&2; exit 1 ;;
  esac
}


resolve_protocol_cfg() {
  case "$1" in
    B2N) echo "configs/protocols/b2n.yaml" ;;
    FS)  echo "configs/protocols/fs.yaml" ;;
    CD)  echo "configs/protocols/cd.yaml" ;;
    *) echo "Unknown PROTOCOL=$1" >&2; exit 1 ;;
  esac
}


method_key() {
  local method=$1
  echo "$method" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g'
}


is_clip_adapter_cfg() {
  local cfg=$1
  [[ "$cfg" == configs/methods/clip_adapters*.yaml ]]
}


resolve_method_cfg() {
  local method=$1
  local key
  key="$(method_key "$method")"

  case "$method" in
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
    DetBayesRTMMRL)
      echo "configs/methods/det_bayesrt_mmrl.yaml"
      return 0
      ;;
      
    DetBayesRTMMRL_CLAMP)
      echo "configs/methods/det_bayesrt_mmrl_clamp.yaml"
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

  case "$key" in
    zs)
      echo "configs/methods/clip_adapters_zs.yaml"
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
    if [[ -f "$cfg" ]]; then
      echo "$cfg"
      return 0
    fi
  done

  echo "Unknown METHOD=$method; no matching method config found." >&2
  echo "Tried:" >&2
  for cfg in "${candidates[@]}"; do
    echo "  - $cfg" >&2
  done

  return 1
}


resolve_runtime_cfg() {
  local method=$1
  local method_cfg
  method_cfg="$(resolve_method_cfg "$method")"

  case "$method" in
    MMRL|MMRLMix|BayesMMRL|BayesRTMMRL|DetBayesRTMMRL|DetBayesRTMMRL_CLAMP|BayesTextMMRL|VCRMMMRL|MMRLpp|MMRLPP)
      echo "configs/runtime/mmrl_family.yaml"
      return 0
      ;;
  esac

  if is_clip_adapter_cfg "$method_cfg"; then
    echo "configs/runtime/adapter_family.yaml"
    return 0
  fi

  echo "configs/runtime/default.yaml"
}


resolve_launch_method() {
  local method=$1
  local method_cfg
  method_cfg="$(resolve_method_cfg "$method")"

  case "$method" in
    DetBayesRTMMRLtheory|DetBayesRTMMRL_CLAMP)
      echo "DetBayesRTMMRL"
      return 0
      ;;
  esac

  if is_clip_adapter_cfg "$method_cfg"; then
    echo "ClipAdapters"
  else
    echo "$method"
  fi
}

resolve_output_method() {
  local method=$1

  case "$method" in
    DetBayesRTMMRLtheory)
      echo "DetBayesRTMMRLtheory"
      ;;
    DetBayesRTMMRL_CLAMP)
      echo "DetBayesRTMMRL_CLAMP"
      ;;
    *)
      resolve_launch_method "$method"
      ;;
  esac
}

resolve_launch_exec_mode() {
  local method=$1
  local method_cfg
  method_cfg="$(resolve_method_cfg "$method")"

  if [[ "$EXEC_MODE" != "auto" ]]; then
    echo "$EXEC_MODE"
    return 0
  fi

  if is_clip_adapter_cfg "$method_cfg"; then
    echo "cache"
  else
    echo "online"
  fi
}


resolve_run_tag() {
  local method=$1
  local method_cfg=$2

  if [[ -n "${TAG}" ]]; then
    echo "${TAG}"
    return 0
  fi

  python - <<PY
import yaml
from pathlib import Path

path = Path("${method_cfg}")
with path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

method_cfg = cfg.get("METHOD", {}) or {}
cad = cfg.get("CLIP_ADAPTERS", {}) or {}

print(method_cfg.get("TAG") or cad.get("INIT") or "default")
PY
}


resolve_configs() {
  local method=$1
  local method_cfg protocol_cfg runtime_cfg

  method_cfg="$(resolve_method_cfg "$method")"
  protocol_cfg="$(resolve_protocol_cfg "$PROTOCOL")"
  runtime_cfg="$(resolve_runtime_cfg "$method")"

  echo "$method_cfg $protocol_cfg $runtime_cfg"
}


build_outdir() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local run_tag=$5

  local launch_method output_method
  launch_method="$(resolve_launch_method "$method")"
  output_method="$(resolve_output_method "$method")"

  read -r phase _subsample <<< "$(resolve_phase_semantics "$PROTOCOL")"

  case "$method" in
    DetBayesRTMMRLtheory|DetBayesRTMMRL_CLAMP)
      echo "${OUTPUT_ROOT}/${output_method}/${PROTOCOL}/${phase}/${dataset}/shots_${shot}/${BACKBONE//\//-}/seed${seed}"
      return 0
      ;;
  esac

  if [[ "$launch_method" == "ClipAdapters" || "$launch_method" == "ClipADAPTER" ]]; then
    echo "${OUTPUT_ROOT}/${launch_method}/${run_tag}/${PROTOCOL}/${phase}/${dataset}/shots_${shot}/${BACKBONE//\//-}/seed${seed}"
  else
    echo "${OUTPUT_ROOT}/${launch_method}/${PROTOCOL}/${phase}/${dataset}/shots_${shot}/${BACKBONE//\//-}/${run_tag}/seed${seed}"
  fi
}


case_is_trained() {
  local outdir=$1
  [[ -f "${outdir}/test_report.json" || -f "${outdir}/grid_search_summary.json" ]]
}


normalize_list() {
  echo "$1" | xargs
}


case_has_ood() {
  local ood_outdir=$1
  local manifest="${ood_outdir}/ood_datasets.txt"

  [[ -f "${ood_outdir}/ood_results.csv" ]] || return 1
  [[ -f "${manifest}" ]] || return 1
  [[ "$(cat "${manifest}")" == "$(normalize_list "${OOD_DATASETS_ARG}")" ]]
}


init_gpu_list() {
  local base=()

  if [[ -n "$GPU_IDS" ]]; then
    read -r -a base <<< "$GPU_IDS"
  else
    local i
    for ((i=0; i<NGPU; i++)); do
      base+=("$i")
    done
  fi

  if [[ ${#base[@]} -eq 0 ]]; then
    echo "No GPU ids resolved. Set NGPU or GPU_IDS." >&2
    exit 1
  fi

  GPU_LIST=()
  local gpu_id rep
  for gpu_id in "${base[@]}"; do
    for ((rep=0; rep<JOBS_PER_GPU; rep++)); do
      GPU_LIST+=("$gpu_id")
    done
  done
}


write_ood_log_header() {
  local logfile=$1
  local gpu_id=$2
  local method=$3
  local shot=$4
  local seed=$5
  local method_cfg=$6
  local protocol_cfg=$7
  local runtime_cfg=$8
  local launch_method=$9
  local launch_exec_mode=${10}
  local run_tag=${11}

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: OOD"
    echo "GPU: ${gpu_id}"
    echo "REQUESTED_METHOD: ${method}"
    echo "LAUNCH_METHOD: ${launch_method}"
    echo "RUN_TAG: ${run_tag}"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "EXEC_MODE: ${launch_exec_mode}"
    echo "ID_DATASET: ${ID_DATASET}"
    echo "OOD_DATASETS: ${OOD_DATASETS_ARG}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "METHOD_CONFIG: ${method_cfg}"
    echo "PROTOCOL_CONFIG: ${protocol_cfg}"
    echo "RUNTIME_CONFIG: ${runtime_cfg}"
    echo "============================================================"
  } >> "$logfile"
}


launch_one_case() {
  local gpu_id=$1
  local method=$2
  local shot=$3
  local seed=$4

  local phase subsample
  read -r phase subsample <<< "$(resolve_phase_semantics "$PROTOCOL")"

  local method_cfg protocol_cfg runtime_cfg
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local launch_method launch_exec_mode run_tag
  launch_method="$(resolve_launch_method "$method")"
  launch_exec_mode="$(resolve_launch_exec_mode "$method")"
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"

  local train_outdir ood_outdir logfile statusfile
  train_outdir="$(build_outdir "$method" "$ID_DATASET" "$shot" "$seed" "$run_tag")"
  ood_outdir="${train_outdir}/ood_eval"
  logfile="${ood_outdir}/run.log"
  statusfile="${ood_outdir}/job_status.txt"

  mkdir -p "$ood_outdir"
  : > "$logfile"

  write_ood_log_header \
    "$logfile" \
    "$gpu_id" \
    "$method" \
    "$shot" \
    "$seed" \
    "$method_cfg" \
    "$protocol_cfg" \
    "$runtime_cfg" \
    "$launch_method" \
    "$launch_exec_mode" \
    "$run_tag"

  if [[ "$SKIP_EXISTING" == "1" ]] && case_has_ood "$ood_outdir"; then
    echo "[SKIP] existing OOD results for OOD_DATASETS=${OOD_DATASETS_ARG}" >> "$logfile"
    echo "SKIP" > "$statusfile"
    return 0
  fi

  if ! case_is_trained "$train_outdir"; then
    if [[ "$TRAIN_IF_MISSING" != "1" ]]; then
      echo "[ERROR] missing trained ID model/report at ${train_outdir}" >> "$logfile"
      echo "FAILED_MISSING_TRAIN" > "$statusfile"
      return 1
    fi

    echo "[TRAIN] missing ID output, launching training" >> "$logfile"

    if ! CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${DATA_ROOT}" \
        --dataset-config-file "configs/datasets/${ID_DATASET}.yaml" \
        --method-config-file "${method_cfg}" \
        --protocol-config-file "${protocol_cfg}" \
        --runtime-config-file "${runtime_cfg}" \
        --output-dir "${train_outdir}" \
        --method "${launch_method}" \
        --protocol "${PROTOCOL}" \
        --exec-mode "${launch_exec_mode}" \
        --seed "${seed}" \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES "${subsample}" \
        MODEL.BACKBONE.NAME "${BACKBONE}" \
        >> "$logfile" 2>&1; then
      echo "[TRAIN] failed" >> "$logfile"
      echo "FAILED_TRAIN" > "$statusfile"
      return 1
    fi
  else
    echo "[TRAIN] found existing ID output: ${train_outdir}" >> "$logfile"
  fi

  local -a ood_args=()
  local ood_dataset
  for ood_dataset in "${OOD_DATASET_LIST[@]}"; do
    ood_args+=(--ood-dataset "$ood_dataset")
  done

  echo "[OOD] launching OOD evaluation" >> "$logfile"

  if CUDA_VISIBLE_DEVICES="${gpu_id}" python eval_ood.py \
      --root "${DATA_ROOT}" \
      --dataset-config-file "configs/datasets/${ID_DATASET}.yaml" \
      --method-config-file "${method_cfg}" \
      --protocol-config-file "${protocol_cfg}" \
      --runtime-config-file "${runtime_cfg}" \
      --output-dir "${ood_outdir}" \
      --model-dir "${train_outdir}" \
      --method "${launch_method}" \
      --protocol "${PROTOCOL}" \
      --exec-mode "${launch_exec_mode}" \
      --seed "${seed}" \
      --ood-output-dir "${ood_outdir}" \
      --ood-batch-size "${OOD_BATCH_SIZE}" \
      --ood-num-workers "${OOD_NUM_WORKERS}" \
      "${ood_args[@]}" \
      DATASET.NUM_SHOTS "${shot}" \
      DATASET.SUBSAMPLE_CLASSES "${subsample}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      >> "$logfile" 2>&1; then

    normalize_list "${OOD_DATASETS_ARG}" > "${ood_outdir}/ood_datasets.txt"

    {
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "============================================================"
    } >> "$logfile"

    echo "SUCCESS" > "$statusfile"
    return 0
  else
    local rc=$?
    {
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: FAILED"
      echo "EXIT_CODE: ${rc}"
      echo "============================================================"
    } >> "$logfile"

    echo "FAILED(${rc})" > "$statusfile"
    return "$rc"
  fi
}


summarize_case() {
  local method=$1

  local method_cfg protocol_cfg runtime_cfg
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local run_tag launch_method output_method
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"
  launch_method="$(resolve_launch_method "$method")"
  output_method="$(resolve_output_method "$method")"

  case "$method" in
    DetBayesRTMMRLtheory|DetBayesRTMMRL_CLAMP)
      python evaluation/result_parser.py \
        "${OUTPUT_ROOT}/${output_method}/${PROTOCOL}" \
        --split ood
      ;;

    *)
      if [[ "$launch_method" == "ClipAdapters" || "$launch_method" == "ClipADAPTER" ]]; then
        python evaluation/result_parser.py \
          "${OUTPUT_ROOT}/${launch_method}/${run_tag}/${PROTOCOL}" \
          --split ood
      else
        python evaluation/result_parser.py \
          "${OUTPUT_ROOT}/${launch_method}/${PROTOCOL}" \
          --split ood
      fi
      ;;
  esac
}


cleanup_children() {
  local p
  for p in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
}


print_finish_status() {
  local rc=$1
  local gpu_id=$2
  local method=$3
  local shot=$4
  local seed=$5
  local logfile=$6

  if [[ "$rc" -eq 0 ]]; then
    echo "[OK]   gpu=${gpu_id} method=${method} id=${ID_DATASET} ood=\"${OOD_DATASETS_ARG}\" shot=${shot} seed=${seed}" >&2
  else
    echo "[FAIL] gpu=${gpu_id} method=${method} id=${ID_DATASET} ood=\"${OOD_DATASETS_ARG}\" shot=${shot} seed=${seed} log=${logfile}" >&2
  fi
}


READY_SLOT=""

wait_for_any_slot() {
  READY_SLOT=""

  while true; do
    local idx
    for idx in "${!RUNNING_PIDS[@]}"; do
      local pid="${RUNNING_PIDS[$idx]}"

      if [[ -z "$pid" ]]; then
        READY_SLOT="$idx"
        return 0
      fi

      if ! kill -0 "$pid" 2>/dev/null; then
        local rc=0
        if wait "$pid"; then
          rc=0
        else
          rc=$?
        fi

        print_finish_status \
          "$rc" \
          "${SLOT_GPU[$idx]}" \
          "${SLOT_METHOD[$idx]}" \
          "${SLOT_SHOT[$idx]}" \
          "${SLOT_SEED[$idx]}" \
          "${SLOT_LOG[$idx]}"

        if [[ "$rc" -ne 0 ]]; then
          FAILED_JOBS=$((FAILED_JOBS + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_GPU[$idx]=""
        SLOT_METHOD[$idx]=""
        SLOT_SHOT[$idx]=""
        SLOT_SEED[$idx]=""
        SLOT_LOG[$idx]=""

        READY_SLOT="$idx"
        return 0
      fi
    done

    sleep "$SLEEP_SEC"
  done
}


wait_all_jobs() {
  local idx
  for idx in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$idx]}"

    if [[ -n "$pid" ]]; then
      local rc=0
      if wait "$pid"; then
        rc=0
      else
        rc=$?
      fi

      print_finish_status \
        "$rc" \
        "${SLOT_GPU[$idx]}" \
        "${SLOT_METHOD[$idx]}" \
        "${SLOT_SHOT[$idx]}" \
        "${SLOT_SEED[$idx]}" \
        "${SLOT_LOG[$idx]}"

      if [[ "$rc" -ne 0 ]]; then
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_METHOD[$idx]=""
      SLOT_SHOT[$idx]=""
      SLOT_SEED[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}


main() {
  init_gpu_list

  declare -ga RUNNING_PIDS
  declare -ga SLOT_GPU
  declare -ga SLOT_METHOD
  declare -ga SLOT_SHOT
  declare -ga SLOT_SEED
  declare -ga SLOT_LOG

  FAILED_JOBS=0

  local nslots=${#GPU_LIST[@]}
  local i
  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_METHOD[$i]=""
    SLOT_SHOT[$i]=""
    SLOT_SEED[$i]=""
    SLOT_LOG[$i]=""
  done

  trap 'echo "[INTERRUPT] stopping child jobs..."; cleanup_children; exit 130' INT TERM

  local method shot seed
  for method in "${METHODS[@]}"; do
    for shot in "${SHOT_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        local method_cfg run_tag outdir logfile
        method_cfg="$(resolve_method_cfg "$method")"
        run_tag="$(resolve_run_tag "$method" "$method_cfg")"
        outdir="$(build_outdir "$method" "$ID_DATASET" "$shot" "$seed" "$run_tag")"
        logfile="${outdir}/ood_eval/run.log"

        wait_for_any_slot
        local slot="$READY_SLOT"
        local gpu_id="${GPU_LIST[$slot]}"

        (
          launch_one_case "$gpu_id" "$method" "$shot" "$seed"
        ) &

        RUNNING_PIDS[$slot]=$!
        SLOT_GPU[$slot]="$gpu_id"
        SLOT_METHOD[$slot]="$method"
        SLOT_SHOT[$slot]="$shot"
        SLOT_SEED[$slot]="$seed"
        SLOT_LOG[$slot]="$logfile"

        echo "[LAUNCH] gpu=${gpu_id} method=${method} id=${ID_DATASET} ood=\"${OOD_DATASETS_ARG}\" shot=${shot} seed=${seed}"
      done
    done
  done

  wait_all_jobs

  for method in "${METHODS[@]}"; do
    summarize_case "$method"
  done

  if [[ "$FAILED_JOBS" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)."
    exit 1
  fi

  echo "[DONE] all OOD jobs finished successfully."
}


if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  for method in "${METHODS[@]}"; do
    summarize_case "$method"
  done
  exit 0
fi


main "$@"