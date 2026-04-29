#!/bin/bash
set -euo pipefail

# Usage:
#   GPU_IDS="0 1" bash run_plan.sh FS "MMRL BayesMMRL" online "caltech101 oxford_pets" "1 2 4" "1 2 3"
#   GPU_IDS="0 1" bash run_plan.sh FS "CLAP CAPEL VNC_CAPEL ZS RANDOM TR ClipA TipA TipA-f- CrossModal BayesAdapter" cache "caltech101" "1 2 4" "1 2 3"
#   online cache
# Notes:
#   - Normal methods use their normal method config.
#   - Adapter aliases map to specific configs/methods/clip_adapters_*.yaml.
#   - For adapter aliases, launch method is always ClipAdapters.
#   - B2N automatically runs test_new after train_base.
# caltech101 oxford_pets dtd
PROTOCOL=${1:-FS}
METHODS_ARG=${2:-  VNC_CAPEL }
EXEC_MODE=${3:-cache}
DATASETS_ARG=${4:-"  dtd  caltech101 "}
SHOTS_ARG=${5:-"1 4 8 "}
SEEDS_ARG=${6:-${SEEDS:-"1 2 3"}}

DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
BACKBONE=${BACKBONE:-ViT-B/16}
TAG=${TAG:-}

NGPU=${NGPU:-1}
GPU_IDS=${GPU_IDS:-0 }
JOBS_PER_GPU=${JOBS_PER_GPU:-2}

SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

read -r -a METHODS <<< "$METHODS_ARG"
read -r -a DATASET_LIST <<< "$DATASETS_ARG"
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


resolve_runtime_cfg() {
  local method=$1

  case "$method" in
    MMRL|MMRLMix|BayesMMRL|MMRLpp|MMRLPP)
      echo "configs/runtime/mmrl_family.yaml"
      ;;

    ZS|CLAP|ZS_CLAP|RANDOM|TR|TaskRes|TR_grid|TaskRes_grid|ClipA|CLIPA|TipA|TipA-f-|TipA-F|TIPA-F|TipA-f-_grid|TipA-F_grid|TIPA-F_grid|CrossModal|CROSSMODAL|BayesAdapter|BAYES_ADAPTER|BayesAdapter_l2|BAYES_ADAPTER_l2|CAPEL|VNC_CAPEL|ClipAdapters|ClipADAPTER)
      echo "configs/runtime/adapter_family.yaml"
      ;;

    *)
      echo "configs/runtime/default.yaml"
      ;;
  esac
}

resolve_configs() {
  local method=$1
  local method_cfg protocol_cfg runtime_cfg

  case "$method" in
    MMRL)
      method_cfg="configs/methods/mmrl.yaml"
      ;;

    MMRLMix)
      method_cfg="configs/methods/mmrl_mix.yaml"
      ;;

    BayesMMRL)
      method_cfg="configs/methods/bayesmmrl.yaml"
      ;;

    MMRLpp|MMRLPP)
      method_cfg="configs/methods/mmrlpp.yaml"
      ;;

    ClipAdapters|ClipADAPTER)
      method_cfg="configs/methods/clip_adapters.yaml"
      ;;

    ZS)
      method_cfg="configs/methods/clip_adapters_zs.yaml"
      ;;

    CLAP)
      method_cfg="configs/methods/clip_adapters_clap.yaml"
      ;;

    ZS_CLAP)
      method_cfg="configs/methods/clip_adapters_clap.yaml"
      ;;
    CAPEL)
      method_cfg="configs/methods/clip_adapters_capel.yaml"
      ;;
    VNC_CAPEL)
      method_cfg="configs/methods/clip_adapters_vnccapel.yaml"
      ;;
    RANDOM)
      method_cfg="configs/methods/clip_adapters_random.yaml"
      ;;

    TR|TaskRes)
      method_cfg="configs/methods/clip_adapters_tr.yaml"
      ;;

    TR_grid|TaskRes_grid)
      method_cfg="configs/methods/clip_adapters_tr_grid.yaml"
      ;;

    ClipA|CLIPA)
      method_cfg="configs/methods/clip_adapters_clipa.yaml"
      ;;

    TipA)
      method_cfg="configs/methods/clip_adapters_tipa.yaml"
      ;;

    TipA-f-|TipA-F|TIPA-F)
      method_cfg="configs/methods/clip_adapters_tipa_f.yaml"
      ;;

    TipA-f-_grid|TipA-F_grid|TIPA-F_grid)
      method_cfg="configs/methods/clip_adapters_tipa_f_grid.yaml"
      ;;

    CrossModal|CROSSMODAL)
      method_cfg="configs/methods/clip_adapters_crossmodal.yaml"
      ;;

    BayesAdapter|BAYES_ADAPTER)
      method_cfg="configs/methods/clip_adapters_bayes.yaml"
      ;;

    BayesAdapter_l2|BAYES_ADAPTER_l2)
      method_cfg="configs/methods/clip_adapters_bayes_clap.yaml"
      ;;

    *)
      echo "Unknown METHOD=$method" >&2
      exit 1
      ;;
  esac


  protocol_cfg="$(resolve_protocol_cfg "$PROTOCOL")"
  runtime_cfg="$(resolve_runtime_cfg "$method")"

  echo "$method_cfg $protocol_cfg $runtime_cfg"
  
}

resolve_launch_method() {
  local method=$1

  case "$method" in
    ZS|CLAP|ZS_CLAP|CAPEL|VNC_CAPEL|RANDOM|TR|TaskRes|TR_grid|TaskRes_grid|ClipA|CLIPA|TipA|TipA-f-|TipA-F|TIPA-F|TipA-f-_grid|TipA-F_grid|TIPA-F_grid|CrossModal|CROSSMODAL|BayesAdapter|BAYES_ADAPTER|BayesAdapter_l2|BAYES_ADAPTER_l2)
      echo "ClipAdapters"
      ;;
    *)
      echo "$method"
      ;;
  esac
}

resolve_launch_exec_mode() {
  local method=$1

  case "$method" in
    ZS|CLAP|ZS_CLAP|CAPEL|VNC_CAPEL|RANDOM|TR|TaskRes|TR_grid|TaskRes_grid|ClipA|CLIPA|TipA|TipA-f-|TipA-F|TIPA-F|TipA-f-_grid|TipA-F_grid|TIPA-F_grid|CrossModal|CROSSMODAL|BayesAdapter|BAYES_ADAPTER|BayesAdapter_l2|BAYES_ADAPTER_l2|ClipAdapters|ClipADAPTER)
      # Respect the third CLI argument.
      #
      # cache:
      #   cold feature cache; feature extraction is done before training
      #
      # online:
      #   realtime image augmentation + realtime CLIP image encoding;
      #   only transient support features are built for CLAP/TipA/CrossModal
      echo "$EXEC_MODE"
      ;;
    *)
      echo "$EXEC_MODE"
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

init_gpu_list() {
  local BASE_GPU_LIST=()

  if [[ -n "$GPU_IDS" ]]; then
    read -r -a BASE_GPU_LIST <<< "$GPU_IDS"
  else
    local i
    for ((i=0; i<NGPU; i++)); do
      BASE_GPU_LIST+=("$i")
    done
  fi

  if [[ ${#BASE_GPU_LIST[@]} -eq 0 ]]; then
    echo "No GPU ids resolved. Set NGPU or GPU_IDS." >&2
    exit 1
  fi

  GPU_LIST=()
  local gpu_id rep
  for gpu_id in "${BASE_GPU_LIST[@]}"; do
    for ((rep=0; rep<JOBS_PER_GPU; rep++)); do
      GPU_LIST+=("$gpu_id")
    done
  done
}

build_outdir() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local run_tag=$5

  local launch_method
  launch_method="$(resolve_launch_method "$method")"

  read -r phase _subsample <<< "$(resolve_phase_semantics "$PROTOCOL")"

  if [[ "$launch_method" == "ClipAdapters" || "$launch_method" == "ClipADAPTER" ]]; then
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
  launch_method="$(resolve_launch_method "$method")"

  if [[ "$launch_method" == "ClipAdapters" || "$launch_method" == "ClipADAPTER" ]]; then
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

  outdir="$(build_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"
  echo "${outdir}/run.log"
}

case_is_complete() {
  local method=$1
  local dataset=$2
  local shot=$3
  local seed=$4

  local method_cfg protocol_cfg runtime_cfg run_tag
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"

  local train_outdir
  train_outdir="$(build_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"

  if [[ ! -f "${train_outdir}/test_metrics.json" && ! -f "${train_outdir}/grid_search_summary.json" ]]; then
    return 1
  fi

  if [[ "$PROTOCOL" != "B2N" ]]; then
    return 0
  fi

  local eval_outdir
  eval_outdir="$(build_b2n_new_eval_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"

  [[ -f "${eval_outdir}/test_metrics.json" ]]
}

write_log_header() {
  local logfile=$1
  local gpu_id=$2
  local method=$3
  local dataset=$4
  local shot=$5
  local seed=$6

  local method_cfg protocol_cfg runtime_cfg run_tag launch_method launch_exec_mode
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"
  launch_method="$(resolve_launch_method "$method")"
  launch_exec_mode="$(resolve_launch_exec_mode "$method")"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
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
  } >> "$logfile"
}

write_b2n_new_eval_log_header() {
  local logfile=$1
  local gpu_id=$2
  local method=$3
  local dataset=$4
  local shot=$5
  local seed=$6
  local model_dir=$7

  local method_cfg protocol_cfg runtime_cfg run_tag launch_method launch_exec_mode
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"
  launch_method="$(resolve_launch_method "$method")"
  launch_exec_mode="$(resolve_launch_exec_mode "$method")"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: B2N test_new"
    echo "GPU: ${gpu_id}"
    echo "REQUESTED_METHOD: ${method}"
    echo "LAUNCH_METHOD: ${launch_method}"
    echo "RUN_TAG: ${run_tag}"
    echo "PROTOCOL: B2N"
    echo "EXEC_MODE: ${launch_exec_mode}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "MODEL_DIR: ${model_dir}"
    echo "METHOD_CONFIG: ${method_cfg}"
    echo "============================================================"
  } >> "$logfile"
}

launch_b2n_new_eval() {
  local gpu_id=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5

  local method_cfg protocol_cfg runtime_cfg
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local launch_method launch_exec_mode
  launch_method="$(resolve_launch_method "$method")"
  launch_exec_mode="$(resolve_launch_exec_mode "$method")"

  local run_tag
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"

  local train_outdir eval_outdir eval_log statusfile
  train_outdir="$(build_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"
  eval_outdir="$(build_b2n_new_eval_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"
  eval_log="${eval_outdir}/run.log"
  statusfile="${eval_outdir}/job_status.txt"

  mkdir -p "$eval_outdir"

  if [[ "$SKIP_EXISTING" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    echo "SKIP" > "$statusfile"
    return 0
  fi

  : > "$eval_log"
  write_b2n_new_eval_log_header "$eval_log" "$gpu_id" "$method" "$dataset" "$shot" "$seed" "$train_outdir"

  if CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
      --root "${DATA_ROOT}" \
      --dataset-config-file "configs/datasets/${dataset}.yaml" \
      --method-config-file "${method_cfg}" \
      --protocol-config-file "configs/protocols/b2n_test_new.yaml" \
      --runtime-config-file "${runtime_cfg}" \
      --output-dir "${eval_outdir}" \
      --model-dir "${train_outdir}" \
      --method "${launch_method}" \
      --protocol "B2N" \
      --exec-mode "${launch_exec_mode}" \
      --seed "${seed}" \
      --eval-only \
      DATASET.NUM_SHOTS "${shot}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      >> "$eval_log" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "============================================================"
    } >> "$eval_log"
    echo "SUCCESS" > "$statusfile"
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
    } >> "$eval_log"
    echo "FAILED(${rc})" > "$statusfile"
    return "$rc"
  fi
}

launch_one_case() {
  local gpu_id=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5

  local phase subsample
  read -r phase subsample <<< "$(resolve_phase_semantics "$PROTOCOL")"

  local method_cfg protocol_cfg runtime_cfg
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local launch_method launch_exec_mode
  launch_method="$(resolve_launch_method "$method")"
  launch_exec_mode="$(resolve_launch_exec_mode "$method")"

  local run_tag
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"

  local outdir logfile statusfile
  outdir="$(build_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"
  logfile="${outdir}/run.log"
  statusfile="${outdir}/job_status.txt"

  mkdir -p "$outdir"

  if [[ "$SKIP_EXISTING" == "1" ]] && case_is_complete "$method" "$dataset" "$shot" "$seed"; then
    echo "SKIP" > "$statusfile"
    return 0
  fi

  local train_metrics="${outdir}/test_metrics.json"
  local grid_metrics="${outdir}/grid_search_summary.json"
  local train_already_done=0

  if [[ -f "$train_metrics" || -f "$grid_metrics" ]]; then
    train_already_done=1
  fi

  if [[ "$train_already_done" -eq 0 ]]; then
    : > "$logfile"
    write_log_header "$logfile" "$gpu_id" "$method" "$dataset" "$shot" "$seed"

    if CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${DATA_ROOT}" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --method-config-file "${method_cfg}" \
        --protocol-config-file "${protocol_cfg}" \
        --runtime-config-file "${runtime_cfg}" \
        --output-dir "${outdir}" \
        --method "${launch_method}" \
        --protocol "${PROTOCOL}" \
        --exec-mode "${launch_exec_mode}" \
        --seed "${seed}" \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES "${subsample}" \
        MODEL.BACKBONE.NAME "${BACKBONE}" \
        >> "$logfile" 2>&1; then
      {
        echo
        echo "============================================================"
        echo "END: $(date '+%F %T')"
        echo "STATUS: SUCCESS"
        echo "============================================================"
      } >> "$logfile"
    else
      local rc=$?
      {
        echo
        echo "============================================================"
        echo "END: $(date '+%F %T')"
        echo "STATUS: FAILED"
        echo "EXIT_CODE: ${rc}"
        echo "============================================================"
      } >> "$logfile"
      echo "FAILED(${rc})" > "$statusfile"
      return "$rc"
    fi
  else
    touch "$logfile"
    {
      echo "============================================================"
      echo "SKIP_TRAIN: existing metrics found at ${outdir}"
      echo "TIME: $(date '+%F %T')"
      echo "============================================================"
    } >> "$logfile"
  fi

  if [[ "$PROTOCOL" == "B2N" ]]; then
    if launch_b2n_new_eval "$gpu_id" "$method" "$dataset" "$shot" "$seed"; then
      {
        echo "============================================================"
        echo "B2N_NEW_EVAL: SUCCESS"
        echo "TIME: $(date '+%F %T')"
        echo "============================================================"
      } >> "$logfile"
    else
      local rc=$?
      {
        echo "============================================================"
        echo "B2N_NEW_EVAL: FAILED"
        echo "EXIT_CODE: ${rc}"
        echo "TIME: $(date '+%F %T')"
        echo "============================================================"
      } >> "$logfile"
      echo "FAILED(${rc})" > "$statusfile"
      return "$rc"
    fi
  fi

  echo "SUCCESS" > "$statusfile"
  return 0
}

summarize_case() {
  local method=$1
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local run_tag launch_method
  run_tag="$(resolve_run_tag "$method" "$method_cfg")"
  launch_method="$(resolve_launch_method "$method")"

  if [[ "$launch_method" == "ClipAdapters" || "$launch_method" == "ClipADAPTER" ]]; then
    python evaluation/result_parser.py "${OUTPUT_ROOT}/${launch_method}/${run_tag}/${PROTOCOL}" --split test >/dev/null 2>&1 || true
  else
    python evaluation/result_parser.py "${OUTPUT_ROOT}/${launch_method}/${PROTOCOL}" --split test >/dev/null 2>&1 || true
  fi
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
  local dataset=$4
  local shot=$5
  local seed=$6
  local logfile=$7

  if [[ "$rc" -eq 0 ]]; then
    echo "[OK]   gpu=${gpu_id} method=${method} dataset=${dataset} shot=${shot} seed=${seed}" >&2
  else
    echo "[FAIL] gpu=${gpu_id} method=${method} dataset=${dataset} shot=${shot} seed=${seed} log=${logfile}" >&2
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

        local gpu_id="${SLOT_GPU[$idx]}"
        local method="${SLOT_METHOD[$idx]}"
        local dataset="${SLOT_DATASET[$idx]}"
        local shot="${SLOT_SHOT[$idx]}"
        local seed="${SLOT_SEED[$idx]}"
        local logfile="${SLOT_LOG[$idx]}"

        print_finish_status "$rc" "$gpu_id" "$method" "$dataset" "$shot" "$seed" "$logfile"

        if [[ "$rc" -ne 0 ]]; then
          FAILED_JOBS=$((FAILED_JOBS + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_GPU[$idx]=""
        SLOT_METHOD[$idx]=""
        SLOT_DATASET[$idx]=""
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

      local gpu_id="${SLOT_GPU[$idx]}"
      local method="${SLOT_METHOD[$idx]}"
      local dataset="${SLOT_DATASET[$idx]}"
      local shot="${SLOT_SHOT[$idx]}"
      local seed="${SLOT_SEED[$idx]}"
      local logfile="${SLOT_LOG[$idx]}"

      print_finish_status "$rc" "$gpu_id" "$method" "$dataset" "$shot" "$seed" "$logfile"

      if [[ "$rc" -ne 0 ]]; then
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_METHOD[$idx]=""
      SLOT_DATASET[$idx]=""
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
  declare -ga SLOT_DATASET
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
    SLOT_DATASET[$i]=""
    SLOT_SHOT[$i]=""
    SLOT_SEED[$i]=""
    SLOT_LOG[$i]=""
  done

  trap 'echo "[INTERRUPT] stopping child jobs..."; cleanup_children; exit 130' INT TERM

  local method dataset shot seed
  for method in "${METHODS[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      for shot in "${SHOT_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
          local method_cfg protocol_cfg runtime_cfg run_tag
          read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"
          run_tag="$(resolve_run_tag "$method" "$method_cfg")"

          local outdir logfile statusfile
          outdir="$(build_outdir "$method" "$dataset" "$shot" "$seed" "$run_tag")"
          logfile="${outdir}/run.log"
          statusfile="${outdir}/job_status.txt"

          if [[ "$SKIP_EXISTING" == "1" ]] && case_is_complete "$method" "$dataset" "$shot" "$seed"; then
            mkdir -p "$outdir"
            echo "SKIP" > "$statusfile"
            echo "[SKIP] method=${method} dataset=${dataset} shot=${shot} seed=${seed}"
            continue
          fi

          wait_for_any_slot
          local slot="$READY_SLOT"
          local gpu_id="${GPU_LIST[$slot]}"

          (
            launch_one_case "$gpu_id" "$method" "$dataset" "$shot" "$seed"
          ) &

          RUNNING_PIDS[$slot]=$!
          SLOT_GPU[$slot]="$gpu_id"
          SLOT_METHOD[$slot]="$method"
          SLOT_DATASET[$slot]="$dataset"
          SLOT_SHOT[$slot]="$shot"
          SLOT_SEED[$slot]="$seed"
          SLOT_LOG[$slot]="$logfile"

          echo "[LAUNCH] gpu=${gpu_id} method=${method} dataset=${dataset} shot=${shot} seed=${seed}"
        done
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

  echo "[DONE] all jobs finished successfully."
}

if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  for method in "${METHODS[@]}"; do
    summarize_case "$method"
  done
  exit 0
fi

main "$@"