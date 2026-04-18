#!/bin/bash
set -euo pipefail

PROTOCOL=${1:-FS}
METHODS_ARG=${2:-BayesMMRL }
EXEC_MODE=${3:-online}
DATASET=${4:-caltech101}
SHOTS_ARG=${5:-16}
SEEDS_ARG=${SEEDS:-1}
DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
BACKBONE=${BACKBONE:-ViT-B/16}
TAG=${TAG:-default}

read -r -a METHODS <<< "$METHODS_ARG"
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

resolve_configs() {
  local method=$1
  local method_cfg protocol_cfg runtime_cfg

  case "$method" in
    MMRL) method_cfg="configs/methods/mmrl.yaml" ;;
    BayesMMRL) method_cfg="configs/methods/bayesmmrl.yaml" ;;
    MMRLpp|MMRLPP) method_cfg="configs/methods/mmrlpp.yaml" ;;
    ClipAdapters|ClipADAPTER) method_cfg="configs/methods/clip_adapters.yaml" ;;
    *) echo "Unknown METHOD=$method" >&2; exit 1 ;;
  esac

  case "$PROTOCOL" in
    B2N) protocol_cfg="configs/protocols/b2n.yaml" ;;
    FS)  protocol_cfg="configs/protocols/fs.yaml" ;;
    CD)  protocol_cfg="configs/protocols/cd.yaml" ;;
    *) echo "Unknown PROTOCOL=$PROTOCOL" >&2; exit 1 ;;
  esac

  runtime_cfg="configs/runtime/default.yaml"
  echo "$method_cfg $protocol_cfg $runtime_cfg"
}

launch_one_case() {
  local method=$1
  local shot=$2
  local seed=$3

  read -r phase subsample <<< "$(resolve_phase_semantics "$PROTOCOL")"
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs "$method")"

  local outdir="${OUTPUT_ROOT}/${method}/${PROTOCOL}/${phase}/${DATASET}/shots_${shot}/${BACKBONE//\//-}/${TAG}/seed${seed}"

  echo "[RUN] method=${method} protocol=${PROTOCOL} exec=${EXEC_MODE} dataset=${DATASET} shots=${shot} seed=${seed}"

  python run.py \
    --root "${DATA_ROOT}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --method-config-file "${method_cfg}" \
    --protocol-config-file "${protocol_cfg}" \
    --runtime-config-file "${runtime_cfg}" \
    --output-dir "${outdir}" \
    --method "${method}" \
    --protocol "${PROTOCOL}" \
    --exec-mode "${EXEC_MODE}" \
    --seed "${seed}" \
    DATASET.NUM_SHOTS "${shot}" \
    DATASET.SUBSAMPLE_CLASSES "${subsample}" \
    MODEL.BACKBONE.NAME "${BACKBONE}"
}

summarize_case() {
  local method=$1
  python evaluation/result_parser.py "${OUTPUT_ROOT}/${method}/${PROTOCOL}" --split test || true
}
for method in "${METHODS[@]}"; do
  for shot in "${SHOT_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      launch_one_case "$method" "$shot" "$seed"
    done
  done
  summarize_case "$method"
done