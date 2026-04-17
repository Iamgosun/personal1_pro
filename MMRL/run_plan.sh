#!/bin/bash
set -euo pipefail

PROTOCOL=${1:-FS}
METHOD=${2:-MMRL}
EXEC_MODE=${3:-online}
DATASET=${4:-caltech101}
SHOTS=${5:-1}
SEEDS=${SEEDS:-1 2 3}
DATA_ROOT=${DATA_ROOT:-DATASETS}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
BACKBONE=${BACKBONE:-ViT-B/16}
TAG=${TAG:-default}

resolve_phase_semantics() {
  case "$PROTOCOL" in
    B2N) echo "train_base base" ;;
    FS) echo "fewshot_train all" ;;
    CD) echo "cross_train all" ;;
    *) echo "unknown all" ;;
  esac
}

resolve_configs() {
  local method_cfg protocol_cfg runtime_cfg
  case "$METHOD" in
    MMRL) method_cfg="configs/methods/mmrl.yaml" ;;
    MMRLpp|MMRLPP) method_cfg="configs/methods/mmrlpp.yaml" ;;
    ClipAdapters|ClipADAPTER) method_cfg="configs/methods/clip_adapters.yaml" ;;
    *) echo "Unknown METHOD=$METHOD" >&2; exit 1 ;;
  esac
  case "$PROTOCOL" in
    B2N) protocol_cfg="configs/protocols/b2n.yaml" ;;
    FS) protocol_cfg="configs/protocols/fs.yaml" ;;
    CD) protocol_cfg="configs/protocols/cd.yaml" ;;
    *) echo "Unknown PROTOCOL=$PROTOCOL" >&2; exit 1 ;;
  esac
  runtime_cfg="configs/runtime/default.yaml"
  echo "$method_cfg $protocol_cfg $runtime_cfg"
}

launch_one_case() {
  local seed=$1
  read -r phase subsample <<< "$(resolve_phase_semantics)"
  read -r method_cfg protocol_cfg runtime_cfg <<< "$(resolve_configs)"
  local outdir="${OUTPUT_ROOT}/${METHOD}/${PROTOCOL}/${phase}/${DATASET}/shots_${SHOTS}/${BACKBONE//\//-}/${TAG}/seed${seed}"
  echo "[RUN] method=${METHOD} protocol=${PROTOCOL} exec=${EXEC_MODE} dataset=${DATASET} shots=${SHOTS} seed=${seed}"
  python run.py     --root "${DATA_ROOT}"     --dataset-config-file "configs/datasets/${DATASET}.yaml"     --method-config-file "${method_cfg}"     --protocol-config-file "${protocol_cfg}"     --runtime-config-file "${runtime_cfg}"     --output-dir "${outdir}"     --method "${METHOD}"     --protocol "${PROTOCOL}"     --exec-mode "${EXEC_MODE}"     --seed "${seed}"     DATASET.NUM_SHOTS "${SHOTS}"     DATASET.SUBSAMPLE_CLASSES "${subsample}"     MODEL.BACKBONE.NAME "${BACKBONE}"
}

summarize_case() {
  python evaluation/result_parser.py "${OUTPUT_ROOT}/${METHOD}/${PROTOCOL}" || true
}

for seed in ${SEEDS}; do
  launch_one_case "$seed"
done
summarize_case
