#!/bin/bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-DATASETS}
TASK=${1:-B2N}
TRAINER=${2:-MMRL}
DATASET=${3:-imagenet}
SHOTS=${4:-16}
SEEDS=${SEEDS:-"1 2 3"}

if [[ "${TRAINER}" != "MMRL" && "${TRAINER}" != "MMRLpp" ]]; then
  echo "Unsupported trainer: ${TRAINER}"
  exit 1
fi

case "${TASK}" in
  B2N)
    SPLIT_SUBSAMPLE=base
    if [[ "${DATASET}" == "imagenet" ]]; then CFG=vit_b16_imagenet; else CFG=vit_b16; fi
    OUT_PREFIX="output/base2new/train_base"
    ;;
  FS)
    SPLIT_SUBSAMPLE=all
    CFG=vit_b16_few_shot
    OUT_PREFIX="output/few_shot"
    ;;
  CD)
    SPLIT_SUBSAMPLE=all
    CFG=vit_b16_cross_datasets
    OUT_PREFIX="output/cross_datasets"
    ;;
  *)
    echo "Unsupported task: ${TASK}"
    exit 1
    ;;
esac

for SEED in ${SEEDS}; do
  DIR=${OUT_PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
  if [[ -d "${DIR}" ]]; then
    echo "Skip existing ${DIR}"
    continue
  fi

  python train.py \
    --root "${DATA_ROOT}" \
    --seed "${SEED}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
    --output-dir "${DIR}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES "${SPLIT_SUBSAMPLE}" \
    TASK "${TASK}"
done

python parse_test_res.py "${OUT_PREFIX}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/"
