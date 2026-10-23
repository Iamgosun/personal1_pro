#!/usr/bin/env bash
set -euo pipefail
# Run from MMRL root after extraction; uses existing native checkpoints.
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SEEDS="${SEEDS:-1 2 3}"
SHOTS="${SHOTS:-16}"
OUT="${OUT:-${OUTPUT:-output_supplementary_v7}}"
GPU="${GPU:-0}"
PPT_PAIR_IMAGES="${PPT_PAIR_IMAGES:-128}"
PPT_PAIR_BATCH="${PPT_PAIR_BATCH:-16}"
PPT_PAIR_INPUT_SEED="${PPT_PAIR_INPUT_SEED:-2027}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
extras=()
if [[ "${SAVE_MATRICES:-1}" == "1" ]]; then extras+=(--save-matrices); fi
read -r -a ds <<< "$DATASETS"
read -r -a ss <<< "$SEEDS"
python -m experiments.supplementary.paired_checkpoint_diagnostics \
  --datasets "${ds[@]}" --seeds "${ss[@]}" --shot "$SHOTS" \
  --images "$PPT_PAIR_IMAGES" --batch "$PPT_PAIR_BATCH" \
  --input-seed "$PPT_PAIR_INPUT_SEED" --output "$OUT" --gpu "$GPU" "${extras[@]}"
