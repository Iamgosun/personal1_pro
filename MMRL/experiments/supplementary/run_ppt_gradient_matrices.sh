#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUTPUT="${OUTPUT:-output_supplementary_v7}"
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SEEDS="${SEEDS:-1 2 3}"
SHOTS="${SHOTS:-16}"
GPU="${GPU:-0}"
DRAWS="${DRAWS:-10}"
GRAD_BATCH="${GRAD_BATCH:-32}"
read -r -a ds <<< "$DATASETS"
read -r -a ss <<< "$SEEDS"
python -m experiments.supplementary.gradient_draw_cosine \
 --output "$OUTPUT" --datasets "${ds[@]}" --seeds "${ss[@]}" \
 --shot "$SHOTS" --gpu "$GPU" --draws "$DRAWS" --batch-size "$GRAD_BATCH"
