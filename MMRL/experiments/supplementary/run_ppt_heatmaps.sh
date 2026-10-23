#!/usr/bin/env bash
# Additive visualization, no GPU training required (except optional separate run_ppt_pairs.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUTPUT="${OUTPUT:-output_supplementary_v7}"
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SHOTS="${SHOTS:-16}"
ONLY="${ONLY:-all}"
MATRIX_SIZE="${MATRIX_SIZE:-32}"
read -r -a ds <<< "$DATASETS"
python -m experiments.supplementary.ppt_heatmaps \
  --output "$OUTPUT" --datasets "${ds[@]}" --shot "$SHOTS" \
  --only "$ONLY" --matrix-size "$MATRIX_SIZE"
