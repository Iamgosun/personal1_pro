#!/usr/bin/env bash
set -euo pipefail
# Run this script from the MMRL directory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NGPU=1 GPU_IDS=0 JOBS_PER_GPU=1
DATA_ROOT="${DATA_ROOT:-DATASETS}"
OUTPUT_EFFICIENCY="${OUTPUT_EFFICIENCY:-output_efficiency_50ep}"
DATASETS="${DATASETS:-caltech101  }"
SEEDS="${SEEDS:-1 }"
METHODS="${METHODS:-BayesRTMMRL DetBayesRTMMRL_CLAMP}"
SHOTS="${SHOTS:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
MC_TRAIN="${MC_TRAIN:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
BACKBONE="${BACKBONE:-ViT-B/16}"

for dataset in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      echo "[10-epoch] dataset=$dataset method=$method seed=$seed MC=$MC_TRAIN epochs=$TRAIN_EPOCHS"
      python benchmark_10epochs.py \
        --root "$DATA_ROOT" --dataset "$dataset" --method "$method" \
        --shots "$SHOTS" --seed "$seed" --batch-size "$BATCH_SIZE" \
        --backbone "$BACKBONE" --train-epochs "$TRAIN_EPOCHS" \
        --mc-train "$MC_TRAIN" --warmup "$WARMUP_STEPS" \
        --output-dir "$OUTPUT_EFFICIENCY" "$@"
    done
  done
done

python summarize_10epochs.py --input "$OUTPUT_EFFICIENCY" \
  --output "$OUTPUT_EFFICIENCY/efficiency_summary.csv"
