#!/usr/bin/env bash
set -euo pipefail
# Run from personal1_pro/MMRL, same FS configs as run_plan.sh.
DATA_ROOT="${DATA_ROOT:-DATASETS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NGPU=1 GPU_IDS=0 JOBS_PER_GPU=1
OUTPUT_EFFICIENCY="${OUTPUT_EFFICIENCY:-output_efficiency}"

# bash run_efficiency.sh --profile-flops

SHOTS="${SHOTS:-16}"
# 1 2 3
SEEDS="${SEEDS:-1}"
#caltech101 eurosat fgvc_aircraft
DATASETS="${DATASETS:-caltech101 }"
# MMRL BayesRTMMRL DetBayesRTMMRL
METHODS="${METHODS:-  DetBayesRTMMRL}"
BACKBONE="${BACKBONE:-ViT-B/16}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MEASURE_STEPS="${MEASURE_STEPS:-20}"

for dataset in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      echo "[efficiency] dataset=$dataset method=$method shots=$SHOTS seed=$seed actual_pretrain_epochs=$PRETRAIN_EPOCHS"
      python benchmark_efficiency.py \
        --root "$DATA_ROOT" --dataset "$dataset" --method "$method" \
        --shots "$SHOTS" --seed "$seed" --backbone "$BACKBONE" \
        --pretrain-epochs "$PRETRAIN_EPOCHS" --warmup "$WARMUP_STEPS" --steps "$MEASURE_STEPS" \
        --output-dir "$OUTPUT_EFFICIENCY" "$@"
    done
  done
done
python summarize_efficiency.py --input "$OUTPUT_EFFICIENCY" --output "$OUTPUT_EFFICIENCY/efficiency_summary.csv"
