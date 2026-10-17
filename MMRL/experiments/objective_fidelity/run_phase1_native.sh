#!/usr/bin/env bash
# READ-ONLY evaluation of existing V5 native checkpoints; no training.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMRL="$(cd "$HERE/../.." && pwd)"
cd "$MMRL"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-caltech101}"
SEEDS="${SEEDS:-1}"
MODES="${MODES:-branch_mc davi}"
SHOT="${SHOT:-16}"
SAMPLES="${SAMPLES:-4096}"
REPEATS="${REPEATS:-1}"
GRID="${GRID:-auto}"
CHUNK="${CHUNK:-32}"
REF_BATCH="${REF_BATCH:-32}"
OUT="${OUT:-output_objective_fidelity_v3}"
SPLIT="${SPLIT:-train_fixed}"
# All summary artifacts live in the same output root as the checkpoints.
# Existing per-checkpoint phase1/*.json remains next to each native checkpoint.
if [[ "$OUT" = /* ]]; then
  OUTPUT_ROOT="$OUT"
else
  OUTPUT_ROOT="$MMRL/$OUT"
fi
SUMMARY_DIR="$OUTPUT_ROOT/phase1_summary/$SPLIT"

# RENDER_ONLY=1 reuses already computed JSON without reloading CLIP/checkpoints.
if [[ "${RENDER_ONLY:-0}" != "1" ]]; then
  for ds in $DATASETS; do
    for seed in $SEEDS; do
      for mode in $MODES; do
        echo "[stage1 native] dataset=$ds seed=$seed mode=$mode S=$SAMPLES R=$REPEATS"
        python "$HERE/stage1_evaluate.py" --mode "$mode" --dataset "$ds" \
          --shot "$SHOT" --seed "$seed" --gpu "$GPU" --samples "$SAMPLES" \
          --mc-repeats "$REPEATS" --sample-grid "$GRID" --chunk "$CHUNK" \
          --reference-batch-size "$REF_BATCH" --output-root "$OUTPUT_ROOT"
      done
    done
  done
fi

python "$HERE/stage1_make_table.py" --output-root "$OUTPUT_ROOT" \
  --datasets $DATASETS --seeds $SEEDS --modes $MODES --shot "$SHOT" \
  --samples "$SAMPLES" --mc-repeats "$REPEATS" --split "$SPLIT"

python "$HERE/stage1_plot.py" \
  --raw "$SUMMARY_DIR/table_a_native_${SPLIT}_raw.csv" \
  --convergence "$SUMMARY_DIR/table_a_native_${SPLIT}_convergence.csv" \
  --out-dir "$SUMMARY_DIR/figures"

echo "[OK] All Stage-1 summary outputs: $SUMMARY_DIR"
