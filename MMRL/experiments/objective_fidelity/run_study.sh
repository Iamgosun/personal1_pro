#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PROFILE="${PROFILE:-native}"
DATASETS="${DATASETS:-caltech101 }"
SEEDS="${SEEDS:-1}"
MODES="${MODES:-davi branch_mc original_mc}"
SHOTS="${SHOTS:-16}"
SAMPLES="${SAMPLES:-10}"
CHUNK="${CHUNK:-16}"
OUT="${OUT:-output_objective_fidelity_v3}"
if [[ "$OUT" = /* ]]; then BASE="$OUT"; else BASE="$ROOT/$OUT"; fi
extra=()
if [[ -n "${EPOCHS:-}" ]]; then extra+=(--epochs "$EPOCHS"); fi
if [[ -n "${MC_TRAIN:-}" ]]; then extra+=(--mc-train "$MC_TRAIN"); fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for mode in $MODES; do
      folder="$BASE/$PROFILE/$mode/$dataset/shot_$SHOTS/seed$seed"
      if [[ -f "$folder/elapsed_wall_seconds.txt" ]]; then
        echo "[skip completed training] $folder"
      else
        python "$HERE/train.py" --mode "$mode" --profile "$PROFILE" --dataset "$dataset" \
          --shot "$SHOTS" --seed "$seed" --gpu "$GPU" --output-root "$OUT" "${extra[@]}"
      fi
      if [[ -f "$folder/fidelity.json" ]]; then
        echo "[skip completed diagnostics] $folder"
      else
        python "$HERE/evaluate.py" --mode "$mode" --profile "$PROFILE" --dataset "$dataset" \
          --shot "$SHOTS" --seed "$seed" --gpu "$GPU" --output-root "$OUT" \
          --samples "$SAMPLES" --chunk "$CHUNK"
      fi
    done
  done
done
python "$HERE/summarize.py" --profile "$PROFILE" --output-root "$OUT"
