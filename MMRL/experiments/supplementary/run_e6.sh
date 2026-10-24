#!/usr/bin/env bash
# Complete reconstructed E6: native pretrained checkpoint diagnostics only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SEEDS="${SEEDS:-1 2 3}"
E6_EPOCHS="${E6_EPOCHS:-5 25 latest}"
OUT="${OUT:-${OUTPUT:-output_supplementary_v7}}"
GPU="${GPU:-0}"
SHOTS="${SHOTS:-16}"
E6_MODE="${E6_MODE:-all}"
read -r -a datasets <<< "$DATASETS"
read -r -a seeds <<< "$SEEDS"
read -r -a epochs <<< "$E6_EPOCHS"
args=(--output "$OUT" --gpu "$GPU" --datasets "${datasets[@]}" --seeds "${seeds[@]}" --epochs "${epochs[@]}" --shot "$SHOTS" --mode "$E6_MODE" \
      --images "${E6_IMAGES:-128}" --batch "${E6_BATCH:-16}" --grad-batch "${E6_GRAD_BATCH:-32}" \
      --grad-draws "${E6_GRAD_DRAWS:-20}" --posterior-draws "${E6_POSTERIOR_DRAWS:-16}" \
      --mc-ref-draws "${E6_MC_REF_DRAWS:-128}" --mc-ref-repeats "${E6_MC_REF_REPEATS:-2}" \
      --prediction-epoch "${E6_PREDICTION_EPOCH:-latest}" \
      --prediction-batch "${E6_PREDICTION_BATCH:-32}" \
      --prediction-samples "${E6_PREDICTION_SAMPLES:-50}" \
      --prediction-chunk "${E6_PREDICTION_CHUNK:-10}" \
      --prediction-repeats "${E6_PREDICTION_REPEATS:-3}" \
      --sample-seed "${E6_SAMPLE_SEED:-99173}" --input-seed "${E6_INPUT_SEED:-2027}")
read -r -a scales <<< "${E6_SIGMA_SCALES:-0.5 2.0}"
args+=(--sigma-scales "${scales[@]}")
if [[ "${E6_NO_PHI:-0}" == "1" ]]; then args+=(--no-phi); fi
if [[ "${E6_MEAN_SWAP:-1}" == "0" ]]; then args+=(--no-mean-swap); fi
if [[ "${FORCE_EVAL:-0}" == "1" ]]; then args+=(--overwrite); fi
python -m experiments.supplementary.e6 "${args[@]}"
python -m experiments.supplementary.e6_figures --output "$OUT"
