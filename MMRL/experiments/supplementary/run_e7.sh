#!/usr/bin/env bash
# E7: local gradient->confidence directional effect on native AVI-B and DAVI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="${OUT:-${OUTPUT:-output_supplementary_v7}}"
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SEEDS="${SEEDS:-1 2 3}"
E7_EPOCHS="${E7_EPOCHS:-latest}"
E7_METHODS="${E7_METHODS:-branch davi}"
read -r -a datasets <<< "$DATASETS"
read -r -a seeds <<< "$SEEDS"
read -r -a epochs <<< "$E7_EPOCHS"
read -r -a methods <<< "$E7_METHODS"
args=(--output "$OUT" --gpu "${GPU:-0}" --datasets "${datasets[@]}" --seeds "${seeds[@]}" \
      --methods "${methods[@]}" --epochs "${epochs[@]}" --shot "${SHOTS:-16}" \
      --grad-batch "${E7_GRAD_BATCH:-32}" --input-seed "${E7_INPUT_SEED:-2027}" \
      --mc-draws "${E7_MC_DRAWS:-32}" --mc-seed "${E7_MC_SEED:-99173}" \
      --eval-images "${E7_EVAL_IMAGES:-128}" --eval-batch "${E7_EVAL_BATCH:-8}" \
      --eval-seed "${E7_EVAL_SEED:-2027}" --samples "${E7_SAMPLES:-50}" \
      --chunk "${E7_CHUNK:-10}" --eval-mc-seed "${E7_EVAL_MC_SEED:-81721}" \
      --relative-step "${E7_RELATIVE_STEP:-0.002}" \
      --conf-threshold "${E7_CONF_THRESHOLD:-0.9}" \
      --sign-threshold "${E7_SIGN_THRESHOLD:-1e-6}")
if [[ "${E7_NO_PARITY:-0}" == 1 ]];then args+=(--no-parity);fi
if [[ "${FORCE_EVAL:-0}" == 1 ]];then args+=(--overwrite);fi
if [[ "${E7_REPORT_ONLY:-0}" == 1 ]];then args=(--output "$OUT" --report);fi
if [[ "${E7_SELF_TEST:-0}" == 1 ]];then args=(--self-test);fi
python -m experiments.supplementary.e7 "${args[@]}"
