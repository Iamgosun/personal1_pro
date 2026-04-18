#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-DATASETS}
DATASET=${DATASET:-caltech101}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}
SHOTS=${SHOTS:-16}
SEEDS=${SEEDS:-"1 2 3"}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_refactor}
TAG_PREFIX=${TAG_PREFIX:-bayes_sweep}

DATASET_CFG="configs/datasets/${DATASET}.yaml"
METHOD_CFG="configs/methods/bayesmmrl.yaml"
PROTOCOL_CFG="configs/protocols/fs.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

PHASE="fewshot_train"
BACKBONE_TAG="${BACKBONE//\//-}"
METHOD_ROOT="${OUTPUT_ROOT}/BayesMMRL/${PROTOCOL}/${PHASE}/${DATASET}/shots_${SHOTS}/${BACKBONE_TAG}"
GLOBAL_SUMMARY="${METHOD_ROOT}/sweep_summary.csv"

mkdir -p "${METHOD_ROOT}"

run_one() {
  local seed=$1
  local alpha=$2
  local kl=$3
  local prior=$4
  local tag=$5

  local outdir="${METHOD_ROOT}/${tag}/seed${seed}"

  echo "[RUN] seed=${seed} alpha=${alpha} kl=${kl} prior=${prior} tag=${tag}"

  python run.py \
    --root "${ROOT}" \
    --dataset-config-file "${DATASET_CFG}" \
    --method-config-file "${METHOD_CFG}" \
    --protocol-config-file "${PROTOCOL_CFG}" \
    --runtime-config-file "${RUNTIME_CFG}" \
    --output-dir "${outdir}" \
    --method BayesMMRL \
    --protocol "${PROTOCOL}" \
    --exec-mode "${EXEC_MODE}" \
    --seed "${seed}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES all \
    MODEL.BACKBONE.NAME "${BACKBONE}" \
    BAYES_MMRL.ALPHA "${alpha}" \
    BAYES_MMRL.KL_WEIGHT "${kl}" \
    BAYES_MMRL.PRIOR_STD "${prior}"
}

summarize_tag() {
  local tag=$1
  local tag_root="${METHOD_ROOT}/${tag}"

  python evaluation/result_parser.py "${tag_root}" --split test || true

  local summary_csv="${tag_root}/test_summary.csv"
  if [ ! -f "${summary_csv}" ]; then
    echo "[WARN] summary file not found: ${summary_csv}"
    return
  fi

  append_summary_to_global "${tag}" "${summary_csv}"
}

append_summary_to_global() {
  local tag=$1
  local summary_csv=$2

  local header
  header="$(head -n 1 "${summary_csv}")"

  if [ ! -f "${GLOBAL_SUMMARY}" ]; then
    echo "tag,alpha,kl_weight,prior_std,${header}" > "${GLOBAL_SUMMARY}"
  fi

  tail -n +2 "${summary_csv}" | while IFS= read -r line; do
    echo "${tag},${CURRENT_ALPHA},${CURRENT_KL},${CURRENT_PRIOR},${line}" >> "${GLOBAL_SUMMARY}"
  done
}

run_tag() {
  local alpha=$1
  local kl=$2
  local prior=$3

  CURRENT_ALPHA="${alpha}"
  CURRENT_KL="${kl}"
  CURRENT_PRIOR="${prior}"

  local tag="${TAG_PREFIX}_alpha_${alpha}_kl_${kl}_prior_${prior}"

  for seed in ${SEEDS}; do
    run_one "${seed}" "${alpha}" "${kl}" "${prior}" "${tag}"
  done

  summarize_tag "${tag}"
  echo "[DONE] ${tag}"
  echo
}

print_global_summary() {
  echo "========================================"
  echo "Global sweep summary saved to:"
  echo "${GLOBAL_SUMMARY}"
  echo "========================================"

  if [ -f "${GLOBAL_SUMMARY}" ]; then
    python - <<'PY'
import pandas as pd
from pathlib import Path

path = Path(r"""'"${GLOBAL_SUMMARY}"'""")
df = pd.read_csv(path)

preferred_cols = [
    "tag",
    "alpha",
    "kl_weight",
    "prior_std",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "ece_mean",
    "ece_std",
    "brier_mean",
    "brier_std",
]

keep = [c for c in preferred_cols if c in df.columns]
if keep:
    print(df[keep].sort_values(by=keep[4] if len(keep) > 4 else keep[0], ascending=False).to_string(index=False))
else:
    print(df.to_string(index=False))
PY
  fi
}

# =========================
# Round 1: sweep ALPHA
# =========================
BASE_KL=1e-5
BASE_PRIOR=0.1

for alpha in 0.4 0.5 0.6 0.7 0.8; do
  run_tag "${alpha}" "${BASE_KL}" "${BASE_PRIOR}"
done

# =========================
# Round 2: sweep KL_WEIGHT
# 手动把 BEST_ALPHA 改成第一轮最优
# =========================
BEST_ALPHA=${BEST_ALPHA:-0.6}

for kl in 5e-6 1e-5 5e-5 1e-4; do
  run_tag "${BEST_ALPHA}" "${kl}" "${BASE_PRIOR}"
done

# =========================
# Round 3: sweep PRIOR_STD
# 手动把 BEST_KL 改成第二轮最优
# =========================
BEST_KL=${BEST_KL:-5e-5}

for prior in 0.02 0.05 0.1 0.2; do
  run_tag "${BEST_ALPHA}" "${BEST_KL}" "${prior}"
done

print_global_summary