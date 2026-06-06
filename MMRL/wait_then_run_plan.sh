#!/usr/bin/env bash
set -euo pipefail

# Wait for run_bayesrt_independent_hpo_ablation.sh process group to finish,
# then run run_plan_clean.sh, then run run_plan.sh.
# Override OLD_PGID/SLEEP_SECONDS at runtime if needed:
#   OLD_PGID=734113 SLEEP_SECONDS=1800 bash wait_clean_then_run_plan_v2.sh
OLD_PGID="${OLD_PGID:-734113}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1800}"

TARGET_SCRIPT="run_bayesrt_independent_hpo_ablation.sh"
CLEAN_PLAN="./run_plan_clean.sh"
RUN_PLAN="./run_plan.sh"

echo "$(date) current conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "$(date) python: $(which python)"
python -V

if [[ ! -f "${CLEAN_PLAN}" ]]; then
  echo "$(date) ERROR: ${CLEAN_PLAN} not found in $(pwd)" >&2
  exit 1
fi

if [[ ! -f "${RUN_PLAN}" ]]; then
  echo "$(date) ERROR: ${RUN_PLAN} not found in $(pwd)" >&2
  exit 1
fi

echo "$(date) waiting for ${TARGET_SCRIPT} process group ${OLD_PGID} to finish..."

while pgrep -g "${OLD_PGID}" >/dev/null 2>&1; do
  echo "$(date) ${TARGET_SCRIPT} process group ${OLD_PGID} is still running. Next check in ${SLEEP_SECONDS}s."
  sleep "${SLEEP_SECONDS}"
done

echo "$(date) ${TARGET_SCRIPT} process group ${OLD_PGID} finished/gone."

echo "$(date) running ${CLEAN_PLAN}..."
bash "${CLEAN_PLAN}"
echo "$(date) ${CLEAN_PLAN} finished."

echo "$(date) running ${RUN_PLAN}..."
bash "${RUN_PLAN}"
echo "$(date) ${RUN_PLAN} finished."

echo "$(date) all scheduled scripts finished."
