#!/usr/bin/env bash
set -euo pipefail

OLD_PGID="${OLD_PGID:-107252}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1800}"

PROTOCOL="${PROTOCOL:-FS}"
METHODS_ARG="${METHODS_ARG:-TipA TipA-f- CrossModal TR}"
EXEC_MODE="${EXEC_MODE:-cache}"

GPU_IDS="${GPU_IDS:-0 1 2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-4}"

echo "$(date) waiting for old run_plan.sh process group ${OLD_PGID} to finish..."

while pgrep -g "${OLD_PGID}" >/dev/null 2>&1; do
  echo "$(date) old process group ${OLD_PGID} is still running. Next check in ${SLEEP_SECONDS}s."
  sleep "${SLEEP_SECONDS}"
done

echo "$(date) old process group ${OLD_PGID} finished/gone."
echo "$(date) starting next run_plan.sh..."
echo "$(date) PROTOCOL=${PROTOCOL}"
echo "$(date) METHODS_ARG=${METHODS_ARG}"
echo "$(date) EXEC_MODE=${EXEC_MODE}"
echo "$(date) GPU_IDS=${GPU_IDS}"
echo "$(date) JOBS_PER_GPU=${JOBS_PER_GPU}"

GPU_IDS="${GPU_IDS}" \
JOBS_PER_GPU="${JOBS_PER_GPU}" \
bash ./run_plan.sh "${PROTOCOL}" "${METHODS_ARG}" "${EXEC_MODE}"

echo "$(date) next run_plan.sh finished."