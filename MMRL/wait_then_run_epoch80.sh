#!/usr/bin/env bash
set -euo pipefail

OLD_PID=574511
OLD_NAME="sweep_det_bayesrt_two_stage_full_confirm.sh"

echo "$(date) current conda env: ${CONDA_DEFAULT_ENV:-unknown}"
echo "$(date) python: $(which python)"
python -V

echo "$(date) waiting for PID ${OLD_PID} to finish..."

while kill -0 "$OLD_PID" 2>/dev/null; do
  if ! ps -p "$OLD_PID" -o args= | grep -q "$OLD_NAME"; then
    echo "$(date) PID ${OLD_PID} no longer matches ${OLD_NAME}; assuming old sweep is gone."
    break
  fi
  sleep 1800
done

echo "$(date) old sweep finished/gone. Starting MAX_EPOCH_LIST=80 sweep..."

MAX_EPOCH_LIST="80" \
OUTPUT_ROOT="output_sweeps/det_bayesrt_two_stage_80" \
bash sweep_det_bayesrt_two_stage_full_confirm.sh

echo "$(date) MAX_EPOCH_LIST=80 sweep finished."
