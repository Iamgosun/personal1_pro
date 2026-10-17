#!/usr/bin/env bash
# DAVI/AVI E1-E5 shell orchestrator; run from any directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
if [[ ! -f core/registry.py || ! -f "$HERE/cli.py" ]]; then
  echo '[ERROR] Put the package at MMRL/experiments/supplementary/.' >&2; exit 2
fi

usage() {
cat <<'HELP'
DAVI-Adapter V7 E1-E5 shell launcher (native models + YAML, no config rewriting)
Usage: bash experiments/supplementary/run_study.sh STAGE
Stages:
  check    CPU tests + GPU model/gradient doctor for all three methods
  train    Native training: Original/Branch with trajectories, DAVI with init snapshot
  e1       Native training + offline responsibility/Rep-ACC trajectories + formal tests
  e2       DAVI MC objective/moment analysis plus gradient analysis (shared E2/E3 run)
  e3       Alias of e2; E2/E3 share a single DAVI evaluation to avoid overwrites
  e4       Complete 3-method posterior/representation diagnostics and formal tests
  e5       Train matched-time Branch/DAVI, evaluate their last checkpoints
  report   Rebuild Tables S1-S4 and Figures S1-S4 (no GPU training)
  all      Run check,e1,e2,e4,e5,report; REQUIRES BUDGET_SECONDS
  all_no_e5  Run check,e1,e2,e4,report without time-budget study
  help     Show this message

Environment variables (all optional unless noted):
  GPU=0                          physical GPU visible via CUDA_VISIBLE_DEVICES
  DATASETS='caltech101'          space-separated: caltech101 eurosat fgvc_aircraft
  SEEDS='1'                      space-separated seeds; formal: '1 2 3'
  SHOTS=16                       number of shots
  OUT=output_supplementary_v7     output root (relative to MMRL unless absolute)
  DATA_ROOT=DATASETS              image data root
  EPOCHS=''                      empty: use each method's native YAML
  MC_TRAIN=''                    empty: use native YAML, DAVI training remains sampling-free
  SAMPLES=1024 CHUNK=16 MC_REPEATS=2 MC_TOLERANCE=0.03
  GRAD_REF=512 GRAD_CHUNK=8 GRAD_REPEATS=5
  E1_SAMPLES=10 E1_BATCHES=4 BATCH_SIZE=32
                                   E1_BATCHES=0 means all fixed-training batches
  NATIVE_CHECKPOINT=latest         E2-E4 model selection; E1 always uses latest epoch
  BUDGET_SECONDS=...             REQUIRED for E5; choose after pilot, no guessed default
  SKIP_EXISTING=1                skip only verified completed outputs; no silent overwrite
  FORCE_EVAL=0                   1: explicitly recompute diagnostics/formal output
                                   E1 trace automatically recomputes if settings/snapshots differ
  DRY_RUN=0                      1: print commands without execution/GPU

Examples:
  bash experiments/supplementary/run_study.sh check
  GPU=0 DATASETS='caltech101' SEEDS='1' bash experiments/supplementary/run_study.sh e1
  GPU=0 DATASETS='caltech101' SEEDS='1' bash experiments/supplementary/run_study.sh e2
  GPU=0 BUDGET_SECONDS=300 bash experiments/supplementary/run_study.sh e5
  DATASETS='caltech101 eurosat fgvc_aircraft' SEEDS='1 2 3' bash experiments/supplementary/run_study.sh all_no_e5
HELP
}

STAGE="${1:-help}"
case "$STAGE" in check|train|e1|e2|e3|e4|e5|report|all|all_no_e5|help) ;; *) usage >&2;exit 2;; esac
if [[ "$STAGE" == help ]]; then usage;exit 0;fi

GPU="${GPU:-0}"
DATASETS="${DATASETS:-caltech101}"
SEEDS="${SEEDS:-1}"
SHOTS="${SHOTS:-16}"
OUT="${OUT:-output_supplementary_v7}"
DATA_ROOT="${DATA_ROOT:-DATASETS}"
EPOCHS="${EPOCHS:-}"
MC_TRAIN="${MC_TRAIN:-}"
SAMPLES="${SAMPLES:-1024}"
CHUNK="${CHUNK:-16}"
MC_REPEATS="${MC_REPEATS:-2}"
MC_TOLERANCE="${MC_TOLERANCE:-0.03}"
GRAD_REF="${GRAD_REF:-512}"
GRAD_CHUNK="${GRAD_CHUNK:-8}"
GRAD_REPEATS="${GRAD_REPEATS:-5}"
E1_SAMPLES="${E1_SAMPLES:-10}"
E1_BATCHES="${E1_BATCHES:-4}"
NATIVE_CHECKPOINT="${NATIVE_CHECKPOINT:-latest}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BUDGET_SECONDS="${BUDGET_SECONDS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
case "$NATIVE_CHECKPOINT" in
  latest|best) ;;
  *) echo '[ERROR] NATIVE_CHECKPOINT must be latest or best' >&2; exit 2;;
esac
if [[ "$STAGE" == all || "$STAGE" == e5 ]] && [[ -z "$BUDGET_SECONDS" ]]; then
  echo '[ERROR] Specify BUDGET_SECONDS (seconds) for E5; no arbitrary default.' >&2
  exit 2
fi
for var in SHOTS SAMPLES CHUNK MC_REPEATS GRAD_REF GRAD_CHUNK GRAD_REPEATS E1_SAMPLES BATCH_SIZE; do
  val="${!var}"
  if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then echo "[ERROR] $var must be positive integer: $val" >&2;exit 2;fi
done
if ! [[ "$E1_BATCHES" =~ ^(0|[1-9][0-9]*)$ ]];then
  echo "[ERROR] E1_BATCHES must be >=0: $E1_BATCHES" >&2;exit 2
fi
read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
if (( ${#DATASET_LIST[@]} == 0 || ${#SEED_LIST[@]} == 0 )); then
  echo '[ERROR] DATASETS and SEEDS cannot be empty.' >&2;exit 2
fi
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in caltech101|eurosat|fgvc_aircraft) ;; *) echo "[ERROR] Unsupported dataset: $dataset" >&2;exit 2;; esac
done

run() {
  printf '[RUN]'; printf ' %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" != 1 ]]; then "$@";fi
}

# Resolve path exactly the same way as MMRL/experiments/supplementary/core.py.
if [[ "$OUT" == /* ]];then BASE="$OUT";else BASE="$ROOT/$OUT";fi
native_dir() { printf '%s/native/%s/%s/shots_%s/seed_%s' "$BASE" "$1" "$2" "$SHOTS" "$3"; }
budget_dir() { printf '%s/matched_time/%s/%s/shots_%s/seed_%s' "$BASE" "$1" "$2" "$SHOTS" "$3"; }

train_args() {
  # Mutable array; EPOCHS and MC_TRAIN are deliberately omitted when unset.
  TRAIN_ARGS=(--dataset "$1" --method "$2" --shot "$SHOTS" --seed "$3"
       --gpu "$GPU" --output "$OUT" --data-root "$DATA_ROOT")
  if [[ -n "$EPOCHS" ]];then TRAIN_ARGS+=(--epochs "$EPOCHS");fi
  if [[ -n "$MC_TRAIN" ]];then TRAIN_ARGS+=(--mc-train "$MC_TRAIN");fi
}

validate_completed() {
  # Exit status 0 only for a completed, correctly identified run with a checkpoint.
  python - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json,sys,pathlib
path=pathlib.Path(sys.argv[1]);study,dataset,method,shot,seed=sys.argv[2:]
try:
    m=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='complete'
    assert (m['study'],m['dataset'],m['method'],int(m['shot']),int(m['seed']))==(study,dataset,method,int(shot),int(seed))
    assert (path/'refactor_model').is_dir()
    assert any((path/'refactor_model').glob('model.pth.tar-*')) or (path/'refactor_model'/'model-best.pth.tar').exists()
except (AssertionError,FileNotFoundError,KeyError,ValueError,TypeError):sys.exit(1)
PY
}

ensure_native() {
  local dataset="$1" method="$2" seed="$3" folder
  folder="$(native_dir "$dataset" "$method" "$seed")"
  # Original/Branch need epoch 0 and each epoch for E1; DAVI only epoch 0 for E4.
  local flag=(--trajectory)
  if [[ "$method" == davi ]];then flag=(--init-snapshot);fi
  if [[ -d "$folder" ]]; then
    if [[ "$DRY_RUN" == 1 ]];then echo "[DRY] Would verify existing run: $folder";return;fi
    if [[ "$SKIP_EXISTING" != 1 ]] || ! validate_completed "$folder" native "$dataset" "$method" "$SHOTS" "$seed";then
      echo "[ERROR] Existing native run is incomplete or not skippable: $folder" >&2;exit 3
    fi
    # Verify explicit overrides before reusing an existing checkpoint.
    python - "$folder/manifest.json" "$EPOCHS" "$MC_TRAIN" <<'PY_CHECK'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
def req(v):return int(v) if v else None
for key,v in [('epochs_override',sys.argv[2]),('mc_train_override',sys.argv[3])]:
    if m.get(key)!=req(v):
        raise SystemExit('Existing run differs on '+key+'; choose a new OUT.')
PY_CHECK
    # E1 deliberately permits a short / sparse trajectory. Missing epoch 0
    # or intermediate snapshots are not reasons to retrain a completed run.
    # E4 prompt-init comparison simply remains unavailable without epoch 0.
    echo "[SKIP completed native training] $folder"
    return
  fi
  train_args "$dataset" "$method" "$seed"
  run python -m experiments.supplementary.cli train "${TRAIN_ARGS[@]}" "${flag[@]}"
}

ensure_budget() {
  local dataset="$1" method="$2" seed="$3" folder
  folder="$(budget_dir "$dataset" "$method" "$seed")"
  if [[ -d "$folder" ]];then
    if [[ "$DRY_RUN" == 1 ]];then echo "[DRY] Would verify existing budget run: $folder";return;fi
    if [[ "$SKIP_EXISTING" != 1 ]] || ! validate_completed "$folder" matched_time "$dataset" "$method" "$SHOTS" "$seed";then
      echo "[ERROR] E5 existing run not completed: $folder" >&2;exit 3
    fi
    python - "$folder/manifest.json" "$BUDGET_SECONDS" "$EPOCHS" "$MC_TRAIN" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
if float(m.get('budget_seconds',-1))!=float(sys.argv[2]):
    raise SystemExit('Budget mismatch with existing E5 run. Choose a new OUT.')
def req(v):return int(v) if v else None
for key,v in [('epochs_override',sys.argv[3]),('mc_train_override',sys.argv[4])]:
    if m.get(key)!=req(v):
        raise SystemExit('E5 training override mismatch on '+key+'. Choose a new OUT.')
PY
    echo "[SKIP completed E5 training] $folder"
    return
  fi
  train_args "$dataset" "$method" "$seed"
  run python -m experiments.supplementary.cli budget "${TRAIN_ARGS[@]}" --budget-seconds "$BUDGET_SECONDS"
}

formal_path() {
  local root="$1" selection="$2" epoch="${3:-}"
  local p="$root/formal/$selection"
  if [[ -n "$epoch" ]];then p="$p/epoch_$epoch";fi
  echo "$p/complete/test.json"
}
reference_dir() {
  local root="$1" selection="$2" epoch="${3:-}"
  local p="$root/diagnostics/$selection"
  if [[ -n "$epoch" ]];then p="$p/epoch_$epoch";fi
  echo "$p/S${SAMPLES}_R${MC_REPEATS}"
}
latest_epoch() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
paths=list((Path(sys.argv[1])/'refactor_model').glob('model.pth.tar-*'))
if not paths:raise SystemExit('No E5 latest checkpoint')
print(max(int(x.name.rsplit('-',1)[-1]) for x in paths))
PY
}

ensure_formal() {
  local folder="$1" selection="$2" epoch="${3:-}" path
  if [[ "$selection" == best && "$DRY_RUN" != 1 && ! -f "$folder/refactor_model/model-best.pth.tar" ]];then
    echo "[ERROR] best checkpoint not found: $folder/refactor_model/model-best.pth.tar" >&2
    echo 'Set NATIVE_CHECKPOINT=latest for the last epoch; this is an explicit selection, not a fake best model.' >&2
    exit 3
  fi
  if [[ "$selection" == latest && -z "$epoch" ]];then
    if [[ "$DRY_RUN" == 1 ]];then epoch=DRY_EPOCH; else epoch="$(latest_epoch "$folder")";fi
  fi
  path="$(formal_path "$folder" "$selection" "$epoch")"
  if [[ -f "$path" && "$FORCE_EVAL" != 1 ]];then
    echo "[SKIP completed native formal inference] $path";return
  fi
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU" --mode formal)
  if [[ "$FORCE_EVAL" == 1 ]];then args+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${args[@]}"
}

ensure_reference() {
  local folder="$1" selection="$2" want_gradient="$3" want_vectors="$4" epoch="${5:-}" dir
  if [[ "$selection" == best && "$DRY_RUN" != 1 && ! -f "$folder/refactor_model/model-best.pth.tar" ]];then
    echo "[ERROR] best checkpoint not found: $folder/refactor_model/model-best.pth.tar" >&2
    echo 'Set NATIVE_CHECKPOINT=latest to evaluate last epoch.' >&2
    exit 3
  fi
  if [[ "$selection" == latest && -z "$epoch" ]];then
    if [[ "$DRY_RUN" == 1 ]];then epoch=DRY_EPOCH; else epoch="$(latest_epoch "$folder")";fi
  fi
  dir="$(reference_dir "$folder" "$selection" "$epoch")"
  local done=0
  if [[ -f "$dir/test/metrics.json" ]];then
    done=1
    if [[ "$want_vectors" == 1 && ! -f "$dir/test/moment_vectors.json" ]];then done=0;fi
    if [[ "$want_gradient" == 1 && ! -f "$dir/gradient.json" ]];then done=0;fi
  fi
  if [[ "$done" == 1 && "$FORCE_EVAL" != 1 ]];then echo "[SKIP completed reference] $dir";return;fi
  if [[ -d "$dir" && "$FORCE_EVAL" != 1 ]];then
    echo "[ERROR] Reference directory exists but requested outputs are missing: $dir" >&2
    echo 'Set FORCE_EVAL=1 to explicitly regenerate this reference, or choose another OUT.' >&2
    exit 3
  fi
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU"
        --mode reference --batch-size "$BATCH_SIZE" --samples "$SAMPLES"
        --chunk "$CHUNK" --mc-repeats "$MC_REPEATS" --mc-tolerance "$MC_TOLERANCE")
  if [[ "$want_vectors" == 1 ]];then args+=(--vectors);fi
  if [[ "$want_gradient" == 1 ]];then
    args+=(--gradient --grad-ref "$GRAD_REF" --grad-chunk "$GRAD_CHUNK"
            --grad-repeats "$GRAD_REPEATS")
  fi
  if [[ "$FORCE_EVAL" == 1 ]];then args+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${args[@]}"
}

run_check() {
  run python -m unittest discover -s experiments/supplementary/tests -v
  local dataset="${DATASET_LIST[0]}"
  for method in original branch davi;do
    run python -m experiments.supplementary.cli doctor --dataset "$dataset" --method "$method" \
      --shot "$SHOTS" --seed "${SEED_LIST[0]}" --gpu "$GPU" --output "$OUT" --data-root "$DATA_ROOT"
  done
}
run_train() {
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    for method in original branch davi;do ensure_native "$dataset" "$method" "$seed";done
  done;done
}
run_e1() {
  run_train
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    for method in original branch;do
      local folder;folder="$(native_dir "$dataset" "$method" "$seed")"
      local cache_current=0
      if [[ "$DRY_RUN" != 1 && "$FORCE_EVAL" != 1 ]];then
        if python - "$folder" "$E1_SAMPLES" "$E1_BATCHES" "$BATCH_SIZE" <<'PY_E1'
import sys
from experiments.supplementary.core import e1_trace_is_current
sys.exit(0 if e1_trace_is_current(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])) else 1)
PY_E1
        then cache_current=1;fi
      fi
      if [[ "$cache_current" == 1 ]];then
        echo "[SKIP unchanged E1 diagnostic] $folder"
      else
        echo "[E1] Computing trace using current samples=$E1_SAMPLES batches=$E1_BATCHES batch_size=$BATCH_SIZE (training unaffected)"
        run python -m experiments.supplementary.cli trace --run-dir "$folder" \
          --gpu "$GPU" --e1-samples "$E1_SAMPLES" --e1-batches "$E1_BATCHES" \
          --batch-size "$BATCH_SIZE"
      fi
    done
    for method in original branch davi;do
      local folder;folder="$(native_dir "$dataset" "$method" "$seed")"
      # E1 is a training-trajectory study, NOT validation-best selection.
      ensure_formal "$folder" latest
    done
  done;done
}
run_e23() {
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    ensure_native "$dataset" davi "$seed"
    local folder;folder="$(native_dir "$dataset" davi "$seed")"
    ensure_reference "$folder" "$NATIVE_CHECKPOINT" 1 1
  done;done
}
run_e4() {
  run_train
  run_e23
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    for method in original branch;do
      local folder;folder="$(native_dir "$dataset" "$method" "$seed")"
      ensure_reference "$folder" "$NATIVE_CHECKPOINT" 0 0
    done
    for method in original branch davi;do
      local folder;folder="$(native_dir "$dataset" "$method" "$seed")"
      ensure_formal "$folder" "$NATIVE_CHECKPOINT"
    done
  done;done
}
run_e5() {
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    for method in branch davi;do ensure_budget "$dataset" "$method" "$seed";done
    for method in branch davi;do
      local folder epoch;folder="$(budget_dir "$dataset" "$method" "$seed")"
      if [[ "$DRY_RUN" == 1 ]];then
        echo "[DRY] Would determine latest epoch from $folder";epoch=50
      else epoch="$(latest_epoch "$folder")";fi
      ensure_reference "$folder" latest 0 0 "$epoch"
      ensure_formal "$folder" latest "$epoch"
    done
  done;done
}
run_report() {
  run python -m experiments.supplementary.cli summarize --output "$OUT"
  run python -m experiments.supplementary.figures --output "$OUT"
}

case "$STAGE" in
  check) run_check;;
  train) run_train;;
  e1) run_e1;run_report;;
  e2|e3) run_e23;run_report;;
  e4) run_e4;run_report;;
  e5) run_e5;run_report;;
  report) run_report;;
  all) run_check;run_e1;run_e23;run_e4;run_e5;run_report;;
  all_no_e5) run_check;run_e1;run_e23;run_e4;run_report;;
esac
