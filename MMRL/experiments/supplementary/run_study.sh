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
  train    Native training: Original/Branch/DAVI with full epoch trajectories
  e1       Native training + offline responsibility/Rep-ACC trajectories + formal tests
  e2       DAVI MC objective, Jensen gaps and logit-moment analysis only
  e3       Independent DAVI gradient analysis only; does not rerun E2
  e4       Three-method E4 + S9 epoch evolution (E4_EVOLUTION=1 by default)
  e5       Train matched-time Branch/DAVI, evaluate their last checkpoints
  report   Rebuild Tables S1-S4 and Figures S1-S4 (no GPU training)
  all      Run check,e1,e2,e3,e4,e5,report; REQUIRES BUDGET_SECONDS
  all_no_e5  Run check,e1,e2,e3,e4,report without time-budget study
  help     Show this message

Environment variables (all optional unless noted):
  GPU=0                          physical GPU visible via CUDA_VISIBLE_DEVICES
  DATASETS='caltech101'          space-separated native dataset config basenames (e.g. oxford_pets)
  SEEDS='1'                      space-separated seeds; formal: '1 2 3'
  SHOTS=16                       number of shots
  OUT=output_supplementary_v7     output root (relative to MMRL unless absolute)
  DATA_ROOT=DATASETS              image data root
  EPOCHS=''                      empty: use each method's native YAML
  MC_TRAIN=''                    empty: use native YAML, DAVI training remains sampling-free
  SAMPLES=1024 CHUNK=16 MC_REPEATS=2 MC_TOLERANCE=0.03
  E2_SPLITS="test"                 E2 endpoint reference split
  E2_GAP_EPOCHS="0 5 10 ... latest"     E2 pure Jensen gap stages
  E3_NOISE_EPOCHS="5 15 25 35 latest" E3 Branch gradient-noise checkpoints
  E3_NOISE_DRAWS=10 E3_NOISE_SEED=99173
  E5_CURVE=1                      E5 new budget runs save 0/25/50/75/100% snapshots
  E5_CURVE_SAMPLES=128 E5_CURVE_REPEATS=1
  # Existing completed runs WITHOUT intermediate snapshots are reused, not fabricated
  E2_SPLITS="test"                 optional: "train_fixed test" (separate diagnostics)
  GRAD_REF=512 GRAD_CHUNK=8 GRAD_REPEATS=5
  GRAD_SIZES='1 5 10 50'          MC budgets in the E3 curve
  GRAD_BATCH_STRATEGY=stratified  Select balanced classes from train_fixed
  GRAD_INPUT_SEED=2027           Reproducible fixed-training input selection
  GRAD_PARITY=1                  Verify full vs chunk gradient on same posterior draws
  E3_EPOCHS='latest'             E3 checkpoints: e.g. '5 25 latest'
                                   New E1/E2/E3/train runs save ALL native epochs
                                   for all three methods (init, 1..MAX_EPOCH).
                                   Existing runs are never retrained.
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
# caltech101 eurosat oxford_pets
DATASETS="${DATASETS:-caltech101 eurosat oxford_pets}"
SEEDS="${SEEDS:-1 2 3}"
SHOTS="${SHOTS:-16}"
OUT="${OUT:-output_supplementary_v7}"
DATA_ROOT="${DATA_ROOT:-DATASETS}"
EPOCHS="${EPOCHS:-}"
MC_TRAIN="${MC_TRAIN:-}"
SAMPLES="${SAMPLES:-1024}"
CHUNK="${CHUNK:-16}"
MC_REPEATS="${MC_REPEATS:-2}"
MC_TOLERANCE="${MC_TOLERANCE:-0.03}"
E2_SPLITS="${E2_SPLITS:-test}"
E2_GAP_EPOCHS="${E2_GAP_EPOCHS:-0 5 10 15 20 25 30 35 40 45 latest}"
E3_NOISE_EPOCHS="${E3_NOISE_EPOCHS:-5 15 25 35 latest}"
E3_NOISE_DRAWS="${E3_NOISE_DRAWS:-10}"
E3_NOISE_SEED="${E3_NOISE_SEED:-99173}"
E5_CURVE="${E5_CURVE:-1}"
E5_CURVE_SAMPLES="${E5_CURVE_SAMPLES:-128}"
E5_CURVE_REPEATS="${E5_CURVE_REPEATS:-1}"
GRAD_REF="${GRAD_REF:-512}"
GRAD_CHUNK="${GRAD_CHUNK:-8}"
GRAD_REPEATS="${GRAD_REPEATS:-5}"
GRAD_SIZES="${GRAD_SIZES:-1 5 10 50}"
GRAD_BATCH_STRATEGY="${GRAD_BATCH_STRATEGY:-stratified}"
GRAD_INPUT_SEED="${GRAD_INPUT_SEED:-2027}"
GRAD_PARITY="${GRAD_PARITY:-1}"
E3_EPOCHS="${E3_EPOCHS:-5 15 25 35 latest}"
# E4 lightweight stage diagnostics; these values do not affect native training.
E4_EVOLUTION="${E4_EVOLUTION:-1}"
E4_EVOLUTION_EPOCHS="${E4_EVOLUTION_EPOCHS:-0 5 13 25 38 latest}"
E4_EVOLUTION_IMAGES="${E4_EVOLUTION_IMAGES:-128}"
E4_EVOLUTION_DRAWS="${E4_EVOLUTION_DRAWS:-64}"
E4_EVOLUTION_CHUNK="${E4_EVOLUTION_CHUNK:-16}"
E4_EVOLUTION_BATCH="${E4_EVOLUTION_BATCH:-32}"
E4_EVOLUTION_INPUT_SEED="${E4_EVOLUTION_INPUT_SEED:-2027}"
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
for var in SHOTS SAMPLES CHUNK MC_REPEATS GRAD_REF GRAD_CHUNK GRAD_REPEATS E1_SAMPLES BATCH_SIZE E3_NOISE_DRAWS E5_CURVE_SAMPLES E5_CURVE_REPEATS E4_EVOLUTION_IMAGES E4_EVOLUTION_DRAWS E4_EVOLUTION_CHUNK E4_EVOLUTION_BATCH; do
  val="${!var}"
  if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then echo "[ERROR] $var must be positive integer: $val" >&2;exit 2;fi
done
if ! [[ "$E1_BATCHES" =~ ^(0|[1-9][0-9]*)$ ]];then
  echo "[ERROR] E1_BATCHES must be >=0: $E1_BATCHES" >&2;exit 2
fi
case "$E5_CURVE" in 0|1) ;; *) echo '[ERROR] E5_CURVE must be 0 or 1' >&2; exit 2;; esac
if ! [[ "$E3_NOISE_SEED" =~ ^[0-9]+$ ]];then echo '[ERROR] E3_NOISE_SEED must be nonnegative integer' >&2;exit 2;fi
read -r -a E2_GAP_EPOCH_LIST <<< "$E2_GAP_EPOCHS"
read -r -a E3_NOISE_EPOCH_LIST <<< "$E3_NOISE_EPOCHS"
read -r -a E4_EVOLUTION_EPOCH_LIST <<< "$E4_EVOLUTION_EPOCHS"
case "$E4_EVOLUTION" in 0|1) ;; *) echo '[ERROR] E4_EVOLUTION must be 0 or 1' >&2;exit 2;; esac
if ! [[ "$E4_EVOLUTION_INPUT_SEED" =~ ^[0-9]+$ ]]; then
  echo '[ERROR] E4_EVOLUTION_INPUT_SEED must be nonnegative' >&2;exit 2
fi
for ep in "${E2_GAP_EPOCH_LIST[@]}" "${E3_NOISE_EPOCH_LIST[@]}" "${E4_EVOLUTION_EPOCH_LIST[@]}";do
  if [[ "$ep" != latest && ! "$ep" =~ ^(0|[1-9][0-9]*)$ ]];then
    echo "[ERROR] Invalid diagnostic epoch '$ep'; use epoch number or latest" >&2;exit 2
  fi
done
case "$GRAD_BATCH_STRATEGY" in
  stratified|first) ;; *) echo '[ERROR] GRAD_BATCH_STRATEGY must be stratified or first' >&2;exit 2;;
esac
if ! [[ "$GRAD_INPUT_SEED" =~ ^[0-9]+$ ]];then
  echo '[ERROR] GRAD_INPUT_SEED must be a nonnegative integer' >&2;exit 2
fi
case "$GRAD_PARITY" in 0|1) ;; *) echo '[ERROR] GRAD_PARITY must be 0 or 1' >&2;exit 2;; esac
read -r -a GRAD_SIZES_LIST <<< "$GRAD_SIZES"
if (( ${#GRAD_SIZES_LIST[@]} == 0 ));then
  echo '[ERROR] GRAD_SIZES cannot be empty' >&2;exit 2
fi
for g in "${GRAD_SIZES_LIST[@]}";do
  if ! [[ "$g" =~ ^[1-9][0-9]*$ ]];then echo "[ERROR] GRAD_SIZES invalid: $g" >&2;exit 2;fi
done
GRAD_SIZE_TAG="$(IFS=-; echo "${GRAD_SIZES_LIST[*]}")"
read -r -a E3_EPOCH_LIST <<< "$E3_EPOCHS"
if (( ${#E3_EPOCH_LIST[@]} == 0 ));then
  echo '[ERROR] E3_EPOCHS cannot be empty' >&2; exit 2
fi
E3_SELECT_EPOCHS=()
E3_SEEN=" "
for ep in "${E3_EPOCH_LIST[@]}";do
  if [[ "$ep" != latest && ! "$ep" =~ ^[1-9][0-9]*$ ]];then
    echo '[ERROR] E3_EPOCHS accepts epoch numbers >=1 and latest: '"$ep" >&2;exit 2
  fi
  if [[ "$E3_SEEN" == *" $ep "* ]];then
    echo '[ERROR] E3_EPOCHS contains duplicate request: '"$ep" >&2;exit 2
  fi
  E3_SEEN+="$ep "
  if [[ "$ep" != latest ]];then E3_SELECT_EPOCHS+=("$ep");fi
done
read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a SEED_LIST <<< "$SEEDS"
if (( ${#DATASET_LIST[@]} == 0 || ${#SEED_LIST[@]} == 0 )); then
  echo '[ERROR] DATASETS and SEEDS cannot be empty.' >&2;exit 2
fi
# Supported datasets are determined by native MMRL configuration, not by
# an E1-E5-specific whitelist. This check also works in DRY_RUN mode.
for dataset in "${DATASET_LIST[@]}"; do
  if [[ ! "$dataset" =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "[ERROR] Invalid dataset config basename: $dataset (expected lowercase letters/digits/underscores)" >&2
    exit 2
  fi
  if [[ ! -f "$ROOT/configs/datasets/$dataset.yaml" ]]; then
    echo "[ERROR] Missing native dataset config: $ROOT/configs/datasets/$dataset.yaml" >&2
    echo '[HINT] DATASETS must contain config basenames, not DATASET.NAME. The dataset must also be registered in the native MMRL code and its images available under DATA_ROOT.' >&2
    exit 2
  fi
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
  local dataset="$1" method="$2" seed="$3" purpose="${4:-}" folder
  folder="$(native_dir "$dataset" "$method" "$seed")"
  # For E1/E2/E3 (including the `all` flow) and `train`, all three native
  # methods save the same initialization + every completed epoch protocol.
  # E4 also saves every epoch by default for S9, unless E4_EVOLUTION=0.
  local flag=(--trajectory)
  if [[ "$purpose" == e4 && "$E4_EVOLUTION" == 0 ]]; then
    flag=(--init-snapshot)
  fi
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
    # R10 requires a verifiable budget-stop checkpoint. A legacy E5 run
    # cannot be silently upgraded to a valid matched-time result.
    python - "$folder" <<'PY_E5_CACHE'
import json,sys
from pathlib import Path
from experiments.supplementary.core import sha256
folder=Path(sys.argv[1]); info=json.loads((folder/'training.json').read_text())
ck=folder/'refactor_model'/f'model.pth.tar-{info["last_checkpoint_epoch"]}'
if not ck.is_file() or not info.get('last_checkpoint_sha256') or \
   sha256(ck)!=info['last_checkpoint_sha256']:
    raise SystemExit('E5 existing budget checkpoint is unverified/legacy. '
                     'Preserve or rename its matched_time/.../seed_N directory, '
                     'then rerun E5 at the same OUT root. No overwrite performed.')
PY_E5_CACHE
    echo "[SKIP completed E5 training] $folder"
    if [[ "$E5_CURVE" == 1 && ! -f "$folder/time_curve/refactor_model/model.pth.tar-0" ]];then
      echo "[E5 CURVE WARNING] Existing E5 run has no time snapshots; cannot infer them from final checkpoint: $folder" >&2
    fi
    return
  fi
  train_args "$dataset" "$method" "$seed"
  local extra=()
  if [[ "$E5_CURVE" == 1 ]];then extra+=(--time-curve);fi
  run python -m experiments.supplementary.cli budget "${TRAIN_ARGS[@]}" --budget-seconds "$BUDGET_SECONDS" "${extra[@]}"
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
  if [[ -f "$path" && "$FORCE_EVAL" != 1 && "$DRY_RUN" != 1 ]] && \
      python - "$folder" "$selection" "$epoch" "$path" <<'PY_FORMAL_CACHE'
import json,sys
from pathlib import Path
from experiments.supplementary.core import checkpoint_path,sha256
from experiments.supplementary.reference_cache import formal_is_current
folder,selection,epoch,path=sys.argv[1:]
try:
    _,ck,_=checkpoint_path(folder,selection,int(epoch) if epoch else None)
    ok=formal_is_current(path,sha256(ck))
except (OSError,ValueError,KeyError,RuntimeError):ok=False
sys.exit(0 if ok else 1)
PY_FORMAL_CACHE
  then
    echo "[SKIP verified current native formal inference] $path";return
  fi
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU" --mode formal)
  if [[ "$FORCE_EVAL" == 1 || -f "$path" ]];then args+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${args[@]}"
}

ensure_reference() {
  local folder="$1" selection="$2" want_vectors="$3" epoch="${4:-}" splits="${5:-test}" dir
  if [[ "$selection" == best && "$DRY_RUN" != 1 && ! -f "$folder/refactor_model/model-best.pth.tar" ]];then
    echo "[ERROR] best checkpoint not found: $folder/refactor_model/model-best.pth.tar" >&2
    exit 3
  fi
  if [[ "$selection" == latest && -z "$epoch" ]];then
    if [[ "$DRY_RUN" == 1 ]];then epoch=DRY_EPOCH; else epoch="$(latest_epoch "$folder")";fi
  fi
  dir="$(reference_dir "$folder" "$selection" "$epoch")"
  local requested=() split
  read -r -a requested <<< "$splits"
  for split in "${requested[@]}";do
    case "$split" in train_fixed|test) ;; *) echo "[ERROR] Unknown reference split: $split" >&2;exit 2;; esac
  done
  # Do NOT skip merely because metrics.json exists. The training run may have
  # been repeated at the same seed/path. Verify actual checkpoint bytes, the
  # epoch-zero reference, MC settings, batch size and the E4 metric schema.
  if [[ "$DRY_RUN" != 1 && "$FORCE_EVAL" != 1 ]] && python - "$folder" "$selection" "$epoch"       "$SAMPLES" "$MC_REPEATS" "$BATCH_SIZE" "$MC_TOLERANCE" "$want_vectors" "${requested[@]}" <<'PY_REFERENCE_CACHE'
import sys
from pathlib import Path
from experiments.supplementary.core import checkpoint_path,sha256
from experiments.supplementary.reference_cache import initial_snapshot_hash,is_current
folder,selection,epoch,samples,repeats,batch,tol,vectors,*splits=sys.argv[1:]
try:
    _,ck,_=checkpoint_path(folder,selection,int(epoch) if epoch else None)
    ck_sha=sha256(ck)
    init_sha=initial_snapshot_hash(folder)
    base=Path(folder)/'diagnostics'/selection
    if epoch:base=base/f'epoch_{epoch}'
    base=base/f'S{samples}_R{repeats}'
    ok=all(is_current(base/split/'metrics.json',checkpoint_sha256=ck_sha,
        initial_sha256=init_sha,samples=int(samples),repeats=int(repeats),
        batch_size=int(batch),tolerance=float(tol),split=split,
        require_vectors=vectors=='1',require_e4=True) for split in splits)
except (OSError,ValueError,RuntimeError,FileNotFoundError):
    ok=False
sys.exit(0 if ok else 1)
PY_REFERENCE_CACHE
  then
    echo "[SKIP verified current reference] $dir ($splits)";return
  fi
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU"
        --mode reference --batch-size "$BATCH_SIZE" --samples "$SAMPLES"
        --chunk "$CHUNK" --mc-repeats "$MC_REPEATS" --mc-tolerance "$MC_TOLERANCE")
  if [[ "$selection" == trajectory || "$selection" == time_curve ]];then
    if [[ -z "$epoch" ]];then echo "[ERROR] Explicit epoch required for $selection reference" >&2;exit 3;fi
    args+=(--epoch "$epoch")
  fi
  args+=(--splits "${requested[@]}")
  if [[ "$want_vectors" == 1 ]];then args+=(--vectors);fi
  if [[ "$FORCE_EVAL" == 1 ]];then args+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${args[@]}"
}

ensure_gradient() {
  local folder="$1" selection="$2" epoch="${3:-}" base dir ck_path
  if [[ "$selection" == best && "$DRY_RUN" != 1 && ! -f "$folder/refactor_model/model-best.pth.tar" ]];then
    echo "[ERROR] best checkpoint not found: $folder/refactor_model/model-best.pth.tar" >&2
    exit 3
  fi
  if [[ "$selection" == latest && -z "$epoch" ]];then
    if [[ "$DRY_RUN" == 1 ]];then epoch=DRY_EPOCH; else epoch="$(latest_epoch "$folder")";fi
  fi
  if [[ "$selection" == trajectory && -z "$epoch" ]];then
    echo '[ERROR] E3 trajectory checkpoint requires epoch number' >&2;exit 2
  fi
  base="$folder/diagnostics/$selection"
  if [[ -n "$epoch" ]];then base="$base/epoch_$epoch";fi
  dir="$base/gradients/refS${GRAD_REF}_chunk${GRAD_CHUNK}_R${GRAD_REPEATS}_B${BATCH_SIZE}_sel${GRAD_BATCH_STRATEGY}_seed${GRAD_INPUT_SEED}_sizes${GRAD_SIZE_TAG}"
  if [[ "$selection" == best ]];then
    ck_path="$folder/refactor_model/model-best.pth.tar"
  elif [[ "$selection" == trajectory ]];then
    ck_path="$folder/trajectory/refactor_model/model.pth.tar-$epoch"
  else
    ck_path="$folder/refactor_model/model.pth.tar-$epoch"
  fi
  local needs_overwrite=0
  if [[ -f "$dir/gradient.json" && "$FORCE_EVAL" != 1 ]];then
    if python - "$dir/gradient.json" "$ck_path" "$GRAD_PARITY" <<'PY_GRAD'
import hashlib,json,sys
from pathlib import Path
try:
    obj=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    ck=Path(sys.argv[2]); sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    assert obj['checkpoint']['checkpoint_sha256']==sha
    assert bool(obj['e3_parameters']['chunk_gradient_parity_enabled'])==(sys.argv[3]=='1')
    assert obj['input']['input_sha256']
    assert obj['e3_output_tag']
except (AssertionError, KeyError, FileNotFoundError, ValueError, OSError):
    sys.exit(1)
PY_GRAD
    then echo "[SKIP matching E3 gradient] $dir";return
    else needs_overwrite=1;echo "[E3] Stale configuration or checkpoint; recomputing E3 only: $dir";fi
  fi
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU"
      --mode gradient --batch-size "$BATCH_SIZE" --grad-ref "$GRAD_REF"
      --grad-chunk "$GRAD_CHUNK" --grad-repeats "$GRAD_REPEATS"
      --grad-sizes "${GRAD_SIZES_LIST[@]}"
      --grad-batch-strategy "$GRAD_BATCH_STRATEGY"
      --grad-input-seed "$GRAD_INPUT_SEED")
  if [[ "$selection" == trajectory ]];then args+=(--epoch "$epoch");fi
  if [[ "$GRAD_PARITY" == 0 ]];then args+=(--skip-grad-parity);fi
  if [[ "$FORCE_EVAL" == 1 || "$needs_overwrite" == 1 ]];then args+=(--overwrite);fi
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
diagnostic_checkpoint_request() {
  local folder="$1" requested="$2"
  if [[ "$requested" == latest ]];then
    echo "latest:_";return
  fi
  if [[ "$DRY_RUN" == 1 ]];then echo "trajectory:$requested";return;fi
  local file="$folder/trajectory/refactor_model/model.pth.tar-$requested"
  if [[ -f "$file" ]];then echo "trajectory:$requested";return;fi
  local final
  final="$(latest_epoch "$folder")"
  if [[ "$requested" == "$final" ]];then echo "latest:_";return;fi
  echo "[ERROR] Checkpoint epoch $requested is missing: $file; final cannot reconstruct earlier weights" >&2
  return 2
}

ensure_gradient_noise() {
  local folder="$1" selection="$2" epoch="${3:-}" args
  local command=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU"
       --mode gradient_noise --noise-draws "$E3_NOISE_DRAWS" --noise-seed "$E3_NOISE_SEED"
       --grad-batch-strategy "$GRAD_BATCH_STRATEGY" --grad-input-seed "$GRAD_INPUT_SEED"
       --batch-size "$BATCH_SIZE")
  if [[ -n "$epoch" ]];then command+=(--epoch "$epoch");fi
  if [[ "$FORCE_EVAL" == 1 ]];then command+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${command[@]}"
}

run_e2() {
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    ensure_native "$dataset" davi "$seed"
    local folder;folder="$(native_dir "$dataset" davi "$seed")"
    ensure_reference "$folder" "$NATIVE_CHECKPOINT" 1 "" "$E2_SPLITS"
    local requested resolved selection epoch
    for requested in "${E2_GAP_EPOCH_LIST[@]}";do
      if ! resolved="$(diagnostic_checkpoint_request "$folder" "$requested")";then
        echo "[E2 GAP WARNING] Skipping missing checkpoint: $dataset seed=$seed epoch=$requested" >&2
        continue
      fi
      selection="${resolved%%:*}";epoch="${resolved#*:}"
      if [[ "$epoch" == _ ]];then epoch='';fi
      # New B4-like mechanism plot uses fixed training images at each epoch.
      ensure_reference "$folder" "$selection" 0 "$epoch" train_fixed
    done
  done;done
}
run_e3() {
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    ensure_native "$dataset" davi "$seed" e3
    local folder spec choice epoch resolved
    folder="$(native_dir "$dataset" davi "$seed")"
    # Resolve ALL requested checkpoints before launching an expensive gradient.
    # Do not substitute the final epoch for a missing intermediate checkpoint.
    local decisions=()
    for spec in "${E3_EPOCH_LIST[@]}";do
      if [[ "$DRY_RUN" == 1 ]];then
        if [[ "$spec" == latest ]];then
          decisions+=("${NATIVE_CHECKPOINT}:_")
        else
          decisions+=("trajectory:${spec}")
        fi
      else
        resolved="$(python - "$folder" "$spec" <<'PY_E3'
from experiments.supplementary.core import e3_checkpoint_request
import sys
selection,epoch=e3_checkpoint_request(sys.argv[1],sys.argv[2])
print(selection+':'+(str(epoch) if epoch is not None else '_'))
PY_E3
)" || exit 3
        decisions+=("$resolved")
      fi
    done
    for resolved in "${decisions[@]}";do
      choice="${resolved%%:*}";epoch="${resolved#*:}"
      if [[ "$epoch" == _ ]];then epoch='';fi
      echo "[E3] dataset=$dataset seed=$seed checkpoint=$choice ${epoch:+epoch=$epoch}"
      ensure_gradient "$folder" "$choice" "$epoch"
    done
    # Supplement to E3: posterior-draw noise of native Branch classification
    # gradients for Phi, R-mean, R-rho and T-rho, not DAVI-vs-MC cosine.
    ensure_native "$dataset" branch "$seed" e3_noise
    local other;other="$(native_dir "$dataset" branch "$seed")"
    local requested noise_spec noise_choice noise_epoch
    for requested in "${E3_NOISE_EPOCH_LIST[@]}";do
      if ! noise_spec="$(diagnostic_checkpoint_request "$other" "$requested")";then
        echo "[E3 NOISE WARNING] Skipping missing checkpoint: $dataset seed=$seed epoch=$requested" >&2
        continue
      fi
      noise_choice="${noise_spec%%:*}";noise_epoch="${noise_spec#*:}"
      if [[ "$noise_epoch" == _ ]];then noise_epoch='';fi
      ensure_gradient_noise "$other" "$noise_choice" "$noise_epoch"
    done
  done;done
}
ensure_e4_evolution() {
  local folder="$1" selection="$2" epoch="${3:-}"
  local args=(--run-dir "$folder" --checkpoint "$selection" --gpu "$GPU"
       --mode e4_evolution --e4-images "$E4_EVOLUTION_IMAGES"
       --e4-draws "$E4_EVOLUTION_DRAWS" --e4-chunk "$E4_EVOLUTION_CHUNK"
       --e4-batch "$E4_EVOLUTION_BATCH" --e4-input-seed "$E4_EVOLUTION_INPUT_SEED")
  if [[ -n "$epoch" ]];then args+=(--epoch "$epoch");fi
  if [[ "$FORCE_EVAL" == 1 ]];then args+=(--overwrite);fi
  run python -m experiments.supplementary.cli evaluate "${args[@]}"
}

run_e4() {
  # New E4 runs save all epochs if evolution is enabled; existing trained
  # weights are never overwritten merely to fill a missing historic epoch.
  for dataset in "${DATASET_LIST[@]}";do for seed in "${SEED_LIST[@]}";do
    local folder method
    for method in original branch davi;do
      ensure_native "$dataset" "$method" "$seed" e4
    done
    for method in original branch davi;do
      folder="$(native_dir "$dataset" "$method" "$seed")"
      # No moment_vectors.json needed for E4 ECDFs; E2 vectors already on disk
      # are reused if their source checkpoint/diagnostic settings match.
      if [[ "$method" == davi ]];then
        ensure_reference "$folder" "$NATIVE_CHECKPOINT" 1
      else
        ensure_reference "$folder" "$NATIVE_CHECKPOINT" 0
      fi
      ensure_formal "$folder" "$NATIVE_CHECKPOINT"
      if [[ "$E4_EVOLUTION" == 1 ]];then
        local requested resolved selection epoch
        for requested in "${E4_EVOLUTION_EPOCH_LIST[@]}";do
          if ! resolved="$(diagnostic_checkpoint_request "$folder" "$requested")";then
            echo "[E4 EVOLUTION WARNING] Missing historic checkpoint: $dataset $method seed=$seed epoch=$requested" >&2
            continue
          fi
          selection="${resolved%%:*}";epoch="${resolved#*:}"
          if [[ "$epoch" == _ ]];then epoch='';fi
          ensure_e4_evolution "$folder" "$selection" "$epoch"
        done
      fi
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
      ensure_reference "$folder" latest 0 "$epoch" "train_fixed test"
      ensure_formal "$folder" latest "$epoch"
      if [[ "$E5_CURVE" == 1 ]];then
        local phase cp
        for phase in 0 25 50 75 100;do
          cp="$folder/time_curve/refactor_model/model.pth.tar-$phase"
          if [[ "$DRY_RUN" == 1 || -f "$cp" ]];then
            # Curve diagnostic uses fewer posterior samples than terminal Table S4.
            SAMPLES="$E5_CURVE_SAMPLES" MC_REPEATS="$E5_CURVE_REPEATS" \
               ensure_reference "$folder" time_curve 0 "$phase" train_fixed
          fi
        done
      fi
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
  e2) run_e2;run_report;;
  e3) run_e3;run_report;;
  e4) run_e4;run_report;;
  e5) run_e5;run_report;;
  report) run_report;;
  all) run_check;run_e1;run_e2;run_e3;run_e4;run_e5;run_report;;
  all_no_e5) run_check;run_e1;run_e2;run_e3;run_e4;run_report;;
esac
