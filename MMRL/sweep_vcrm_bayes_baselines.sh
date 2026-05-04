#!/bin/bash
set -euo pipefail

# ============================================================
# Joint sweep + confirm script for:
#   1) VCRMMMRL: tune VCRM_ETA and VCRM_MOD_WEIGHT on 16-shot seed1.
#      Required train batch size: 24.
#   2) BayesMMRL: tune REP_PRIOR_STD, REP_KL_WEIGHT,
#      MAIN_CONSISTENCY_MODE and MAIN_CONSISTENCY_WEIGHT on 16-shot seed1.
#      All other BayesMMRL settings, including batch size,
#      BAYES_TARGET and REP_SIGMA_MODE, come from the YAML defaults.
#   3) MMRL and BayesAdapter: confirm-only baselines, no tuning.
#
# After tuning, the script runs confirm experiments for all selected datasets,
# shots, seeds, and optionally B2N train_base + test_new.
#
# Scheduling policy:
#   VCRMMMRL is deferred until all other enabled experiments finish.
#   The default order is:
#     1) non-VCRMMMRL tune/confirm/B2N
#     2) VCRMMMRL tune/confirm/B2N
#
# Usage:
#   cd MMRL
#   bash sweep_vcrm_bayes_baselines.sh GPU_IDS="0 1 2 3" JOBS_PER_GPU=1
#
# Override examples:
#   bash sweep_vcrm_bayes_baselines.sh \
#     DATASETS="caltech101 oxford_pets ucf101" \
#     CONFIRM_SHOTS="1 2 4 8 16 32" \
#     CONFIRM_SEEDS="1 2 3" \
#     VCRM_ETA_LIST="0.05 0.1 0.2" \
#     VCRM_MOD_WEIGHT_LIST="0.0 1e-4 1e-3" \
#     BAYES_REP_PRIOR_STD_LIST="0.1 0.5 1.0" \
#     BAYES_REP_KL_WEIGHT_LIST="1e-3 1e-2 5e-2" \
#     BAYES_MAIN_CONSISTENCY_MODE_LIST="prob logit" \
#     BAYES_MAIN_CONSISTENCY_WEIGHT_LIST="0.0 0.01 0.03 0.1"
#
# Notes:
#   - Search/tune stage is always FS, 16-shot, seed1 unless overridden by
#     TUNE_SHOTS/TUNE_SEEDS.
#   - B2N uses the best hyperparameters selected from FS 16-shot seed1.
#   - BayesAdapter is launched through ClipAdapters with
#     configs/methods/clip_adapters_bayes.yaml.
# ============================================================

# ------------------------------------------------------------
# KEY=VALUE args
# ------------------------------------------------------------
apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      case "${key}" in
        PROJECT_DIR|ROOT|DATA_ROOT|OUTPUT_ROOT|BACKBONE|EXEC_MODE|BAYES_ADAPTER_EXEC_MODE|\
        DATASETS|TUNE_DATASETS|TUNE_SHOTS|TUNE_SEEDS|CONFIRM_DATASETS|CONFIRM_SHOTS|CONFIRM_SEEDS|\
        GPU_IDS|NGPU|JOBS_PER_GPU|SKIP_EXISTING|SLEEP_SEC|DELETE_CKPT_AFTER_TEST|\
        AUTO_TUNE|AUTO_CONFIRM_FS|AUTO_CONFIRM_B2N|RUN_VCRM|RUN_BAYES|RUN_MMRL|RUN_BAYES_ADAPTER|\
        VCRM_ETA_LIST|VCRM_MOD_WEIGHT_LIST|VCRM_BATCH_SIZE|\
        BAYES_REP_PRIOR_STD_LIST|BAYES_REP_KL_WEIGHT_LIST|BAYES_MAIN_CONSISTENCY_MODE_LIST|BAYES_MAIN_CONSISTENCY_WEIGHT_LIST|\
        SUMMARY_ONLY)
          printf -v "${key}" '%s' "${val}"
          export "${key}"
          ;;
        *)
          echo "[warn] unknown KEY=VALUE ignored: ${arg}" >&2
          ;;
      esac
    else
      echo "[warn] non KEY=VALUE ignored: ${arg}" >&2
    fi
  done
}

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
ROOT=${ROOT:-${DATA_ROOT:-DATASETS}}
DATA_ROOT=${DATA_ROOT:-${ROOT}}

BACKBONE=${BACKBONE:-ViT-B/16}
EXEC_MODE=${EXEC_MODE:-online}
BAYES_ADAPTER_EXEC_MODE=${BAYES_ADAPTER_EXEC_MODE:-cache}

OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/vcrm_bayes_joint}

# Default to all 11 datasets for final experiments.
DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft    oxford_pets  stanford_cars ucf101"}

# Tuning is FS 16-shot seed1 by requirement.
TUNE_DATASETS=${TUNE_DATASETS:-${DATASETS}}
TUNE_SHOTS=${TUNE_SHOTS:-"16"}
TUNE_SEEDS=${TUNE_SEEDS:-"1"}

# Confirm stage: all shots/seeds by default.
CONFIRM_DATASETS=${CONFIRM_DATASETS:-${DATASETS}}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-"1 2 4 8 16 32"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"1 2 3"}

GPU_IDS=${GPU_IDS:-}
NGPU=${NGPU:-}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}

AUTO_TUNE=${AUTO_TUNE:-1}
AUTO_CONFIRM_FS=${AUTO_CONFIRM_FS:-1}
AUTO_CONFIRM_B2N=${AUTO_CONFIRM_B2N:-1}

RUN_VCRM=${RUN_VCRM:-1}
RUN_BAYES=${RUN_BAYES:-1}
RUN_MMRL=${RUN_MMRL:-1}
RUN_BAYES_ADAPTER=${RUN_BAYES_ADAPTER:-1}

# Explicit search spaces. Change these lists if you want a larger/smaller sweep.
# MAIN_CONSISTENCY_WEIGHT default list is explicit here: 0.0 0.01 0.03 0.1.
VCRM_ETA_LIST=${VCRM_ETA_LIST:-"0.05 0.1 0.2"}
VCRM_MOD_WEIGHT_LIST=${VCRM_MOD_WEIGHT_LIST:-"0.0 1e-4 1e-3"}

BAYES_REP_PRIOR_STD_LIST=${BAYES_REP_PRIOR_STD_LIST:-"0.02 0.1 0.5 1.0"}
BAYES_REP_KL_WEIGHT_LIST=${BAYES_REP_KL_WEIGHT_LIST:-"1e-3 1e-2 5e-2 1e-1"}
BAYES_MAIN_CONSISTENCY_MODE_LIST=${BAYES_MAIN_CONSISTENCY_MODE_LIST:-"prob logit"}

BAYES_MAIN_CONSISTENCY_WEIGHT_LIST=${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST:-"0.0 0.01 0.03 "}
# Required VCRMMMRL batch size.
# Other non-swept settings come from method/runtime YAML defaults.
VCRM_BATCH_SIZE=${VCRM_BATCH_SIZE:-24}

apply_kv_args "$@"

TUNE_ROOT="${OUTPUT_ROOT}/tune_16shot_seed1"
CONFIRM_ROOT="${OUTPUT_ROOT}/confirm"
B2N_ROOT="${OUTPUT_ROOT}/b2n"

MANIFEST="${OUTPUT_ROOT}/run_manifest.csv"
TUNE_SUMMARY="${OUTPUT_ROOT}/tune_summary.csv"
BEST_SUMMARY="${OUTPUT_ROOT}/best_config_per_method_dataset.csv"
BEST_ENV="${OUTPUT_ROOT}/best_config_per_method_dataset.env"
CONFIRM_SUMMARY="${CONFIRM_ROOT}/confirm_summary.csv"
B2N_SUMMARY="${B2N_ROOT}/b2n_summary.csv"

FAILED_JOBS=0
READY_SLOT=""

declare -ga PHYSICAL_GPUS
declare -ga GPU_LIST
declare -ga RUNNING_PIDS
declare -ga SLOT_GPU
declare -ga SLOT_WEIGHT
declare -ga SLOT_DESC
declare -ga SLOT_LOG
declare -gA GPU_USED

declare -ga VCRM_TAGS
declare -gA VCRM_OPTS
declare -ga BAYES_TAGS
declare -gA BAYES_OPTS
declare -gA BEST_TAG_BY_METHOD_DATASET

# ------------------------------------------------------------
# Method/config resolution
# ------------------------------------------------------------
method_key() {
  local method=$1
  echo "$method" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g'
}

is_clip_adapter_alias() {
  local method=$1
  local key
  key="$(method_key "${method}")"
  [[ "${key}" == "bayesadapter" || "${key}" == "bayes_adapter" ]]
}

resolve_launch_method() {
  local method=$1
  if is_clip_adapter_alias "${method}"; then
    echo "ClipAdapters"
  else
    echo "${method}"
  fi
}

resolve_method_cfg() {
  local method=$1
  case "${method}" in
    VCRMMMRL)
      echo "configs/methods/vcrm_mmrl.yaml"
      ;;
    BayesMMRL)
      echo "configs/methods/bayesmmrl.yaml"
      ;;
    MMRL)
      echo "configs/methods/mmrl.yaml"
      ;;
    BayesAdapter|bayesadapter|bayes_adapter)
      echo "configs/methods/clip_adapters_bayes.yaml"
      ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 1
      ;;
  esac
}

resolve_runtime_cfg() {
  local method=$1
  case "${method}" in
    VCRMMMRL|BayesMMRL|MMRL)
      echo "configs/runtime/mmrl_family.yaml"
      ;;
    BayesAdapter|bayesadapter|bayes_adapter)
      echo "configs/runtime/adapter_family.yaml"
      ;;
    *)
      echo "configs/runtime/default.yaml"
      ;;
  esac
}

resolve_exec_mode() {
  local method=$1
  if is_clip_adapter_alias "${method}"; then
    echo "${BAYES_ADAPTER_EXEC_MODE}"
  else
    echo "${EXEC_MODE}"
  fi
}

resolve_phase_semantics() {
  local protocol=$1
  case "${protocol}" in
    FS)  echo "fewshot_train all configs/protocols/fs.yaml" ;;
    B2N) echo "train_base base configs/protocols/b2n.yaml" ;;
    CD)  echo "cross_train all configs/protocols/cd.yaml" ;;
    *)
      echo "Unknown protocol: ${protocol}" >&2
      exit 1
      ;;
  esac
}

resolve_run_tag_from_cfg() {
  local method=$1
  local method_cfg=$2

  python - <<PY
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

path = Path("${method_cfg}")
tag = "default"

if yaml is not None and path.exists():
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    method_cfg = cfg.get("METHOD", {}) or {}
    cad = cfg.get("CLIP_ADAPTERS", {}) or {}
    tag = method_cfg.get("TAG") or cad.get("INIT") or tag

if "${method}" in {"BayesAdapter", "bayesadapter", "bayes_adapter"} and tag == "default":
    tag = "BAYES_ADAPTER"

print(tag)
PY
}

sanitize() {
  local s="$1"
  s="${s//\//-}"
  s="${s// /_}"
  s="${s//,/}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s//:/-}"
  s="${s//|/-}"
  echo "${s}"
}

backbone_tag() {
  echo "${BACKBONE//\//-}"
}

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
build_case_root() {
  local base_root=$1
  local method=$2
  local protocol=$3
  local phase=$4
  local dataset=$5
  local shot=$6
  local tag=$7

  local launch_method
  launch_method="$(resolve_launch_method "${method}")"

  if [[ "${launch_method}" == "ClipAdapters" ]]; then
    echo "${base_root}/${launch_method}/${tag}/${protocol}/${phase}/${dataset}/shots_${shot}/$(backbone_tag)"
  else
    echo "${base_root}/${launch_method}/${protocol}/${phase}/${dataset}/shots_${shot}/$(backbone_tag)/${tag}"
  fi
}

build_outdir() {
  local base_root=$1
  local method=$2
  local protocol=$3
  local phase=$4
  local dataset=$5
  local shot=$6
  local seed=$7
  local tag=$8

  echo "$(build_case_root "${base_root}" "${method}" "${protocol}" "${phase}" "${dataset}" "${shot}" "${tag}")/seed${seed}"
}

build_b2n_train_outdir() {
  local base_root=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5
  local tag=$6

  echo "$(build_outdir "${base_root}" "${method}" B2N train_base "${dataset}" "${shot}" "${seed}" "${tag}")"
}

build_b2n_test_new_outdir() {
  local base_root=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local seed=$5
  local tag=$6

  local launch_method
  launch_method="$(resolve_launch_method "${method}")"

  if [[ "${launch_method}" == "ClipAdapters" ]]; then
    echo "${base_root}/${launch_method}/${tag}/B2N/test_new/${dataset}/shots_${shot}/$(backbone_tag)/seed${seed}"
  else
    echo "${base_root}/${launch_method}/B2N/test_new/${dataset}/shots_${shot}/$(backbone_tag)/${tag}/seed${seed}"
  fi
}

# ------------------------------------------------------------
# GPU scheduler
# ------------------------------------------------------------
init_gpu_list() {
  PHYSICAL_GPUS=()

  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a PHYSICAL_GPUS <<< "${GPU_IDS}"
  elif [[ -n "${NGPU}" ]]; then
    local i
    for ((i=0; i<NGPU; i++)); do
      PHYSICAL_GPUS+=("${i}")
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r idx; do
      [[ -n "${idx}" ]] && PHYSICAL_GPUS+=("${idx}")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  fi

  if [[ ${#PHYSICAL_GPUS[@]} -eq 0 ]]; then
    echo "No visible GPU found. Set GPU_IDS or NGPU." >&2
    exit 1
  fi

  if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid JOBS_PER_GPU=${JOBS_PER_GPU}" >&2
    exit 1
  fi

  # Keep GPU_LIST as an alias for backwards-compatible debug printing.
  GPU_LIST=("${PHYSICAL_GPUS[@]}")

  GPU_USED=()
  local gpu
  for gpu in "${PHYSICAL_GPUS[@]}"; do
    GPU_USED["${gpu}"]=0
  done

  echo "[GPU] physical: ${PHYSICAL_GPUS[*]}"
  echo "[GPU] default jobs_per_gpu: ${JOBS_PER_GPU}"
  echo "[GPU] VCRMMMRL jobs_per_gpu: 1"
  echo "[GPU] scheduling rule: VCRMMMRL consumes a full GPU; all other methods consume one slot"
}

init_slots() {
  local nslots=$((${#PHYSICAL_GPUS[@]} * JOBS_PER_GPU))
  local i
  RUNNING_PIDS=()
  SLOT_GPU=()
  SLOT_WEIGHT=()
  SLOT_DESC=()
  SLOT_LOG=()
  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_WEIGHT[$i]=""
    SLOT_DESC[$i]=""
    SLOT_LOG[$i]=""
  done
}


cleanup_children() {
  local p
  for p in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${p:-}" ]] && kill -0 "${p}" 2>/dev/null; then
      kill "${p}" 2>/dev/null || true
    fi
  done
}

job_slot_weight() {
  local method=$1

  if [[ "${method}" == "VCRMMMRL" ]]; then
    # Requirement: VCRMMMRL must run at one job per GPU.
    # With weighted scheduling, it consumes the full per-GPU capacity.
    echo "${JOBS_PER_GPU}"
  else
    # Other methods may run up to JOBS_PER_GPU jobs per GPU.
    echo "1"
  fi
}

_find_empty_process_slot() {
  local idx
  for idx in "${!RUNNING_PIDS[@]}"; do
    if [[ -z "${RUNNING_PIDS[$idx]}" ]]; then
      echo "${idx}"
      return 0
    fi
  done
  return 1
}

_reap_finished_jobs() {
  local idx pid rc gpu weight
  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"

    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      gpu="${SLOT_GPU[$idx]}"
      weight="${SLOT_WEIGHT[$idx]:-0}"

      if [[ -n "${gpu}" ]]; then
        GPU_USED["${gpu}"]=$(( ${GPU_USED["${gpu}"]:-0} - weight ))
        if [[ ${GPU_USED["${gpu}"]} -lt 0 ]]; then
          GPU_USED["${gpu}"]=0
        fi
      fi

      if [[ "${rc}" -eq 0 ]]; then
        echo "[done] ${SLOT_DESC[$idx]}"
      else
        echo "[failed] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_WEIGHT[$idx]=""
      SLOT_DESC[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}

READY_GPU=""
READY_WEIGHT=""

wait_for_compatible_slot() {
  local method=$1
  local weight
  weight="$(job_slot_weight "${method}")"

  READY_SLOT=""
  READY_GPU=""
  READY_WEIGHT="${weight}"

  while true; do
    _reap_finished_jobs

    local gpu used slot
    for gpu in "${PHYSICAL_GPUS[@]}"; do
      used="${GPU_USED["${gpu}"]:-0}"
      if (( used + weight <= JOBS_PER_GPU )); then
        slot="$(_find_empty_process_slot || true)"
        if [[ -n "${slot}" ]]; then
          READY_SLOT="${slot}"
          READY_GPU="${gpu}"
          return 0
        fi
      fi
    done

    sleep "${SLEEP_SEC}"
  done
}

wait_all_jobs() {
  local idx pid rc gpu weight
  for idx in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$idx]}"
    if [[ -n "${pid}" ]]; then
      rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      gpu="${SLOT_GPU[$idx]}"
      weight="${SLOT_WEIGHT[$idx]:-0}"

      if [[ -n "${gpu}" ]]; then
        GPU_USED["${gpu}"]=$(( ${GPU_USED["${gpu}"]:-0} - weight ))
        if [[ ${GPU_USED["${gpu}"]} -lt 0 ]]; then
          GPU_USED["${gpu}"]=0
        fi
      fi

      if [[ "${rc}" -eq 0 ]]; then
        echo "[done] ${SLOT_DESC[$idx]}"
      else
        echo "[failed] ${SLOT_DESC[$idx]} log=${SLOT_LOG[$idx]}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_WEIGHT[$idx]=""
      SLOT_DESC[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}


launch_background() {
  local gpu_id=$1
  local logfile=$2
  local desc=$3
  shift 3

  # launch_background is retained for compatibility. It uses default one-slot
  # scheduling because it does not receive a method argument.
  wait_for_compatible_slot "__generic__"
  local slot="${READY_SLOT}"
  gpu_id="${READY_GPU}"

  (
    "$@"
  ) &

  RUNNING_PIDS[$slot]=$!
  SLOT_GPU[$slot]="${gpu_id}"
  SLOT_WEIGHT[$slot]="${READY_WEIGHT}"
  GPU_USED["${gpu_id}"]=$(( ${GPU_USED["${gpu_id}"]:-0} + READY_WEIGHT ))
  SLOT_DESC[$slot]="${desc}"
  SLOT_LOG[$slot]="${logfile}"

  echo "[launch] slot=${slot} gpu=${gpu_id} pid=${RUNNING_PIDS[$slot]} ${desc}"
  echo "         log=${logfile}"
}

# ------------------------------------------------------------
# Manifest and summaries
# ------------------------------------------------------------
ensure_manifest_header() {
  mkdir -p "$(dirname "${MANIFEST}")"
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "stage,method,protocol,phase,subsample,dataset,shot,seed,tag,outdir" > "${MANIFEST}"
  fi
}

append_manifest() {
  local stage=$1 method=$2 protocol=$3 phase=$4 subsample=$5 dataset=$6 shot=$7 seed=$8 tag=$9 outdir=${10}
  ensure_manifest_header
  echo "${stage},${method},${protocol},${phase},${subsample},${dataset},${shot},${seed},${tag},${outdir}" >> "${MANIFEST}"
}

summarize_manifest() {
  local manifest=$1
  local summary_csv=$2

  python - <<PY
import csv
import json
from pathlib import Path

manifest = Path(r"${manifest}")
out = Path(r"${summary_csv}")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
if manifest.exists():
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metrics_path = Path(row["outdir"]) / "test_metrics.json"
            status = "ok" if metrics_path.exists() else "missing"
            metrics = {}
            num_samples = ""
            if metrics_path.exists():
                try:
                    data = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics = data.get("metrics", {}) or {}
                    num_samples = data.get("num_samples", "")
                except Exception as exc:
                    status = f"error:{type(exc).__name__}"

            rows.append({
                **row,
                "status": status,
                "num_samples": num_samples,
                "accuracy": metrics.get("accuracy", ""),
                "error": metrics.get("error", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "ece": metrics.get("ece", ""),
                "nll": metrics.get("nll", ""),
                "brier": metrics.get("brier", ""),
                "metrics_path": str(metrics_path),
            })

fieldnames = [
    "stage", "method", "protocol", "phase", "subsample",
    "dataset", "shot", "seed", "tag", "outdir", "status", "num_samples",
    "accuracy", "error", "macro_f1", "ece", "nll", "brier", "metrics_path",
]
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[summary] wrote {out} rows={len(rows)}")
PY
}

select_best_from_tune_summary() {
  local tune_summary=$1
  local best_summary=$2
  local best_env=$3

  python - <<PY
import csv
import math
from pathlib import Path
from collections import defaultdict

summary = Path(r"${tune_summary}")
best_csv = Path(r"${best_summary}")
best_env = Path(r"${best_env}")

rows = []
with summary.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("stage") != "tune":
            continue
        if row.get("protocol") != "FS":
            continue
        if row.get("shot") != "16" or row.get("seed") != "1":
            continue
        if row.get("method") not in {"VCRMMMRL", "BayesMMRL"}:
            continue
        if row.get("status") != "ok":
            continue
        try:
            row["_acc"] = float(row.get("accuracy", "nan"))
        except Exception:
            row["_acc"] = float("nan")
        try:
            row["_ece"] = float(row.get("ece", "nan"))
        except Exception:
            row["_ece"] = float("nan")
        if math.isfinite(row["_acc"]):
            rows.append(row)

groups = defaultdict(list)
for row in rows:
    groups[(row["method"], row["dataset"])].append(row)

best_rows = []
for key, group in sorted(groups.items()):
    # Primary: accuracy high. Secondary: ECE low. Third: tag lexical.
    group = sorted(
        group,
        key=lambda r: (
            -r["_acc"],
            r["_ece"] if math.isfinite(r["_ece"]) else float("inf"),
            r["tag"],
        ),
    )
    best_rows.append(group[0])

best_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "method", "dataset", "tag", "accuracy", "ece", "nll", "brier",
    "outdir", "metrics_path",
]
with best_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in best_rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

lines = [
    "declare -gA BEST_TAG_BY_METHOD_DATASET",
]
for r in best_rows:
    key = f"{r['method']}|{r['dataset']}"
    lines.append(f"BEST_TAG_BY_METHOD_DATASET[{shq(key)}]={shq(r['tag'])}")
lines.append("")
best_env.write_text("\n".join(lines), encoding="utf-8")

print(f"[best] wrote {best_csv}")
print(f"[best] wrote {best_env}")
for r in best_rows:
    print(f"[best] method={r['method']} dataset={r['dataset']} tag={r['tag']} acc={r.get('accuracy')} ece={r.get('ece')}")
PY
}

load_best_env() {
  if [[ ! -f "${BEST_ENV}" ]]; then
    echo "[error] missing ${BEST_ENV}. Run tuning first or provide the env file." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${BEST_ENV}"
}

# ------------------------------------------------------------
# Search spaces
# ------------------------------------------------------------
register_case() {
  local array_name=$1
  local assoc_name=$2
  local raw_tag=$3
  shift 3
  local opts_str="$*"
  local tag
  tag="$(sanitize "${raw_tag}")"
  eval "${array_name}+=(\"\${tag}\")"
  eval "${assoc_name}[\"\${tag}\"]=\"\${opts_str}\""
}

init_search_spaces() {
  VCRM_TAGS=()
  BAYES_TAGS=()
  VCRM_OPTS=()
  BAYES_OPTS=()

  local eta mod prior_std kl mode

  for eta in ${VCRM_ETA_LIST}; do
    for mod in ${VCRM_MOD_WEIGHT_LIST}; do
      register_case VCRM_TAGS VCRM_OPTS \
        "vcrm_eta-${eta}_mod-${mod}" \
        VCRM_MMRL.VCRM_ETA "${eta}" \
        VCRM_MMRL.VCRM_MOD_WEIGHT "${mod}" \
        DATALOADER.TRAIN_X.BATCH_SIZE "${VCRM_BATCH_SIZE}" \
        DATALOADER.TRAIN_U.BATCH_SIZE "${VCRM_BATCH_SIZE}"
    done
  done

  local cons_weight
  for prior_std in ${BAYES_REP_PRIOR_STD_LIST}; do
    for kl in ${BAYES_REP_KL_WEIGHT_LIST}; do
      for mode in ${BAYES_MAIN_CONSISTENCY_MODE_LIST}; do
        for cons_weight in ${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST}; do
          register_case BAYES_TAGS BAYES_OPTS \
            "bayes_pstd-${prior_std}_kl-${kl}_cons-${mode}_consw-${cons_weight}" \
            BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
            BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
            BAYES_MMRL.MAIN_CONSISTENCY_MODE "${mode}" \
            BAYES_MMRL.MAIN_CONSISTENCY_WEIGHT "${cons_weight}"
        done
      done
    done
  done

  echo "[search] VCRMMMRL cases: ${#VCRM_TAGS[@]}"
  echo "[search] BayesMMRL cases: ${#BAYES_TAGS[@]}"
  echo "[search] only swept params are overridden; non-swept settings use YAML defaults"
}

get_opts_for_method_tag() {
  local method=$1
  local tag=$2
  case "${method}" in
    VCRMMMRL)
      echo "${VCRM_OPTS[$tag]:-}"
      ;;
    BayesMMRL)
      echo "${BAYES_OPTS[$tag]:-}"
      ;;
    *)
      echo ""
      ;;
  esac
}

default_tag_for_method() {
  local method=$1
  local cfg
  cfg="$(resolve_method_cfg "${method}")"
  resolve_run_tag_from_cfg "${method}" "${cfg}"
}

# ------------------------------------------------------------
# Run commands
# ------------------------------------------------------------
checkpoint_ready() {
  local outdir=$1
  local ckpt_dir="${outdir}/refactor_model"
  [[ -f "${ckpt_dir}/checkpoint" ]] && compgen -G "${ckpt_dir}/model*.pth.tar*" >/dev/null
}

cleanup_checkpoint_if_ready() {
  local outdir=$1
  local logfile=${2:-}
  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${outdir}/test_metrics.json" && -d "${outdir}/refactor_model" ]]; then
    rm -rf "${outdir}/refactor_model"
    if [[ -n "${logfile}" ]]; then
      echo "[cleanup] removed ${outdir}/refactor_model" >> "${logfile}"
    fi
  fi
}

cleanup_b2n_train_checkpoint_after_test_new() {
  local train_outdir=$1
  local eval_outdir=$2
  local logfile=${3:-}

  # B2N rule:
  #   keep train_base/refactor_model until test_new has completed.
  #   delete it only after test_new/test_metrics.json exists.
  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi

  if [[ ! -f "${eval_outdir}/test_metrics.json" ]]; then
    if [[ -n "${logfile}" ]]; then
      echo "[cleanup] keep ${train_outdir}/refactor_model because test_new metrics are missing" >> "${logfile}"
    fi
    return 0
  fi

  if [[ -d "${train_outdir}/refactor_model" ]]; then
    rm -rf "${train_outdir}/refactor_model"
    if [[ -n "${logfile}" ]]; then
      echo "[cleanup] removed ${train_outdir}/refactor_model after test_new completed" >> "${logfile}"
    fi
  fi
}

write_log_header() {
  local logfile=$1
  local gpu_id=$2
  local stage=$3
  local method=$4
  local protocol=$5
  local phase=$6
  local subsample=$7
  local dataset=$8
  local shot=$9
  local seed=${10}
  local tag=${11}
  local outdir=${12}

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: ${stage}"
    echo "GPU: ${gpu_id}"
    echo "METHOD: ${method}"
    echo "LAUNCH_METHOD: $(resolve_launch_method "${method}")"
    echo "PROTOCOL: ${protocol}"
    echo "PHASE: ${phase}"
    echo "SUBSAMPLE: ${subsample}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "OUTDIR: ${outdir}"
    echo "BACKBONE: ${BACKBONE}"
    echo "ROOT: ${ROOT}"
    echo "EXEC_MODE: $(resolve_exec_mode "${method}")"
    echo "METHOD_CFG: $(resolve_method_cfg "${method}")"
    echo "RUNTIME_CFG: $(resolve_runtime_cfg "${method}")"
    echo "============================================================"
  } >> "${logfile}"
}

run_case_fs_like() {
  local gpu_id=$1
  local stage=$2
  local base_root=$3
  local method=$4
  local protocol=$5
  local dataset=$6
  local shot=$7
  local seed=$8
  local tag=$9
  local opts_str=${10}

  local phase subsample protocol_cfg
  read -r phase subsample protocol_cfg <<< "$(resolve_phase_semantics "${protocol}")"

  local method_cfg runtime_cfg launch_method launch_exec outdir logfile
  method_cfg="$(resolve_method_cfg "${method}")"
  runtime_cfg="$(resolve_runtime_cfg "${method}")"
  launch_method="$(resolve_launch_method "${method}")"
  launch_exec="$(resolve_exec_mode "${method}")"
  outdir="$(build_outdir "${base_root}" "${method}" "${protocol}" "${phase}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  logfile="${outdir}/run.log"

  mkdir -p "${outdir}"
  append_manifest "${stage}" "${method}" "${protocol}" "${phase}" "${subsample}" "${dataset}" "${shot}" "${seed}" "${tag}" "${outdir}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
    echo "[skip] ${stage} ${method} ${dataset} shot=${shot} seed=${seed} tag=${tag}"
    cleanup_checkpoint_if_ready "${outdir}" "${logfile}"
    return 0
  fi

  : > "${logfile}"
  write_log_header "${logfile}" "${gpu_id}" "${stage}" "${method}" "${protocol}" "${phase}" "${subsample}" "${dataset}" "${shot}" "${seed}" "${tag}" "${outdir}"

  local -a EXTRA_OPTS
  if [[ -n "${opts_str}" ]]; then
    read -r -a EXTRA_OPTS <<< "${opts_str}"
  else
    EXTRA_OPTS=()
  fi

  (
    cd "${PROJECT_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
      --root "${ROOT}" \
      --dataset-config-file "configs/datasets/${dataset}.yaml" \
      --method-config-file "${method_cfg}" \
      --protocol-config-file "${protocol_cfg}" \
      --runtime-config-file "${runtime_cfg}" \
      --output-dir "${outdir}" \
      --method "${launch_method}" \
      --protocol "${protocol}" \
      --exec-mode "${launch_exec}" \
      --seed "${seed}" \
      DATASET.NUM_SHOTS "${shot}" \
      DATASET.SUBSAMPLE_CLASSES "${subsample}" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      "${EXTRA_OPTS[@]}"
  ) >> "${logfile}" 2>&1

  cleanup_checkpoint_if_ready "${outdir}" "${logfile}"
}

run_case_b2n() {
  # Important:
  #   train_base checkpoint is required by test_new.
  #   Do not delete train_base/refactor_model until test_new/test_metrics.json exists.
  local gpu_id=$1
  local stage=$2
  local base_root=$3
  local method=$4
  local dataset=$5
  local shot=$6
  local seed=$7
  local tag=$8
  local opts_str=$9

  local method_cfg runtime_cfg launch_method launch_exec
  method_cfg="$(resolve_method_cfg "${method}")"
  runtime_cfg="$(resolve_runtime_cfg "${method}")"
  launch_method="$(resolve_launch_method "${method}")"
  launch_exec="$(resolve_exec_mode "${method}")"

  local train_outdir eval_outdir train_log eval_log
  train_outdir="$(build_b2n_train_outdir "${base_root}" "${method}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  eval_outdir="$(build_b2n_test_new_outdir "${base_root}" "${method}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  train_log="${train_outdir}/run.log"
  eval_log="${eval_outdir}/run.log"

  mkdir -p "${train_outdir}" "${eval_outdir}"

  append_manifest "${stage}_train_base" "${method}" B2N train_base base "${dataset}" "${shot}" "${seed}" "${tag}" "${train_outdir}"
  append_manifest "${stage}_test_new" "${method}" B2N test_new new "${dataset}" "${shot}" "${seed}" "${tag}" "${eval_outdir}"

  local -a EXTRA_OPTS
  if [[ -n "${opts_str}" ]]; then
    read -r -a EXTRA_OPTS <<< "${opts_str}"
  else
    EXTRA_OPTS=()
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    echo "[skip] ${stage} B2N ${method} ${dataset} shot=${shot} seed=${seed} tag=${tag}"
    cleanup_b2n_train_checkpoint_after_test_new "${train_outdir}" "${eval_outdir}" "${train_log}"
    return 0
  fi

  if [[ ! -f "${train_outdir}/test_metrics.json" || ! $(checkpoint_ready "${train_outdir}" && echo yes || true) ]]; then
    : > "${train_log}"
    write_log_header "${train_log}" "${gpu_id}" "${stage}_train_base" "${method}" B2N train_base base "${dataset}" "${shot}" "${seed}" "${tag}" "${train_outdir}"

    (
      cd "${PROJECT_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${ROOT}" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --method-config-file "${method_cfg}" \
        --protocol-config-file "configs/protocols/b2n.yaml" \
        --runtime-config-file "${runtime_cfg}" \
        --output-dir "${train_outdir}" \
        --method "${launch_method}" \
        --protocol B2N \
        --exec-mode "${launch_exec}" \
        --seed "${seed}" \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES base \
        MODEL.BACKBONE.NAME "${BACKBONE}" \
        "${EXTRA_OPTS[@]}"
    ) >> "${train_log}" 2>&1
  else
    touch "${train_log}"
    echo "[skip] existing train_base metrics and checkpoint: ${train_outdir}" >> "${train_log}"
  fi

  if ! checkpoint_ready "${train_outdir}"; then
    echo "[error] missing checkpoint for B2N test_new: ${train_outdir}/refactor_model" >> "${train_log}"
    return 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    echo "[skip] existing test_new metrics: ${eval_outdir}" >> "${eval_log}"
  else
    : > "${eval_log}"
    write_log_header "${eval_log}" "${gpu_id}" "${stage}_test_new" "${method}" B2N test_new new "${dataset}" "${shot}" "${seed}" "${tag}" "${eval_outdir}"
    echo "MODEL_DIR: ${train_outdir}" >> "${eval_log}"

    (
      cd "${PROJECT_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${ROOT}" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --method-config-file "${method_cfg}" \
        --protocol-config-file "configs/protocols/b2n_test_new.yaml" \
        --runtime-config-file "${runtime_cfg}" \
        --output-dir "${eval_outdir}" \
        --model-dir "${train_outdir}" \
        --method "${launch_method}" \
        --protocol B2N \
        --exec-mode "${launch_exec}" \
        --seed "${seed}" \
        --eval-only \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES new \
        MODEL.BACKBONE.NAME "${BACKBONE}" \
        "${EXTRA_OPTS[@]}"
    ) >> "${eval_log}" 2>&1
  fi

  cleanup_b2n_train_checkpoint_after_test_new "${train_outdir}" "${eval_outdir}" "${train_log}"
}

schedule_case() {
  local stage=$1
  local base_root=$2
  local method=$3
  local protocol=$4
  local dataset=$5
  local shot=$6
  local seed=$7
  local tag=$8
  local opts_str=$9

  wait_for_compatible_slot "${method}"
  local slot="${READY_SLOT}"
  local gpu_id="${READY_GPU}"
  local slot_weight="${READY_WEIGHT}"
  local phase subsample protocol_cfg outdir logfile
  read -r phase subsample protocol_cfg <<< "$(resolve_phase_semantics "${protocol}")"

  if [[ "${protocol}" == "B2N" ]]; then
    outdir="$(build_b2n_train_outdir "${base_root}" "${method}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  else
    outdir="$(build_outdir "${base_root}" "${method}" "${protocol}" "${phase}" "${dataset}" "${shot}" "${seed}" "${tag}")"
  fi
  logfile="${outdir}/run.log"

  local desc="${stage} method=${method} protocol=${protocol} dataset=${dataset} shot=${shot} seed=${seed} tag=${tag}"

  (
    if [[ "${protocol}" == "B2N" ]]; then
      run_case_b2n "${gpu_id}" "${stage}" "${base_root}" "${method}" "${dataset}" "${shot}" "${seed}" "${tag}" "${opts_str}"
    else
      run_case_fs_like "${gpu_id}" "${stage}" "${base_root}" "${method}" "${protocol}" "${dataset}" "${shot}" "${seed}" "${tag}" "${opts_str}"
    fi
  ) &

  RUNNING_PIDS[$slot]=$!
  SLOT_GPU[$slot]="${gpu_id}"
  SLOT_WEIGHT[$slot]="${slot_weight}"
  GPU_USED["${gpu_id}"]=$(( ${GPU_USED["${gpu_id}"]:-0} + slot_weight ))
  SLOT_DESC[$slot]="${desc}"
  SLOT_LOG[$slot]="${logfile}"

  echo "[launch] slot=${slot} gpu=${gpu_id} pid=${RUNNING_PIDS[$slot]} ${desc}"
  echo "         log=${logfile}"
}

# ------------------------------------------------------------
# Stages
# ------------------------------------------------------------
run_tune_stage() {
  echo
  echo "################################################################"
  echo "# Tune stage: FS 16-shot seed1"
  echo "################################################################"

  init_slots

  local dataset shot seed tag opts
  for dataset in ${TUNE_DATASETS}; do
    for shot in ${TUNE_SHOTS}; do
      for seed in ${TUNE_SEEDS}; do
        if [[ "${RUN_VCRM}" == "1" ]]; then
          for tag in "${VCRM_TAGS[@]}"; do
            opts="${VCRM_OPTS[$tag]}"
            schedule_case tune "${TUNE_ROOT}" VCRMMMRL FS "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          done
        fi

        if [[ "${RUN_BAYES}" == "1" ]]; then
          for tag in "${BAYES_TAGS[@]}"; do
            opts="${BAYES_OPTS[$tag]}"
            schedule_case tune "${TUNE_ROOT}" BayesMMRL FS "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          done
        fi
      done
    done
  done

  wait_all_jobs

  summarize_manifest "${MANIFEST}" "${TUNE_SUMMARY}"
  select_best_from_tune_summary "${TUNE_SUMMARY}" "${BEST_SUMMARY}" "${BEST_ENV}"
}

run_confirm_fs_stage() {
  echo
  echo "################################################################"
  echo "# Confirm FS stage"
  echo "################################################################"

  load_best_env
  init_slots

  local method dataset shot seed tag key opts

  for dataset in ${CONFIRM_DATASETS}; do
    for shot in ${CONFIRM_SHOTS}; do
      for seed in ${CONFIRM_SEEDS}; do

        if [[ "${RUN_VCRM}" == "1" ]]; then
          method=VCRMMMRL
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${VCRM_OPTS[$tag]}"
            schedule_case confirm_fs "${CONFIRM_ROOT}" "${method}" FS "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          else
            echo "[warn] no best tag for ${key}; skip confirm FS" >&2
          fi
        fi

        if [[ "${RUN_BAYES}" == "1" ]]; then
          method=BayesMMRL
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${BAYES_OPTS[$tag]}"
            schedule_case confirm_fs "${CONFIRM_ROOT}" "${method}" FS "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          else
            echo "[warn] no best tag for ${key}; skip confirm FS" >&2
          fi
        fi

        if [[ "${RUN_MMRL}" == "1" ]]; then
          tag="$(default_tag_for_method MMRL)"
          schedule_case confirm_fs "${CONFIRM_ROOT}" MMRL FS "${dataset}" "${shot}" "${seed}" "${tag}" ""
        fi

        if [[ "${RUN_BAYES_ADAPTER}" == "1" ]]; then
          tag="$(default_tag_for_method BayesAdapter)"
          schedule_case confirm_fs "${CONFIRM_ROOT}" BayesAdapter FS "${dataset}" "${shot}" "${seed}" "${tag}" ""
        fi

      done
    done
  done

  wait_all_jobs
  summarize_manifest "${MANIFEST}" "${CONFIRM_SUMMARY}"
}

run_confirm_b2n_stage() {
  echo
  echo "################################################################"
  echo "# Confirm B2N stage"
  echo "################################################################"

  load_best_env
  init_slots

  local method dataset shot seed tag key opts

  for dataset in ${CONFIRM_DATASETS}; do
    for shot in ${CONFIRM_SHOTS}; do
      for seed in ${CONFIRM_SEEDS}; do

        if [[ "${RUN_VCRM}" == "1" ]]; then
          method=VCRMMMRL
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${VCRM_OPTS[$tag]}"
            schedule_case confirm_b2n "${B2N_ROOT}" "${method}" B2N "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          else
            echo "[warn] no best tag for ${key}; skip B2N" >&2
          fi
        fi

        if [[ "${RUN_BAYES}" == "1" ]]; then
          method=BayesMMRL
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${BAYES_OPTS[$tag]}"
            schedule_case confirm_b2n "${B2N_ROOT}" "${method}" B2N "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
          else
            echo "[warn] no best tag for ${key}; skip B2N" >&2
          fi
        fi

        if [[ "${RUN_MMRL}" == "1" ]]; then
          tag="$(default_tag_for_method MMRL)"
          schedule_case confirm_b2n "${B2N_ROOT}" MMRL B2N "${dataset}" "${shot}" "${seed}" "${tag}" ""
        fi

        if [[ "${RUN_BAYES_ADAPTER}" == "1" ]]; then
          tag="$(default_tag_for_method BayesAdapter)"
          schedule_case confirm_b2n "${B2N_ROOT}" BayesAdapter B2N "${dataset}" "${shot}" "${seed}" "${tag}" ""
        fi

      done
    done
  done

  wait_all_jobs
  summarize_manifest "${MANIFEST}" "${B2N_SUMMARY}"
}

print_config() {
  echo
  echo "################################################################"
  echo "# Sweep configuration"
  echo "################################################################"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "ROOT=${ROOT}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "BACKBONE=${BACKBONE}"
  echo "EXEC_MODE=${EXEC_MODE}"
  echo "BAYES_ADAPTER_EXEC_MODE=${BAYES_ADAPTER_EXEC_MODE}"
  echo "DATASETS=${DATASETS}"
  echo "TUNE_DATASETS=${TUNE_DATASETS}"
  echo "TUNE_SHOTS=${TUNE_SHOTS}"
  echo "TUNE_SEEDS=${TUNE_SEEDS}"
  echo "CONFIRM_DATASETS=${CONFIRM_DATASETS}"
  echo "CONFIRM_SHOTS=${CONFIRM_SHOTS}"
  echo "CONFIRM_SEEDS=${CONFIRM_SEEDS}"
  echo "VCRM_ETA_LIST=${VCRM_ETA_LIST}"
  echo "VCRM_MOD_WEIGHT_LIST=${VCRM_MOD_WEIGHT_LIST}"
  echo "VCRM_BATCH_SIZE=${VCRM_BATCH_SIZE}"
  echo "BAYES_REP_PRIOR_STD_LIST=${BAYES_REP_PRIOR_STD_LIST}"
  echo "BAYES_REP_KL_WEIGHT_LIST=${BAYES_REP_KL_WEIGHT_LIST}"
  echo "BAYES_MAIN_CONSISTENCY_MODE_LIST=${BAYES_MAIN_CONSISTENCY_MODE_LIST}"
  echo "BAYES_MAIN_CONSISTENCY_WEIGHT_LIST=${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST:-}"
  echo "JOBS_PER_GPU=${JOBS_PER_GPU}"
  echo "VCRM_EFFECTIVE_JOBS_PER_GPU=1"
  echo "OTHER_METHODS_EFFECTIVE_JOBS_PER_GPU=${JOBS_PER_GPU}"
  echo "NON_SWEEP_SETTINGS=YAML_DEFAULTS"
  echo "AUTO_TUNE=${AUTO_TUNE}"
  echo "AUTO_CONFIRM_FS=${AUTO_CONFIRM_FS}"
  echo "AUTO_CONFIRM_B2N=${AUTO_CONFIRM_B2N}"
  echo "RUN_VCRM=${RUN_VCRM}"
  echo "VCRM_RUN_ORDER=AFTER_ALL_OTHER_ENABLED_EXPERIMENTS"
  echo "RUN_BAYES=${RUN_BAYES}"
  echo "RUN_MMRL=${RUN_MMRL}"
  echo "RUN_BAYES_ADAPTER=${RUN_BAYES_ADAPTER}"
  echo "MANIFEST=${MANIFEST}"
}

main() {
  trap 'echo "[interrupt] stopping children"; cleanup_children; exit 130' INT TERM

  mkdir -p "${OUTPUT_ROOT}"
  init_gpu_list
  init_search_spaces
  print_config

  if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
    summarize_manifest "${MANIFEST}" "${TUNE_SUMMARY}"
    if [[ -f "${TUNE_SUMMARY}" ]]; then
      select_best_from_tune_summary "${TUNE_SUMMARY}" "${BEST_SUMMARY}" "${BEST_ENV}" || true
    fi
    summarize_manifest "${MANIFEST}" "${CONFIRM_SUMMARY}"
    summarize_manifest "${MANIFEST}" "${B2N_SUMMARY}"
    exit 0
  fi

  # ----------------------------------------------------------
  # Requirement:
  #   VCRMMMRL experiments must run after all other experiments.
  #
  # Implementation:
  #   Phase A runs every enabled non-VCRMMMRL method.
  #   Phase B runs VCRMMMRL only.
  #
  # This means VCRMMMRL tune is also deferred. It is not required for
  # BayesMMRL/MMRL/BayesAdapter confirms, because those methods either use
  # their own Bayes best config or YAML defaults.
  # ----------------------------------------------------------
  local original_run_vcrm="${RUN_VCRM}"
  local original_run_bayes="${RUN_BAYES}"
  local original_run_mmrl="${RUN_MMRL}"
  local original_run_bayes_adapter="${RUN_BAYES_ADAPTER}"

  if [[ "${original_run_vcrm}" == "1" ]]; then
    echo
    echo "################################################################"
    echo "# Phase A: run all non-VCRMMMRL experiments first"
    echo "################################################################"

    RUN_VCRM=0
    RUN_BAYES="${original_run_bayes}"
    RUN_MMRL="${original_run_mmrl}"
    RUN_BAYES_ADAPTER="${original_run_bayes_adapter}"

    if [[ "${AUTO_TUNE}" == "1" ]]; then
      run_tune_stage
    else
      echo "[info] AUTO_TUNE=0; using existing BEST_ENV=${BEST_ENV}"
    fi

    if [[ "${AUTO_CONFIRM_FS}" == "1" ]]; then
      run_confirm_fs_stage
    fi

    if [[ "${AUTO_CONFIRM_B2N}" == "1" ]]; then
      run_confirm_b2n_stage
    fi

    echo
    echo "################################################################"
    echo "# Phase B: run VCRMMMRL after all other experiments"
    echo "################################################################"

    RUN_VCRM=1
    RUN_BAYES=0
    RUN_MMRL=0
    RUN_BAYES_ADAPTER=0

    if [[ "${AUTO_TUNE}" == "1" ]]; then
      run_tune_stage
    else
      echo "[info] AUTO_TUNE=0; using existing BEST_ENV=${BEST_ENV}"
    fi

    if [[ "${AUTO_CONFIRM_FS}" == "1" ]]; then
      run_confirm_fs_stage
    fi

    if [[ "${AUTO_CONFIRM_B2N}" == "1" ]]; then
      run_confirm_b2n_stage
    fi

    RUN_VCRM="${original_run_vcrm}"
    RUN_BAYES="${original_run_bayes}"
    RUN_MMRL="${original_run_mmrl}"
    RUN_BAYES_ADAPTER="${original_run_bayes_adapter}"
  else
    if [[ "${AUTO_TUNE}" == "1" ]]; then
      run_tune_stage
    else
      echo "[info] AUTO_TUNE=0; using existing BEST_ENV=${BEST_ENV}"
    fi

    if [[ "${AUTO_CONFIRM_FS}" == "1" ]]; then
      run_confirm_fs_stage
    fi

    if [[ "${AUTO_CONFIRM_B2N}" == "1" ]]; then
      run_confirm_b2n_stage
    fi
  fi

  summarize_manifest "${MANIFEST}" "${OUTPUT_ROOT}/all_summary.csv"

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)."
    exit 1
  fi

  echo "[DONE] all jobs finished successfully."
  echo "[outputs]"
  echo "  manifest: ${MANIFEST}"
  echo "  tune summary: ${TUNE_SUMMARY}"
  echo "  best configs: ${BEST_SUMMARY}"
  echo "  confirm summary: ${CONFIRM_SUMMARY}"
  echo "  b2n summary: ${B2N_SUMMARY}"
  echo "  all summary: ${OUTPUT_ROOT}/all_summary.csv"
}

main "$@"
