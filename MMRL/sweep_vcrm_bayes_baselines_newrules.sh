#!/bin/bash
set -euo pipefail

# ============================================================
# Joint sweep + confirm script with new rules + MNDL method:
#
# Methods:
#   1) VCRMMMRL:
#      - Tune VCRM_ETA and VCRM_MOD_WEIGHT on FS 16-shot seed1.
#      - Force train batch size 24 by default. Override with VCRM_BATCH_SIZE=...
#   2) BayesMMRL:
#      - Tune REP_PRIOR_STD, REP_KL_WEIGHT, MAIN_CONSISTENCY_MODE,
#        MAIN_CONSISTENCY_WEIGHT on FS 16-shot seed1.
#      - Non-swept BayesMMRL settings come from YAML defaults.
#   3) matrix_normal_diag_lowrank:
#      - Exposed as a separate experiment method in manifest/summary/best files.
#      - Launched through BayesMMRL trainer/config.
#      - Adds BAYES_MMRL.REP_SIGMA_MODE matrix_normal_diag_lowrank
#        and tunes the same BayesMMRL hyperparameters plus low-rank rank.
#   4) MMRL and BayesAdapter:
#      - Confirm-only baselines.
#
# Rules:
#   - B2N confirm defaults to 16-shot only: B2N_SHOTS="16".
#   - Best config selection for BayesMMRL, matrix_normal_diag_lowrank,
#     and VCRMMMRL:
#       per method+dataset, find best ACC, allow ACC_DROP=0.2 percentage
#       points, then choose the lowest ECE within that ACC window.
#   - VCRMMMRL is deferred until all other enabled experiments finish.
#   - RESET_MANIFEST=1 rebuilds manifest/summary/best files only.
#     It does not delete existing experiment outputs or test_metrics.json.
#     With SKIP_EXISTING=1, finished jobs are skipped and re-registered.
#
# Example:
#   bash sweep_vcrm_bayes_baselines_mndl.sh \
#     PROJECT_DIR="$PWD" \
#     OUTPUT_ROOT=output_sweeps/vcrm_bayes_joint \
#     RESET_MANIFEST=1 SKIP_EXISTING=1 \
#     RUN_MNDL=1 MNDL_LOWRANK_RANK_LIST="4" \
#     GPU_IDS="0 1 2 3 4 5" JOBS_PER_GPU=2
# ============================================================

apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      case "${key}" in
        PROJECT_DIR|ROOT|DATA_ROOT|OUTPUT_ROOT|BACKBONE|EXEC_MODE|BAYES_ADAPTER_EXEC_MODE|\
        DATASETS|TUNE_DATASETS|TUNE_SHOTS|TUNE_SEEDS|CONFIRM_DATASETS|CONFIRM_SHOTS|B2N_SHOTS|CONFIRM_SEEDS|\
        GPU_IDS|NGPU|JOBS_PER_GPU|SKIP_EXISTING|SLEEP_SEC|DELETE_CKPT_AFTER_TEST|RESET_MANIFEST|\
        AUTO_TUNE|AUTO_CONFIRM_FS|AUTO_CONFIRM_B2N|RUN_VCRM|RUN_BAYES|RUN_MNDL|RUN_MMRL|RUN_BAYES_ADAPTER|\
        VCRM_ETA_LIST|VCRM_MOD_WEIGHT_LIST|VCRM_BATCH_SIZE|\
        BAYES_REP_PRIOR_STD_LIST|BAYES_REP_KL_WEIGHT_LIST|BAYES_MAIN_CONSISTENCY_MODE_LIST|BAYES_MAIN_CONSISTENCY_WEIGHT_LIST|\
        MNDL_REP_PRIOR_STD_LIST|MNDL_REP_KL_WEIGHT_LIST|MNDL_MAIN_CONSISTENCY_MODE_LIST|MNDL_MAIN_CONSISTENCY_WEIGHT_LIST|MNDL_LOWRANK_RANK_LIST|\
        ACC_DROP|SUMMARY_ONLY)
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

# Apply KEY=VALUE overrides before defaults so derived defaults such as
# TUNE_DATASETS=${DATASETS} and ROOT=${DATA_ROOT} honor user overrides.
apply_kv_args "$@"

# Defaults
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
ROOT=${ROOT:-${DATA_ROOT:-DATASETS}}
DATA_ROOT=${DATA_ROOT:-${ROOT}}
BACKBONE=${BACKBONE:-ViT-B/16}
EXEC_MODE=${EXEC_MODE:-online}
BAYES_ADAPTER_EXEC_MODE=${BAYES_ADAPTER_EXEC_MODE:-cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/vcrm_bayes_joint}

DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft oxford_pets stanford_cars ucf101"}

TUNE_DATASETS=${TUNE_DATASETS:-${DATASETS}}
TUNE_SHOTS=${TUNE_SHOTS:-"16"}
TUNE_SEEDS=${TUNE_SEEDS:-"1"}

CONFIRM_DATASETS=${CONFIRM_DATASETS:-${DATASETS}}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-"1 2 4 8 16 32"}
B2N_SHOTS=${B2N_SHOTS:-"16"}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-"1 2 3"}

GPU_IDS=${GPU_IDS:-}
NGPU=${NGPU:-}
JOBS_PER_GPU=${JOBS_PER_GPU:-2}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}
RESET_MANIFEST=${RESET_MANIFEST:-0}

AUTO_TUNE=${AUTO_TUNE:-1}
AUTO_CONFIRM_FS=${AUTO_CONFIRM_FS:-1}
AUTO_CONFIRM_B2N=${AUTO_CONFIRM_B2N:-1}

RUN_VCRM=${RUN_VCRM:-1}
RUN_BAYES=${RUN_BAYES:-1}
RUN_MNDL=${RUN_MNDL:-1}
RUN_MMRL=${RUN_MMRL:-1}
RUN_BAYES_ADAPTER=${RUN_BAYES_ADAPTER:-1}

VCRM_ETA_LIST=${VCRM_ETA_LIST:-"0.05 0.1 0.2"}
VCRM_MOD_WEIGHT_LIST=${VCRM_MOD_WEIGHT_LIST:-"0.0 1e-4 1e-3"}
VCRM_BATCH_SIZE=${VCRM_BATCH_SIZE:-24}

BAYES_REP_PRIOR_STD_LIST=${BAYES_REP_PRIOR_STD_LIST:-"0.02 0.1 1.0"}
BAYES_REP_KL_WEIGHT_LIST=${BAYES_REP_KL_WEIGHT_LIST:-" 1e-2 5e-2 1e-1"}
BAYES_MAIN_CONSISTENCY_MODE_LIST=${BAYES_MAIN_CONSISTENCY_MODE_LIST:-" logit"}
BAYES_MAIN_CONSISTENCY_WEIGHT_LIST=${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST:-"0.0 0.01 0.03"}

# matrix_normal_diag_lowrank uses BayesMMRL trainer but is tracked as a separate method.
MNDL_REP_PRIOR_STD_LIST=${MNDL_REP_PRIOR_STD_LIST:-${BAYES_REP_PRIOR_STD_LIST}}
MNDL_REP_KL_WEIGHT_LIST=${MNDL_REP_KL_WEIGHT_LIST:-${BAYES_REP_KL_WEIGHT_LIST}}
MNDL_MAIN_CONSISTENCY_MODE_LIST=${MNDL_MAIN_CONSISTENCY_MODE_LIST:-${BAYES_MAIN_CONSISTENCY_MODE_LIST}}
MNDL_MAIN_CONSISTENCY_WEIGHT_LIST=${MNDL_MAIN_CONSISTENCY_WEIGHT_LIST:-${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST}}
MNDL_LOWRANK_RANK_LIST=${MNDL_LOWRANK_RANK_LIST:-"4"}

# Accuracy is in percentage points. ACC_DROP=0.2 means allow 96.91 -> 96.71.
ACC_DROP=${ACC_DROP:-0.2}

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
READY_GPU=""
READY_WEIGHT=""

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
declare -ga MNDL_TAGS
declare -gA MNDL_OPTS
declare -gA BEST_TAG_BY_METHOD_DATASET

method_key() {
  local method=$1
  echo "$method" | tr '[:upper:]' '[:lower:]' | sed 's/-/_/g'
}

is_clip_adapter_alias() {
  local method=$1 key
  key="$(method_key "${method}")"
  [[ "${key}" == "bayesadapter" || "${key}" == "bayes_adapter" ]]
}

is_mndl_alias() {
  local method=$1 key
  key="$(method_key "${method}")"
  [[ "${key}" == "matrix_normal_diag_lowrank" || "${key}" == "matrixnormaldiaglowrank" || "${key}" == "mndl" ]]
}

resolve_launch_method() {
  local method=$1
  if is_clip_adapter_alias "${method}"; then
    echo "ClipAdapters"
  elif is_mndl_alias "${method}"; then
    echo "BayesMMRL"
  else
    echo "${method}"
  fi
}

resolve_method_cfg() {
  local method=$1
  case "${method}" in
    VCRMMMRL) echo "configs/methods/vcrm_mmrl.yaml" ;;
    BayesMMRL) echo "configs/methods/bayesmmrl.yaml" ;;
    matrix_normal_diag_lowrank|MatrixNormalDiagLowrank|MNDL|mndl) echo "configs/methods/bayesmmrl.yaml" ;;
    MMRL) echo "configs/methods/mmrl.yaml" ;;
    BayesAdapter|bayesadapter|bayes_adapter) echo "configs/methods/clip_adapters_bayes.yaml" ;;
    *) echo "Unsupported method: ${method}" >&2; exit 1 ;;
  esac
}

resolve_runtime_cfg() {
  local method=$1
  case "${method}" in
    VCRMMMRL|BayesMMRL|MMRL|matrix_normal_diag_lowrank|MatrixNormalDiagLowrank|MNDL|mndl) echo "configs/runtime/mmrl_family.yaml" ;;
    BayesAdapter|bayesadapter|bayes_adapter) echo "configs/runtime/adapter_family.yaml" ;;
    *) echo "configs/runtime/default.yaml" ;;
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
    *) echo "Unknown protocol: ${protocol}" >&2; exit 1 ;;
  esac
}

resolve_run_tag_from_cfg() {
  local method=$1 method_cfg=$2
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

build_case_root() {
  local base_root=$1 method=$2 protocol=$3 phase=$4 dataset=$5 shot=$6 tag=$7
  local launch_method
  launch_method="$(resolve_launch_method "${method}")"
  if [[ "${launch_method}" == "ClipAdapters" ]]; then
    echo "${base_root}/${launch_method}/${tag}/${protocol}/${phase}/${dataset}/shots_${shot}/$(backbone_tag)"
  else
    echo "${base_root}/${launch_method}/${protocol}/${phase}/${dataset}/shots_${shot}/$(backbone_tag)/${tag}"
  fi
}

build_outdir() {
  local base_root=$1 method=$2 protocol=$3 phase=$4 dataset=$5 shot=$6 seed=$7 tag=$8
  echo "$(build_case_root "${base_root}" "${method}" "${protocol}" "${phase}" "${dataset}" "${shot}" "${tag}")/seed${seed}"
}

build_b2n_train_outdir() {
  local base_root=$1 method=$2 dataset=$3 shot=$4 seed=$5 tag=$6
  echo "$(build_outdir "${base_root}" "${method}" B2N train_base "${dataset}" "${shot}" "${seed}" "${tag}")"
}

build_b2n_test_new_outdir() {
  local base_root=$1 method=$2 dataset=$3 shot=$4 seed=$5 tag=$6
  local launch_method
  launch_method="$(resolve_launch_method "${method}")"
  if [[ "${launch_method}" == "ClipAdapters" ]]; then
    echo "${base_root}/${launch_method}/${tag}/B2N/test_new/${dataset}/shots_${shot}/$(backbone_tag)/seed${seed}"
  else
    echo "${base_root}/${launch_method}/B2N/test_new/${dataset}/shots_${shot}/$(backbone_tag)/${tag}/seed${seed}"
  fi
}

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
    echo "${JOBS_PER_GPU}"
  else
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
      if wait "${pid}"; then rc=0; else rc=$?; fi

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

wait_for_compatible_slot() {
  local method=$1 gpu used slot
  READY_WEIGHT="$(job_slot_weight "${method}")"
  READY_SLOT=""
  READY_GPU=""

  while true; do
    _reap_finished_jobs
    for gpu in "${PHYSICAL_GPUS[@]}"; do
      used="${GPU_USED["${gpu}"]:-0}"
      if (( used + READY_WEIGHT <= JOBS_PER_GPU )); then
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
      if wait "${pid}"; then rc=0; else rc=$?; fi

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

ensure_manifest_header() {
  mkdir -p "$(dirname "${MANIFEST}")"
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "stage,method,protocol,phase,subsample,dataset,shot,seed,tag,outdir" > "${MANIFEST}"
  fi
}

reset_index_files_if_requested() {
  if [[ "${RESET_MANIFEST}" != "1" || "${SUMMARY_ONLY:-0}" == "1" ]]; then
    return 0
  fi
  echo "[reset] rebuilding manifest/summary/best files only; experiment outputs are preserved"
  rm -f "${MANIFEST}" "${TUNE_SUMMARY}" "${BEST_SUMMARY}" "${BEST_ENV}" \
        "${CONFIRM_SUMMARY}" "${B2N_SUMMARY}" "${OUTPUT_ROOT}/all_summary.csv"
}

append_manifest() {
  local stage=$1 method=$2 protocol=$3 phase=$4 subsample=$5 dataset=$6 shot=$7 seed=$8 tag=$9 outdir=${10}
  ensure_manifest_header
  echo "${stage},${method},${protocol},${phase},${subsample},${dataset},${shot},${seed},${tag},${outdir}" >> "${MANIFEST}"
}

summarize_manifest() {
  local manifest=$1 summary_csv=$2

  python - <<PY
import csv
import json
from pathlib import Path

manifest = Path(r"${manifest}")
out = Path(r"${summary_csv}")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
seen = set()
if manifest.exists():
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("stage"), row.get("method"), row.get("protocol"),
                row.get("phase"), row.get("dataset"), row.get("shot"),
                row.get("seed"), row.get("tag"), row.get("outdir"),
            )
            if key in seen:
                continue
            seen.add(key)

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
    "stage", "method", "protocol", "phase", "subsample", "dataset", "shot",
    "seed", "tag", "outdir", "status", "num_samples", "accuracy", "error",
    "macro_f1", "ece", "nll", "brier", "metrics_path",
]
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[summary] wrote {out} rows={len(rows)}")
PY
}

select_best_from_tune_summary() {
  local tune_summary=$1 best_summary=$2 best_env=$3

  python - <<PY
import csv
import math
from pathlib import Path
from collections import defaultdict

summary = Path(r"${tune_summary}")
best_csv = Path(r"${best_summary}")
best_env = Path(r"${best_env}")
ACC_DROP = float("${ACC_DROP}")

rows = []
if summary.exists():
    with summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage") != "tune":
                continue
            if row.get("protocol") != "FS":
                continue
            if row.get("shot") != "16" or row.get("seed") != "1":
                continue
            if row.get("method") not in {"VCRMMMRL", "BayesMMRL", "matrix_normal_diag_lowrank"}:
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
    best_acc = max(r["_acc"] for r in group)
    acc_threshold = best_acc - ACC_DROP
    candidates = [r for r in group if r["_acc"] >= acc_threshold]
    candidates = sorted(
        candidates,
        key=lambda r: (
            r["_ece"] if math.isfinite(r["_ece"]) else float("inf"),
            -r["_acc"],
            r["tag"],
        ),
    )
    chosen = candidates[0]
    chosen["_best_acc"] = best_acc
    chosen["_acc_threshold"] = acc_threshold
    chosen["_num_candidates"] = len(candidates)
    chosen["_num_finished"] = len(group)
    best_rows.append(chosen)

best_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "method", "dataset", "tag", "accuracy", "ece", "nll", "brier",
    "best_acc", "acc_threshold", "acc_drop", "num_candidates",
    "num_finished", "outdir", "metrics_path",
]
with best_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in best_rows:
        writer.writerow({
            "method": r.get("method", ""),
            "dataset": r.get("dataset", ""),
            "tag": r.get("tag", ""),
            "accuracy": r.get("accuracy", ""),
            "ece": r.get("ece", ""),
            "nll": r.get("nll", ""),
            "brier": r.get("brier", ""),
            "best_acc": r.get("_best_acc", ""),
            "acc_threshold": r.get("_acc_threshold", ""),
            "acc_drop": ACC_DROP,
            "num_candidates": r.get("_num_candidates", ""),
            "num_finished": r.get("_num_finished", ""),
            "outdir": r.get("outdir", ""),
            "metrics_path": r.get("metrics_path", ""),
        })

def shq(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

lines = ["declare -gA BEST_TAG_BY_METHOD_DATASET"]
for r in best_rows:
    key = f"{r['method']}|{r['dataset']}"
    lines.append(f"BEST_TAG_BY_METHOD_DATASET[{shq(key)}]={shq(r['tag'])}")
lines.append("")
best_env.write_text("\n".join(lines), encoding="utf-8")

print(f"[best] selection rule: ACC >= best_acc - {ACC_DROP}, then lowest ECE")
print(f"[best] wrote {best_csv}")
print(f"[best] wrote {best_env}")
for r in best_rows:
    print(
        f"[best] method={r['method']} dataset={r['dataset']} tag={r['tag']} "
        f"acc={r.get('accuracy')} ece={r.get('ece')} best_acc={r.get('_best_acc')} "
        f"threshold={r.get('_acc_threshold')} candidates={r.get('_num_candidates')}/{r.get('_num_finished')}"
    )
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

maybe_load_best_env() {
  if [[ "${RUN_VCRM}" == "1" || "${RUN_BAYES}" == "1" || "${RUN_MNDL}" == "1" ]]; then
    load_best_env
  fi
}

register_case() {
  local array_name=$1 assoc_name=$2 raw_tag=$3
  shift 3
  local opts_str="$*" tag
  tag="$(sanitize "${raw_tag}")"
  eval "${array_name}+=(\"\${tag}\")"
  eval "${assoc_name}[\"\${tag}\"]=\"\${opts_str}\""
}

init_search_spaces() {
  VCRM_TAGS=()
  BAYES_TAGS=()
  MNDL_TAGS=()
  VCRM_OPTS=()
  BAYES_OPTS=()
  MNDL_OPTS=()

  local eta mod prior_std kl mode cons_weight rank

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

  for prior_std in ${MNDL_REP_PRIOR_STD_LIST}; do
    for kl in ${MNDL_REP_KL_WEIGHT_LIST}; do
      for mode in ${MNDL_MAIN_CONSISTENCY_MODE_LIST}; do
        for cons_weight in ${MNDL_MAIN_CONSISTENCY_WEIGHT_LIST}; do
          for rank in ${MNDL_LOWRANK_RANK_LIST}; do
            register_case MNDL_TAGS MNDL_OPTS \
              "mndl_pstd-${prior_std}_kl-${kl}_cons-${mode}_consw-${cons_weight}_rank-${rank}" \
              BAYES_MMRL.REP_SIGMA_MODE "matrix_normal_diag_lowrank" \
              BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
              BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
              BAYES_MMRL.MAIN_CONSISTENCY_MODE "${mode}" \
              BAYES_MMRL.MAIN_CONSISTENCY_WEIGHT "${cons_weight}" \
              BAYES_MMRL.REP_MN_LOWRANK_RANK "${rank}"
          done
        done
      done
    done
  done

  echo "[search] VCRMMMRL cases: ${#VCRM_TAGS[@]}"
  echo "[search] BayesMMRL cases: ${#BAYES_TAGS[@]}"
  echo "[search] matrix_normal_diag_lowrank cases: ${#MNDL_TAGS[@]}"
  echo "[search] non-swept settings use YAML defaults"
}

default_tag_for_method() {
  local method=$1 cfg
  cfg="$(resolve_method_cfg "${method}")"
  resolve_run_tag_from_cfg "${method}" "${cfg}"
}

checkpoint_ready() {
  local outdir=$1 ckpt_dir="${outdir}/refactor_model"

  [[ -d "${ckpt_dir}" ]] || return 1

  # Match latest RefactorRunner.load_model():
  #   1) prefer model-best.pth.tar
  #   2) otherwise load any model.pth*
  #
  # The latest RefactorRunner.save_model() does NOT write
  # refactor_model/checkpoint, so do not require it here.
  if [[ -f "${ckpt_dir}/model-best.pth.tar" ]]; then
    return 0
  fi

  compgen -G "${ckpt_dir}/model.pth*" >/dev/null
}

b2n_requires_refactor_checkpoint() {
  local method=$1
  # BayesAdapter is launched through ClipAdapters and may not write
  # refactor_model/checkpoint + model*.pth.tar*.
  # Do not block B2N test_new here; let run.py decide whether --model-dir is usable.
  if is_clip_adapter_alias "${method}"; then
    return 1
  fi
  return 0
}



cleanup_checkpoint_if_ready() {
  local outdir=$1 logfile=${2:-}
  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${outdir}/test_metrics.json" && -d "${outdir}/refactor_model" ]]; then
    rm -rf "${outdir}/refactor_model"
    [[ -n "${logfile}" ]] && echo "[cleanup] removed ${outdir}/refactor_model" >> "${logfile}"
  fi
}

cleanup_broken_resume_if_needed() {
  local outdir=$1 logfile=${2:-} ckpt_dir="${outdir}/refactor_model" broken=0

  if [[ ! -d "${ckpt_dir}" ]]; then
    return 0
  fi

  # Match latest RefactorRunner.load_model().
  # A valid lightweight checkpoint is either:
  #   refactor_model/model-best.pth.tar
  # or:
  #   refactor_model/model.pth*
  #
  # Do NOT require refactor_model/checkpoint because latest RefactorRunner
  # does not create it.
  if [[ ! -f "${ckpt_dir}/model-best.pth.tar" ]] && ! compgen -G "${ckpt_dir}/model.pth*" >/dev/null; then
    broken=1
  fi

  if [[ "${broken}" == "1" ]]; then
    rm -rf "${ckpt_dir}"
    [[ -n "${logfile}" ]] && echo "[cleanup] removed broken resume state: ${ckpt_dir}" >> "${logfile}"
  fi
}

cleanup_b2n_train_checkpoint_after_test_new() {
  local train_outdir=$1 eval_outdir=$2 logfile=${3:-}
  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi

  if [[ ! -f "${eval_outdir}/test_metrics.json" ]]; then
    [[ -n "${logfile}" ]] && echo "[cleanup] keep ${train_outdir}/refactor_model because test_new metrics are missing" >> "${logfile}"
    return 0
  fi

  if [[ -d "${train_outdir}/refactor_model" ]]; then
    rm -rf "${train_outdir}/refactor_model"
    [[ -n "${logfile}" ]] && echo "[cleanup] removed ${train_outdir}/refactor_model after test_new completed" >> "${logfile}"
  fi
}

write_log_header() {
  local logfile=$1 gpu_id=$2 stage=$3 method=$4 protocol=$5 phase=$6 subsample=$7 dataset=$8 shot=$9 seed=${10} tag=${11} outdir=${12}
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
  local gpu_id=$1 stage=$2 base_root=$3 method=$4 protocol=$5 dataset=$6 shot=$7 seed=$8 tag=$9 opts_str=${10}
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
  cleanup_broken_resume_if_needed "${outdir}" "${logfile}"

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
  local gpu_id=$1 stage=$2 base_root=$3 method=$4 dataset=$5 shot=$6 seed=$7 tag=$8 opts_str=$9

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

  if [[ ! -f "${train_outdir}/test_metrics.json" ]] || ! checkpoint_ready "${train_outdir}"; then
    : > "${train_log}"
    write_log_header "${train_log}" "${gpu_id}" "${stage}_train_base" "${method}" B2N train_base base "${dataset}" "${shot}" "${seed}" "${tag}" "${train_outdir}"
    cleanup_broken_resume_if_needed "${train_outdir}" "${train_log}"
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


  if b2n_requires_refactor_checkpoint "${method}"; then
    if ! checkpoint_ready "${train_outdir}"; then
      echo "[error] no loadable model file for B2N test_new: ${train_outdir}/refactor_model" >> "${train_log}"
      echo "[error] expected model-best.pth.tar or model.pth*" >> "${train_log}"
      return 1
    fi
  else
      if ! checkpoint_ready "${train_outdir}"; then
        echo "[info] ${method}: no refactor_model checkpoint found; continue to B2N test_new without pre-check" >> "${train_log}"
      fi
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
  local stage=$1 base_root=$2 method=$3 protocol=$4 dataset=$5 shot=$6 seed=$7 tag=$8 opts_str=$9

  wait_for_compatible_slot "${method}"
  local slot="${READY_SLOT}" gpu_id="${READY_GPU}" slot_weight="${READY_WEIGHT}"
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

        if [[ "${RUN_MNDL}" == "1" ]]; then
          for tag in "${MNDL_TAGS[@]}"; do
            opts="${MNDL_OPTS[$tag]}"
            schedule_case tune "${TUNE_ROOT}" matrix_normal_diag_lowrank FS "${dataset}" "${shot}" "${seed}" "${tag}" "${opts}"
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

  maybe_load_best_env
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

        if [[ "${RUN_MNDL}" == "1" ]]; then
          method=matrix_normal_diag_lowrank
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${MNDL_OPTS[$tag]}"
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
  echo "# Confirm B2N stage: B2N_SHOTS=${B2N_SHOTS}"
  echo "################################################################"

  maybe_load_best_env
  init_slots

  local method dataset shot seed tag key opts
  for dataset in ${CONFIRM_DATASETS}; do
    for shot in ${B2N_SHOTS}; do
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

        if [[ "${RUN_MNDL}" == "1" ]]; then
          method=matrix_normal_diag_lowrank
          key="${method}|${dataset}"
          tag="${BEST_TAG_BY_METHOD_DATASET[$key]:-}"
          if [[ -n "${tag}" ]]; then
            opts="${MNDL_OPTS[$tag]}"
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
  echo "B2N_SHOTS=${B2N_SHOTS}"
  echo "CONFIRM_SEEDS=${CONFIRM_SEEDS}"
  echo "VCRM_ETA_LIST=${VCRM_ETA_LIST}"
  echo "VCRM_MOD_WEIGHT_LIST=${VCRM_MOD_WEIGHT_LIST}"
  echo "VCRM_BATCH_SIZE=${VCRM_BATCH_SIZE}"
  echo "BAYES_REP_PRIOR_STD_LIST=${BAYES_REP_PRIOR_STD_LIST}"
  echo "BAYES_REP_KL_WEIGHT_LIST=${BAYES_REP_KL_WEIGHT_LIST}"
  echo "BAYES_MAIN_CONSISTENCY_MODE_LIST=${BAYES_MAIN_CONSISTENCY_MODE_LIST}"
  echo "BAYES_MAIN_CONSISTENCY_WEIGHT_LIST=${BAYES_MAIN_CONSISTENCY_WEIGHT_LIST}"
  echo "MNDL_REP_PRIOR_STD_LIST=${MNDL_REP_PRIOR_STD_LIST}"
  echo "MNDL_REP_KL_WEIGHT_LIST=${MNDL_REP_KL_WEIGHT_LIST}"
  echo "MNDL_MAIN_CONSISTENCY_MODE_LIST=${MNDL_MAIN_CONSISTENCY_MODE_LIST}"
  echo "MNDL_MAIN_CONSISTENCY_WEIGHT_LIST=${MNDL_MAIN_CONSISTENCY_WEIGHT_LIST}"
  echo "MNDL_LOWRANK_RANK_LIST=${MNDL_LOWRANK_RANK_LIST}"
  echo "ACC_DROP=${ACC_DROP}"
  echo "BEST_SELECTION=ACC >= best_acc - ACC_DROP, then lowest ECE"
  echo "JOBS_PER_GPU=${JOBS_PER_GPU}"
  echo "VCRM_EFFECTIVE_JOBS_PER_GPU=1"
  echo "OTHER_METHODS_EFFECTIVE_JOBS_PER_GPU=${JOBS_PER_GPU}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"
  echo "RESET_MANIFEST=${RESET_MANIFEST}"
  echo "DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST}"
  echo "AUTO_TUNE=${AUTO_TUNE}"
  echo "AUTO_CONFIRM_FS=${AUTO_CONFIRM_FS}"
  echo "AUTO_CONFIRM_B2N=${AUTO_CONFIRM_B2N}"
  echo "RUN_VCRM=${RUN_VCRM}"
  echo "VCRM_RUN_ORDER=AFTER_ALL_OTHER_ENABLED_EXPERIMENTS"
  echo "RUN_BAYES=${RUN_BAYES}"
  echo "RUN_MNDL=${RUN_MNDL}"
  echo "RUN_MMRL=${RUN_MMRL}"
  echo "RUN_BAYES_ADAPTER=${RUN_BAYES_ADAPTER}"
  echo "MANIFEST=${MANIFEST}"
}

main() {
  trap 'echo "[interrupt] stopping children"; cleanup_children; exit 130' INT TERM

  mkdir -p "${OUTPUT_ROOT}"
  reset_index_files_if_requested

  if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
    summarize_manifest "${MANIFEST}" "${TUNE_SUMMARY}"
    if [[ -f "${TUNE_SUMMARY}" ]]; then
      select_best_from_tune_summary "${TUNE_SUMMARY}" "${BEST_SUMMARY}" "${BEST_ENV}" || true
    fi
    summarize_manifest "${MANIFEST}" "${CONFIRM_SUMMARY}"
    summarize_manifest "${MANIFEST}" "${B2N_SUMMARY}"
    summarize_manifest "${MANIFEST}" "${OUTPUT_ROOT}/all_summary.csv"
    exit 0
  fi

  init_gpu_list
  init_search_spaces
  print_config

  local original_run_vcrm="${RUN_VCRM}"
  local original_run_bayes="${RUN_BAYES}"
  local original_run_mndl="${RUN_MNDL}"
  local original_run_mmrl="${RUN_MMRL}"
  local original_run_bayes_adapter="${RUN_BAYES_ADAPTER}"

  if [[ "${original_run_vcrm}" == "1" ]]; then
    echo
    echo "################################################################"
    echo "# Phase A: run all non-VCRMMMRL experiments first"
    echo "################################################################"

    RUN_VCRM=0
    RUN_BAYES="${original_run_bayes}"
    RUN_MNDL="${original_run_mndl}"
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
    RUN_MNDL=0
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
    RUN_MNDL="${original_run_mndl}"
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
  echo "  best env: ${BEST_ENV}"
  echo "  confirm summary: ${CONFIRM_SUMMARY}"
  echo "  b2n summary: ${B2N_SUMMARY}"
  echo "  all summary: ${OUTPUT_ROOT}/all_summary.csv"
}

main "$@"
