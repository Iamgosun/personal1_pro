#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# BayesRTMMRL independent-HPO ablation runner
#
# This script DOES NOT modify run_plan.sh.
#
# Protocol:
#   For each ablation variant x dataset x shot x seed:
#     1. Apply only the ablation-specific architecture switches.
#     2. Run HPO independently for that ablated method.
#     3. Let the validation set select this variant's own best hyperparameters.
#     4. Write ablation outputs to ABLATION_OUTPUT_ROOT, not output_refactor.
#     5. Build a full + independently-tuned ablation summary table.
#
# Optional full-model baseline path used by the summary:
#   ${FULL_OUTPUT_ROOT}/BayesRTMMRL/FS/fewshot_train/${dataset}/shots_${shot}/${BACKBONE_DIR}/${FULL_TAG}/seed${seed}/hpo_best_opts.json
#
# Ablation output path:
#   ${ABLATION_OUTPUT_ROOT}/BayesRTMMRL/FS/fewshot_train/${dataset}/shots_${shot}/${BACKBONE_DIR}/${ABLATION_TAG}/seed${seed}/
#
# Default target:
#   datasets: fgvc_aircraft eurosat food101
#   shots:    1 2 4 8 16 32
#   seeds:    1 2 3
# ============================================================

PROTOCOL=${PROTOCOL:-FS}
PHASE=${PHASE:-fewshot_train}
METHOD=${METHOD:-BayesRTMMRL}
EXEC_MODE=${EXEC_MODE:-online}

DATA_ROOT=${DATA_ROOT:-DATASETS}

# Full BayesRTMMRL results are used only as the optional baseline in the summary.
FULL_OUTPUT_ROOT=${FULL_OUTPUT_ROOT:-output_refactor}
FULL_TAG=${FULL_TAG:-default}

# Ablation results are written here.
ABLATION_OUTPUT_ROOT=${ABLATION_OUTPUT_ROOT:-output_bayesrt_ablation_independent_hpo}

BACKBONE=${BACKBONE:-ViT-B/16}
BACKBONE_DIR=${BACKBONE//\//-}

DATASETS_ARG=${DATASETS_ARG:-"fgvc_aircraft eurosat food101"}
SHOTS_ARG=${SHOTS_ARG:-"1 2 4 8 16 32"}
SEEDS_ARG=${SEEDS_ARG:-"1 2 3"}

# Variant registry
# ----------------
# Variants are data-driven. To add a new ablation, add one line to VARIANT_SPECS
# or pass VARIANT_SPECS_FILE=/path/to/variants.tsv.
#
# Format:
#   name|tag|KEY=VALUE;KEY=VALUE;...
#
# Examples:
#   r_only : Bayesian R projection only; T side disabled.
#   t_only : Bayesian T projection only; R side deterministic.
#
# Built-in variants preserve the original script behavior and add t_only.
# To add more variants, pass VARIANT_SPECS or VARIANT_SPECS_FILE without editing code.
DEFAULT_VARIANT_SPECS=$'# name|tag|opts\nr_only|bayesrt_r_only|BAYESRT_MMRL.BAYES_R_ENABLED=True;BAYESRT_MMRL.BAYES_T_ENABLED=False;BAYESRT_MMRL.T_TRAIN_MEAN=False;BAYESRT_MMRL.T_KL_WEIGHT=0.0\ntmean_trainable|bayesrt_tmean_trainable|BAYESRT_MMRL.BAYES_R_ENABLED=True;BAYESRT_MMRL.BAYES_T_ENABLED=True;BAYESRT_MMRL.T_TRAIN_MEAN=True\nt_only|bayesrt_t_only|BAYESRT_MMRL.BAYES_R_ENABLED=False;BAYESRT_MMRL.BAYES_T_ENABLED=True;BAYESRT_MMRL.R_KL_WEIGHT=0.0'
VARIANT_SPECS=${VARIANT_SPECS:-${DEFAULT_VARIANT_SPECS}}
VARIANT_SPECS_FILE=${VARIANT_SPECS_FILE:-}

# Default variants; restrict with VARIANTS_ARG="r_only" or similar.
VARIANTS_ARG=${VARIANTS_ARG:-"r_only tmean_trainable t_only"}

GPU_IDS=${GPU_IDS:-"0 1 2"}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

METHOD_CONFIG=${METHOD_CONFIG:-configs/methods/bayesrt_mmrl.yaml}
PROTOCOL_CONFIG=${PROTOCOL_CONFIG:-configs/protocols/fs.yaml}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-configs/runtime/mmrl_family.yaml}

read -r -a DATASET_LIST <<< "${DATASETS_ARG}"
read -r -a SHOT_LIST <<< "${SHOTS_ARG}"
read -r -a SEED_LIST <<< "${SEEDS_ARG}"
read -r -a VARIANT_LIST <<< "${VARIANTS_ARG}"

declare -gA VARIANT_TAG_BY_NAME=()
declare -gA VARIANT_OPTS_BY_NAME=()



trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}


load_variant_specs() {
  VARIANT_TAG_BY_NAME=()
  VARIANT_OPTS_BY_NAME=()

  local specs_content=""
  if [[ -n "${VARIANT_SPECS_FILE}" ]]; then
    if [[ ! -f "${VARIANT_SPECS_FILE}" ]]; then
      echo "[ERROR] VARIANT_SPECS_FILE not found: ${VARIANT_SPECS_FILE}" >&2
      exit 1
    fi
    specs_content="$(cat "${VARIANT_SPECS_FILE}")"
  else
    specs_content="${VARIANT_SPECS}"
  fi

  local line name tag opts extra
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line//$'\r'/}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue

    IFS='|' read -r name tag opts extra <<< "${line}"
    name="$(trim "${name:-}")"
    tag="$(trim "${tag:-}")"
    opts="$(trim "${opts:-}")"

    if [[ -n "${extra:-}" ]]; then
      echo "[ERROR] invalid variant spec with too many | fields:" >&2
      echo "        ${line}" >&2
      exit 1
    fi

    if [[ -z "${name}" || -z "${tag}" ]]; then
      echo "[ERROR] invalid variant spec; expected name|tag|KEY=VALUE;... :" >&2
      echo "        ${line}" >&2
      exit 1
    fi

    VARIANT_TAG_BY_NAME["${name}"]="${tag}"
    VARIANT_OPTS_BY_NAME["${name}"]="${opts}"
  done <<< "${specs_content}"

  if [[ "${#VARIANT_TAG_BY_NAME[@]}" -eq 0 ]]; then
    echo "[ERROR] no variant specs loaded. Set VARIANT_SPECS or VARIANT_SPECS_FILE." >&2
    exit 1
  fi

  local variant allowed
  allowed="$(printf '%s\n' "${!VARIANT_TAG_BY_NAME[@]}" | sort | tr '\n' ' ')"
  for variant in "${VARIANT_LIST[@]}"; do
    if [[ -z "${VARIANT_TAG_BY_NAME[${variant}]:-}" ]]; then
      echo "[ERROR] unknown variant: ${variant}" >&2
      echo "[ERROR] allowed variants: ${allowed}" >&2
      echo "[ERROR] add it via VARIANT_SPECS or VARIANT_SPECS_FILE." >&2
      exit 1
    fi
  done

  echo "[VARIANTS] loaded specs:"
  for variant in "${VARIANT_LIST[@]}"; do
    echo "  ${variant} -> tag=${VARIANT_TAG_BY_NAME[${variant}]} opts=${VARIANT_OPTS_BY_NAME[${variant}]}"
  done
}


write_effective_variant_specs() {
  local path=$1
  mkdir -p "$(dirname "${path}")"
  : > "${path}"

  local variant
  for variant in "${VARIANT_LIST[@]}"; do
    printf '%s\t%s\t%s\n' \
      "${variant}" \
      "${VARIANT_TAG_BY_NAME[${variant}]}" \
      "${VARIANT_OPTS_BY_NAME[${variant}]}" \
      >> "${path}"
  done
}


ablation_tag() {
  local variant=$1
  local tag="${VARIANT_TAG_BY_NAME[${variant}]:-}"

  if [[ -z "${tag}" ]]; then
    echo "[ERROR] unknown variant: ${variant}" >&2
    echo "[ERROR] call load_variant_specs before ablation_tag." >&2
    exit 1
  fi

  echo "${tag}"
}


ablation_outdir() {
  local dataset=$1
  local shot=$2
  local seed=$3
  local variant=$4

  local tag
  tag="$(ablation_tag "${variant}")"

  echo "${ABLATION_OUTPUT_ROOT}/${METHOD}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/${BACKBONE_DIR}/${tag}/seed${seed}"
}



variant_opts() {
  local variant=$1
  local opts="${VARIANT_OPTS_BY_NAME[${variant}]:-}"

  if [[ -z "${VARIANT_TAG_BY_NAME[${variant}]:-}" ]]; then
    echo "[ERROR] unknown variant: ${variant}" >&2
    exit 1
  fi

  # Print an argv-compatible alternating key/value list, one token per line.
  # VARIANT_OPTS syntax is semicolon-separated KEY=VALUE pairs.
  # Example:
  #   BAYESRT_MMRL.BAYES_R_ENABLED=False;BAYESRT_MMRL.BAYES_T_ENABLED=True
  [[ -z "${opts}" ]] && return 0

  local -a entries=()
  IFS=';' read -r -a entries <<< "${opts}"

  local entry key value
  for entry in "${entries[@]}"; do
    entry="$(trim "${entry}")"
    [[ -z "${entry}" ]] && continue

    if [[ "${entry}" != *=* ]]; then
      echo "[ERROR] invalid option in variant=${variant}: ${entry}" >&2
      echo "[ERROR] expected KEY=VALUE inside VARIANT_SPECS." >&2
      exit 1
    fi

    key="${entry%%=*}"
    value="${entry#*=}"
    key="$(trim "${key}")"
    value="$(trim "${value}")"

    if [[ -z "${key}" ]]; then
      echo "[ERROR] empty config key in variant=${variant}: ${entry}" >&2
      exit 1
    fi

    printf '%s\n%s\n' "${key}" "${value}"
  done
}


init_gpu_slots() {
  local -a base_gpus=()
  read -r -a base_gpus <<< "${GPU_IDS}"

  if [[ ${#base_gpus[@]} -eq 0 ]]; then
    echo "[ERROR] no GPU ids resolved. Set GPU_IDS, e.g. GPU_IDS=\"0 1\"." >&2
    exit 1
  fi

  GPU_LIST=()
  local gpu rep
  for gpu in "${base_gpus[@]}"; do
    for ((rep=0; rep<JOBS_PER_GPU; rep++)); do
      GPU_LIST+=("${gpu}")
    done
  done

  local nslots=${#GPU_LIST[@]}
  RUNNING_PIDS=()
  SLOT_GPU=()
  SLOT_DESC=()
  SLOT_LOG=()

  local i
  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_DESC[$i]=""
    SLOT_LOG[$i]=""
  done
}


wait_for_free_slot() {
  READY_SLOT=""

  while true; do
    local idx
    for idx in "${!RUNNING_PIDS[@]}"; do
      local pid="${RUNNING_PIDS[$idx]}"

      if [[ -z "${pid}" ]]; then
        READY_SLOT="${idx}"
        return 0
      fi

      if ! kill -0 "${pid}" 2>/dev/null; then
        local rc=0
        if wait "${pid}"; then
          rc=0
        else
          rc=$?
        fi

        local desc="${SLOT_DESC[$idx]}"
        local log="${SLOT_LOG[$idx]}"

        if [[ "${rc}" -eq 0 ]]; then
          echo "[OK]   ${desc}"
        else
          echo "[FAIL] ${desc} log=${log}" >&2
          FAILED_JOBS=$((FAILED_JOBS + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_GPU[$idx]=""
        SLOT_DESC[$idx]=""
        SLOT_LOG[$idx]=""

        READY_SLOT="${idx}"
        return 0
      fi
    done

    sleep "${SLEEP_SEC}"
  done
}


wait_all_jobs() {
  local idx
  for idx in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$idx]}"

    if [[ -n "${pid}" ]]; then
      local rc=0
      if wait "${pid}"; then
        rc=0
      else
        rc=$?
      fi

      local desc="${SLOT_DESC[$idx]}"
      local log="${SLOT_LOG[$idx]}"

      if [[ "${rc}" -eq 0 ]]; then
        echo "[OK]   ${desc}"
      else
        echo "[FAIL] ${desc} log=${log}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_DESC[$idx]=""
      SLOT_LOG[$idx]=""
    fi
  done
}


launch_one() {
  local gpu_id=$1
  local dataset=$2
  local shot=$3
  local seed=$4
  local variant=$5

  local outdir
  outdir="$(ablation_outdir "${dataset}" "${shot}" "${seed}" "${variant}")"

  local logfile="${outdir}/run.log"
  local statusfile="${outdir}/job_status.txt"
  local hpo_best="${outdir}/hpo_best_opts.json"
  local grid_summary="${outdir}/grid_search_summary.json"
  local test_report="${outdir}/test_report.json"
  local test_metrics="${outdir}/test_metrics.json"

  mkdir -p "${outdir}"

  if [[ "${SKIP_EXISTING}" == "1" ]] && { [[ -f "${hpo_best}" ]] || [[ -f "${grid_summary}" ]] || [[ -f "${test_report}" ]] || [[ -f "${test_metrics}" ]]; }; then
    echo "SKIP" > "${statusfile}"
    echo "[SKIP] variant=${variant} dataset=${dataset} shot=${shot} seed=${seed}"
    return 0
  fi

  local -a v_opts=()
  mapfile -t v_opts < <(variant_opts "${variant}")

  local tag
  tag="$(ablation_tag "${variant}")"

  : > "${logfile}"

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "GPU: ${gpu_id}"
    echo "METHOD: ${METHOD}"
    echo "VARIANT: ${variant}"
    echo "RUN_TAG: ${tag}"
    echo "HPO_MODE: independent_per_ablation_variant"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "PHASE: ${PHASE}"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "DATA_ROOT: ${DATA_ROOT}"
    echo "ABLATION_OUTPUT_ROOT: ${ABLATION_OUTPUT_ROOT}"
    echo "BACKBONE: ${BACKBONE}"
    echo "METHOD_CONFIG: ${METHOD_CONFIG}"
    echo "PROTOCOL_CONFIG: ${PROTOCOL_CONFIG}"
    echo "RUNTIME_CONFIG: ${RUNTIME_CONFIG}"
    echo "VARIANT_OPTS:"
    printf '  %q\n' "${v_opts[@]}"
    echo "============================================================"
  } >> "${logfile}"

  if CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
      --root "${DATA_ROOT}" \
      --dataset-config-file "configs/datasets/${dataset}.yaml" \
      --method-config-file "${METHOD_CONFIG}" \
      --protocol-config-file "${PROTOCOL_CONFIG}" \
      --runtime-config-file "${RUNTIME_CONFIG}" \
      --output-dir "${outdir}" \
      --method "${METHOD}" \
      --protocol "${PROTOCOL}" \
      --exec-mode "${EXEC_MODE}" \
      --seed "${seed}" \
      DATASET.NUM_SHOTS "${shot}" \
      DATASET.SUBSAMPLE_CLASSES "all" \
      MODEL.BACKBONE.NAME "${BACKBONE}" \
      HPO.ENABLED True \
      METHOD.TAG "${tag}" \
      "${v_opts[@]}" \
      >> "${logfile}" 2>&1; then
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: SUCCESS"
      echo "============================================================"
    } >> "${logfile}"

    echo "SUCCESS" > "${statusfile}"
    return 0
  else
    local rc=$?
    {
      echo
      echo "============================================================"
      echo "END: $(date '+%F %T')"
      echo "STATUS: FAILED"
      echo "EXIT_CODE: ${rc}"
      echo "============================================================"
    } >> "${logfile}"

    echo "FAILED(${rc})" > "${statusfile}"
    return "${rc}"
  fi
}


summarize_results() {
  local summary_dir="${ABLATION_OUTPUT_ROOT}/summary"
  mkdir -p "${summary_dir}"

  if [[ -d "${ABLATION_OUTPUT_ROOT}/${METHOD}/${PROTOCOL}" ]]; then
    echo "[SUMMARY] parsing independently tuned ablation root with result_parser.py"
    python evaluation/result_parser.py \
      "${ABLATION_OUTPUT_ROOT}/${METHOD}/${PROTOCOL}" \
      --split test || true
  fi

  echo "[SUMMARY] building full + independently tuned ablation comparison table"

  local variant_spec_path="${summary_dir}/variant_specs_effective.tsv"
  write_effective_variant_specs "${variant_spec_path}"

  python - \
    "${FULL_OUTPUT_ROOT}" \
    "${ABLATION_OUTPUT_ROOT}" \
    "${summary_dir}" \
    "${BACKBONE}" \
    "${FULL_TAG}" \
    "${DATASETS_ARG}" \
    "${SHOTS_ARG}" \
    "${SEEDS_ARG}" \
    "${VARIANTS_ARG}" \
    "${variant_spec_path}" \
    "${METHOD}" \
    "${PROTOCOL}" \
    "${PHASE}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


full_root = Path(sys.argv[1])
ablation_root = Path(sys.argv[2])
summary_dir = Path(sys.argv[3])
backbone = sys.argv[4]
full_tag = sys.argv[5]
datasets = sys.argv[6].split()
shots = sys.argv[7].split()
seeds = sys.argv[8].split()
variants = sys.argv[9].split()
variant_spec_path = Path(sys.argv[10])
method = sys.argv[11]
protocol = sys.argv[12]
phase = sys.argv[13]
backbone_dir = backbone.replace("/", "-")

tag_by_variant: dict[str, str] = {}
if variant_spec_path.is_file():
    with variant_spec_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tag_by_variant[parts[0]] = parts[1]


def tag_for_variant(variant: str) -> str:
    if variant == "full":
        return full_tag
    try:
        return tag_by_variant[variant]
    except KeyError as exc:
        raise ValueError(f"unknown variant in effective spec: {variant}") from exc


def root_for_variant(variant: str) -> Path:
    return full_root if variant == "full" else ablation_root


def case_dir(root: Path, dataset: str, shot: str, seed: str, tag: str) -> Path:
    return (
        root
        / method
        / protocol
        / phase
        / dataset
        / f"shots_{shot}"
        / backbone_dir
        / tag
        / f"seed{seed}"
    )


def to_float(value: Any):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_best_dir(base_dir: Path) -> Path | None:
    best_path = base_dir / "hpo_best_opts.json"
    if not best_path.is_file():
        return None

    try:
        data = load_json(best_path)
    except Exception:
        return None

    best = data.get("best", {})
    if not isinstance(best, dict):
        return None

    output_dir = best.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        return None

    path = Path(output_dir)
    if not path.is_absolute():
        cwd_candidate = path
        basedir_candidate = base_dir / path
        if cwd_candidate.exists():
            path = cwd_candidate
        else:
            path = basedir_candidate

    return path


def candidate_report_paths(base_dir: Path) -> list[Path]:
    paths = [
        base_dir / "test_metrics.json",
        base_dir / "test_report.json",
    ]

    best_dir = resolve_best_dir(base_dir)
    if best_dir is not None:
        paths.extend([
            best_dir / "test_metrics.json",
            best_dir / "test_report.json",
        ])

    hpo_dir = base_dir / "hpo_candidates"
    if hpo_dir.is_dir():
        paths.extend(sorted(hpo_dir.glob("**/test_metrics.json")))
        paths.extend(sorted(hpo_dir.glob("**/test_report.json")))

    out = []
    seen = set()
    for p in paths:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(p)
    return out


def find_report_path(base_dir: Path) -> Path | None:
    for path in candidate_report_paths(base_dir):
        if path.is_file():
            return path
    return None


def load_best_opts(base_dir: Path) -> dict[str, str]:
    path = base_dir / "hpo_best_opts.json"
    if not path.is_file():
        return {}

    data = load_json(path)
    opts = data.get("opts", [])

    out = {}
    if isinstance(opts, list):
        for i in range(0, len(opts), 2):
            if i + 1 < len(opts):
                out[str(opts[i])] = str(opts[i + 1])

    return out


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    row = {}

    for block_name, prefix in [
        ("metrics", ""),
        ("metrics_calibrated", "calibrated_"),
    ]:
        block = report.get(block_name, {})
        if not isinstance(block, dict):
            continue

        for key, value in block.items():
            val = to_float(value)
            if val is not None:
                row[f"{prefix}{key}"] = val

    for key, value in report.items():
        if key in {"metrics", "metrics_calibrated", "temperature_scaling"}:
            continue
        val = to_float(value)
        if val is not None:
            row[key] = val

    temp_info = report.get("temperature_scaling", {})
    if isinstance(temp_info, dict):
        val = to_float(temp_info.get("temperature"))
        if val is not None:
            row["temperature"] = val

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


run_rows = []
missing = []
all_variants = ["full"] + variants

for dataset in datasets:
    for shot in shots:
        for seed in seeds:
            for variant in all_variants:
                tag = tag_for_variant(variant)
                root = root_for_variant(variant)
                base_dir = case_dir(root, dataset, shot, seed, tag)
                report_path = find_report_path(base_dir)

                if report_path is None:
                    missing.append(str(base_dir))
                    continue

                report = load_json(report_path)
                hp = load_best_opts(base_dir)

                row = {
                    "dataset": dataset,
                    "shot": int(shot),
                    "seed": int(seed),
                    "variant": variant,
                    "tag": tag,
                    "case_dir": str(base_dir),
                    "report_path": str(report_path),
                    "hpo_best_opts_path": str(base_dir / "hpo_best_opts.json"),
                }

                row.update(flatten_report(report))

                for k, v in hp.items():
                    row[f"hp__{k}"] = v

                run_rows.append(row)


summary_rows = []
groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}

for row in run_rows:
    key = (row["dataset"], int(row["shot"]), row["variant"])
    groups.setdefault(key, []).append(row)


for (dataset, shot, variant), rows in sorted(groups.items()):
    out = {
        "dataset": dataset,
        "shot": shot,
        "variant": variant,
        "num_seeds": len(rows),
        "seeds": " ".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])),
    }

    metric_keys = []
    for row in rows:
        for key, value in row.items():
            if key in {"dataset", "shot", "seed", "variant", "tag", "case_dir", "report_path", "hpo_best_opts_path"}:
                continue
            if key.startswith("hp__"):
                continue
            if to_float(value) is not None and key not in metric_keys:
                metric_keys.append(key)

    for key in metric_keys:
        values = [to_float(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            out[f"{key}_mean"] = mean(values)
            out[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0

    hp_keys = []
    for row in rows:
        for key in row:
            if key.startswith("hp__") and key not in hp_keys:
                hp_keys.append(key)
    for key in hp_keys:
        out[key] = " | ".join(str(r.get(key, "")) for r in sorted(rows, key=lambda x: x["seed"]))

    summary_rows.append(out)


summary_dir.mkdir(parents=True, exist_ok=True)

runs_csv = summary_dir / "bayesrt_independent_hpo_ablation_runs.csv"
summary_csv = summary_dir / "bayesrt_independent_hpo_ablation_summary.csv"
missing_txt = summary_dir / "missing_reports.txt"

write_csv(runs_csv, run_rows)
write_csv(summary_csv, summary_rows)

with missing_txt.open("w", encoding="utf-8") as f:
    for path in missing:
        f.write(path + "\n")

print(f"[SUMMARY] saved per-run table: {runs_csv}")
print(f"[SUMMARY] saved aggregate table: {summary_csv}")
print(f"[SUMMARY] saved missing report list: {missing_txt}")
print(f"[SUMMARY] num_run_rows={len(run_rows)}")
print(f"[SUMMARY] num_summary_rows={len(summary_rows)}")
print(f"[SUMMARY] num_missing={len(missing)}")
PY
}


main() {
  load_variant_specs
  init_gpu_slots

  FAILED_JOBS=0

  local dataset shot seed variant
  for variant in "${VARIANT_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      for shot in "${SHOT_LIST[@]}"; do
        for seed in "${SEED_LIST[@]}"; do
          local outdir
          outdir="$(ablation_outdir "${dataset}" "${shot}" "${seed}" "${variant}")"

          local logfile="${outdir}/run.log"
          local hpo_best="${outdir}/hpo_best_opts.json"
          local grid_summary="${outdir}/grid_search_summary.json"
          local test_report="${outdir}/test_report.json"
          local test_metrics="${outdir}/test_metrics.json"

          if [[ "${SKIP_EXISTING}" == "1" ]] && { [[ -f "${hpo_best}" ]] || [[ -f "${grid_summary}" ]] || [[ -f "${test_report}" ]] || [[ -f "${test_metrics}" ]]; }; then
            mkdir -p "${outdir}"
            echo "SKIP" > "${outdir}/job_status.txt"
            echo "[SKIP] variant=${variant} dataset=${dataset} shot=${shot} seed=${seed}"
            continue
          fi

          wait_for_free_slot

          local slot="${READY_SLOT}"
          local gpu_id="${GPU_LIST[$slot]}"
          local desc="gpu=${gpu_id} variant=${variant} dataset=${dataset} shot=${shot} seed=${seed}"

          (
            launch_one "${gpu_id}" "${dataset}" "${shot}" "${seed}" "${variant}"
          ) &

          RUNNING_PIDS[$slot]=$!
          SLOT_GPU[$slot]="${gpu_id}"
          SLOT_DESC[$slot]="${desc}"
          SLOT_LOG[$slot]="${logfile}"

          echo "[LAUNCH] ${desc}"
        done
      done
    done
  done

  wait_all_jobs

  summarize_results

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)." >&2
    exit 1
  fi

  echo "[DONE] all BayesRTMMRL independent-HPO ablations finished."
}


if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
  load_variant_specs
  summarize_results
  exit 0
fi

main "$@"
