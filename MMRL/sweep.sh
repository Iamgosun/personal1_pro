#!/bin/bash
set -euo pipefail

# ============================================================
# BayesMMRL rep_tokens-only sweep
#
# 目标：
# 1) 只搜索 rep_tokens 的 posterior 方案：
#    global | per_token | diagonal | matrix_normal_diag | matrix_normal_diag_lowrank
# 2) 默认只搜索 zero prior（不搜索 clip_joint prior）
# 3) 自动选最优配置：
#    - 先看 ACC
#    - 如果 ACC 与最优只差 <= 0.15，则优先选 ECE 更低的
# 4) 找到 best tag 后，自动做最终确认：
#    - shots: 1..16
#    - seeds: 1..3
#
# 用法：
#   bash MMRL/bayes_mmrl_rep_tokens_only_sweep.sh GPU_IDS="0 1 2 3"
#
# 说明：
#   请在 MMRL 根目录运行，或显式设置 PROJECT_DIR=/path/to/MMRL
# ============================================================

# ---------------------------
# 基础环境
# ---------------------------
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
ROOT=${ROOT:-DATASETS}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}

# 搜索阶段
STAGE=${STAGE:-coarse3}
DATASETS=${DATASETS:-}
SHOTS=${SHOTS:-}
SEEDS=${SEEDS:-}

# 最终确认阶段
CONFIRM_DATASETS=${CONFIRM_DATASETS:-}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-}

OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayes_mmrl_rep_tokens_only}

# GPU 调度
NGPU=${NGPU:-}
GPU_IDS=${GPU_IDS:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# 开关
RUN_ZERO_PRIOR=${RUN_ZERO_PRIOR:-1}
RUN_CLIP_JOINT_PRIOR=${RUN_CLIP_JOINT_PRIOR:-0}
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}
AUTO_SELECT_BEST=${AUTO_SELECT_BEST:-1}
AUTO_CONFIRM=${AUTO_CONFIRM:-1}

# 当 ACC 与最优配置差距 <= 0.15 时，优先选 ECE 更低的
# 这里默认 accuracy_mean 使用百分数制，例如 84.32
ACC_CLOSE_THRESHOLD=${ACC_CLOSE_THRESHOLD:-0.15}

# ---------------------------
# 固定主体超参
# ---------------------------
PAPER_ALPHA=${PAPER_ALPHA:-0.7}
PAPER_REG_WEIGHT=${PAPER_REG_WEIGHT:-0.5}
PAPER_N_REP_TOKENS=${PAPER_N_REP_TOKENS:-5}
PAPER_REP_LAYERS=${PAPER_REP_LAYERS:-'[6,7,8,9,10,11,12]'}
PAPER_REP_DIM=${PAPER_REP_DIM:-512}

COMMON_N_MC_TRAIN=${COMMON_N_MC_TRAIN:-3}
COMMON_N_MC_TEST=${COMMON_N_MC_TEST:-10}
COMMON_EVAL_MODE=${COMMON_EVAL_MODE:-mc_predictive}
COMMON_EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION:-logit_mean}
COMMON_KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS:-8}

# ---------------------------
# rep_tokens 搜索空间
# ---------------------------
REP_SIGMA_MODES=${REP_SIGMA_MODES:-"global per_token diagonal matrix_normal_diag matrix_normal_diag_lowrank"}

# zero prior
ZERO_PRIOR_STDS=${ZERO_PRIOR_STDS:-"0.02 0.05 0.1 0.5 1.0"}
ZERO_KL_LIST=${ZERO_KL_LIST:-"1e-6 1e-5 1e-4 5e-4 1e-3"}

# clip_joint prior（默认关闭；保留代码是为了以后需要时可直接打开）
CLIP_PRIOR_STDS=${CLIP_PRIOR_STDS:-"0.05"}
CLIP_KL_LIST=${CLIP_KL_LIST:-"1e-4 5e-4 1e-3"}
CLIP_BLEND_LIST=${CLIP_BLEND_LIST:-"0.2 0.5"}
CLIP_SCALE_LIST=${CLIP_SCALE_LIST:-"0.02 0.05"}

# matrix normal 相关
MN_ENFORCE_TRACE=${MN_ENFORCE_TRACE:-True}
MN_LOWRANK_RANKS=${MN_LOWRANK_RANKS:-"4 8"}

METHOD_CFG="configs/methods/bayesmmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

SEARCH_ROOT=""
CONFIRM_ROOT=""
GLOBAL_SEARCH_SUMMARY=""
GLOBAL_CONFIRM_SUMMARY=""
BEST_CONFIG_SUMMARY=""
BEST_CONFIG_ENV=""

# ------------------------------------------------------------
# 协议语义
# ------------------------------------------------------------
resolve_phase_semantics() {
  case "$1" in
    B2N) echo "train_base base configs/protocols/b2n.yaml" ;;
    FS)  echo "fewshot_train all configs/protocols/fs.yaml" ;;
    CD)  echo "cross_train all configs/protocols/cd.yaml" ;;
    *)
      echo "Unknown PROTOCOL=$1" >&2
      exit 1
      ;;
  esac
}

read -r PHASE SUBSAMPLE PROTOCOL_CFG <<< "$(resolve_phase_semantics "$PROTOCOL")"

# ------------------------------------------------------------
# 第一阶段默认规模
# ------------------------------------------------------------
resolve_stage_defaults() {
  if [[ -z "${DATASETS}" ]]; then
    case "$STAGE" in
      coarse3)
        DATASETS="caltech101 oxford_pets ucf101"
        ;;
      full11)
        DATASETS="caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets sun397 stanford_cars ucf101"
        ;;
      *)
        echo "Unknown STAGE=${STAGE}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ -z "${SHOTS}" ]]; then
    SHOTS="16"
  fi

  if [[ -z "${SEEDS}" ]]; then
    SEEDS="1"
  fi
}

# ------------------------------------------------------------
# 最终确认默认规模
# ------------------------------------------------------------
resolve_confirm_defaults() {
  if [[ -z "${CONFIRM_DATASETS}" ]]; then
    CONFIRM_DATASETS="${DATASETS}"
  fi

  if [[ -z "${CONFIRM_SHOTS}" ]]; then
    CONFIRM_SHOTS="1 2 4 8 16 32"
  fi

  if [[ -z "${CONFIRM_SEEDS}" ]]; then
    CONFIRM_SEEDS="1 2 3"
  fi
}

# ------------------------------------------------------------
# GPU 初始化
# ------------------------------------------------------------
init_gpu_list() {
  GPU_LIST=()

  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a GPU_LIST <<< "${GPU_IDS}"
  elif [[ -n "${NGPU}" ]]; then
    local i
    for ((i=0; i<NGPU; i++)); do
      GPU_LIST+=("$i")
    done
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      while IFS= read -r idx; do
        [[ -n "${idx}" ]] && GPU_LIST+=("${idx}")
      done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    fi
  fi

  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "No visible GPU found. Please set GPU_IDS or NGPU." >&2
    exit 1
  fi

  if [[ -z "${NGPU}" ]]; then
    NGPU=${#GPU_LIST[@]}
  fi

  if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="${GPU_LIST[*]}"
  fi

  echo "[GPU] using: ${GPU_IDS}"
  echo "[GPU] slots: ${#GPU_LIST[@]}"
}

# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
sanitize() {
  local s="$1"
  s="${s//\//-}"
  s="${s// /_}"
  s="${s//,/}"
  s="${s//[/}"
  s="${s//]/}"
  s="${s//:/-}"
  echo "${s}"
}

build_case_root() {
  local base_root=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local tag=$5
  local backbone_tag="${BACKBONE//\//-}"
  echo "${base_root}/${method}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/${backbone_tag}/${tag}"
}

build_outdir() {
  local base_root=$1
  local method=$2
  local dataset=$3
  local shot=$4
  local tag=$5
  local seed=$6
  echo "$(build_case_root "${base_root}" "${method}" "${dataset}" "${shot}" "${tag}")/seed${seed}"
}

write_log_header() {
  local logfile=$1
  local gpu_id=$2
  local method=$3
  local dataset=$4
  local shot=$5
  local seed=$6
  local tag=$7
  local stage_name=$8
  local base_root=$9

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: ${stage_name}"
    echo "GPU: ${gpu_id}"
    echo "METHOD: ${method}"
    echo "PROTOCOL: ${PROTOCOL}"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "DATA_ROOT: ${ROOT}"
    echo "OUTPUT_ROOT: ${base_root}"
    echo "BACKBONE: ${BACKBONE}"
    echo "============================================================"
  } >> "${logfile}"
}

print_banner() {
  local msg="$1"
  echo
  echo "################################################################"
  echo "# ${msg}"
  echo "################################################################"
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
      echo "[cleanup] removed checkpoint dir: ${outdir}/refactor_model" >> "${logfile}"
    fi
  fi
}

cleanup_broken_resume_if_needed() {
  local outdir=$1
  local logfile=${2:-}
  local ckpt_dir="${outdir}/refactor_model"

  if [[ ! -d "${ckpt_dir}" ]]; then
    return 0
  fi

  local broken=0
  if [[ ! -f "${ckpt_dir}/checkpoint" ]]; then
    broken=1
  elif ! compgen -G "${ckpt_dir}/model*.pth.tar*" > /dev/null; then
    broken=1
  fi

  if [[ "${broken}" == "1" ]]; then
    rm -rf "${ckpt_dir}"
    if [[ -n "${logfile}" ]]; then
      echo "[cleanup] removed broken resume state: ${ckpt_dir}" >> "${logfile}"
    fi
  fi
}

append_summary_to_global() {
  local summary_csv=$1
  local global_summary=$2
  local stage_name=$3
  local scheme=$4
  local dataset=$5
  local shot=$6
  local tag=$7

  local header
  header="$(head -n 1 "${summary_csv}")"

  if [[ ! -f "${global_summary}" ]]; then
    echo "stage_name,scheme,dataset,shot,tag,${header}" > "${global_summary}"
  fi

  tail -n +2 "${summary_csv}" | while IFS= read -r line; do
    echo "${stage_name},${scheme},${dataset},${shot},${tag},${line}" >> "${global_summary}"
  done
}

summarize_one_tag() {
  local base_root=$1
  local global_summary=$2
  local method=$3
  local stage_name=$4
  local scheme=$5
  local dataset=$6
  local shot=$7
  local tag=$8

  local case_root summary_csv
  case_root="$(build_case_root "${base_root}" "${method}" "${dataset}" "${shot}" "${tag}")"

  (
    cd "${PROJECT_DIR}"
    python evaluation/result_parser.py "${case_root}" --split test >/dev/null 2>&1 || true
  )

  summary_csv="${case_root}/test_summary.csv"
  if [[ -f "${summary_csv}" ]]; then
    append_summary_to_global "${summary_csv}" "${global_summary}" "${stage_name}" "${scheme}" "${dataset}" "${shot}" "${tag}"
  fi
}

print_summary_table() {
  local summary_csv=$1
  local title=$2

  echo
  echo "============================================================"
  echo "${title}"
  echo "${summary_csv}"
  echo "============================================================"

  if [[ -f "${summary_csv}" ]]; then
    python - <<PY
import pandas as pd
from pathlib import Path

path = Path(r"${summary_csv}")
df = pd.read_csv(path)
preferred_cols = [
    "stage_name", "scheme", "dataset", "shot", "tag",
    "accuracy_mean", "accuracy_std",
    "macro_f1_mean", "ece_mean", "brier_mean", "nll_mean",
]
keep = [c for c in preferred_cols if c in df.columns]
if keep:
    sort_cols = [c for c in ["scheme", "dataset", "shot", "accuracy_mean"] if c in df.columns]
    ascending = [False if c == "accuracy_mean" else True for c in sort_cols]
    print(df[keep].sort_values(by=sort_cols, ascending=ascending).to_string(index=False))
else:
    print(df.to_string(index=False))
PY
  fi
}

print_best_table() {
  local summary_csv=$1
  echo
  echo "============================================================"
  echo "Auto-selected best configs"
  echo "${summary_csv}"
  echo "============================================================"
  if [[ -f "${summary_csv}" ]]; then
    python - <<PY
import pandas as pd
from pathlib import Path
path = Path(r"${summary_csv}")
df = pd.read_csv(path)
print(df.to_string(index=False))
PY
  fi
}

# ------------------------------------------------------------
# 并行任务池
# ------------------------------------------------------------
declare -ga RUNNING_PIDS
declare -ga SLOT_GPU
declare -ga SLOT_LOG
declare -ga SLOT_DESC

FAILED_JOBS=0
READY_SLOT=""

cleanup_children() {
  local p
  for p in "${RUNNING_PIDS[@]:-}"; do
    if [[ -n "${p:-}" ]] && kill -0 "${p}" 2>/dev/null; then
      kill "${p}" 2>/dev/null || true
    fi
  done
}

wait_for_any_slot() {
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

        if [[ "${rc}" -eq 0 ]]; then
          echo "[done] ${SLOT_DESC[$idx]}"
        else
          echo "[failed] ${SLOT_DESC[$idx]}  log: ${SLOT_LOG[$idx]}" >&2
          FAILED_JOBS=$((FAILED_JOBS + 1))
        fi

        RUNNING_PIDS[$idx]=""
        SLOT_GPU[$idx]=""
        SLOT_LOG[$idx]=""
        SLOT_DESC[$idx]=""
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

      if [[ "${rc}" -eq 0 ]]; then
        echo "[done] ${SLOT_DESC[$idx]}"
      else
        echo "[failed] ${SLOT_DESC[$idx]}  log: ${SLOT_LOG[$idx]}" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
      fi

      RUNNING_PIDS[$idx]=""
      SLOT_GPU[$idx]=""
      SLOT_LOG[$idx]=""
      SLOT_DESC[$idx]=""
    fi
  done
}

init_slots() {
  local nslots=${#GPU_LIST[@]}
  local i
  for ((i=0; i<nslots; i++)); do
    RUNNING_PIDS[$i]=""
    SLOT_GPU[$i]=""
    SLOT_LOG[$i]=""
    SLOT_DESC[$i]=""
  done
}

announce_job_started() {
  local slot=$1
  echo "[launch] [slot=${slot}] [gpu=${SLOT_GPU[$slot]}] [pid=${RUNNING_PIDS[$slot]}] ${SLOT_DESC[$slot]}"
  echo "         log: ${SLOT_LOG[$slot]}"
}

# ------------------------------------------------------------
# 搜索空间构造
# ------------------------------------------------------------
declare -a REP_TAGS
declare -A REP_OPTS

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

register_rep_case_common() {
  local tag=$1
  local prior_mode=$2
  local sigma=$3
  local prior_std=$4
  local kl=$5
  shift 5

  register_case REP_TAGS REP_OPTS \
    "${tag}" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE "${prior_mode}" \
    BAYES_MMRL.REP_SIGMA_MODE "${sigma}" \
    BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
    BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
    "$@"
}

init_rep_search_space() {
  REP_TAGS=()
  REP_OPTS=()

  local sigma prior_std kl blend scale rank base_tag

  if [[ "${RUN_ZERO_PRIOR}" == "1" ]]; then
    for sigma in ${REP_SIGMA_MODES}; do
      for prior_std in ${ZERO_PRIOR_STDS}; do
        for kl in ${ZERO_KL_LIST}; do
          base_tag="rep_zero_sig-${sigma}_pstd-${prior_std}_kl-${kl}"

          if [[ "${sigma}" == "matrix_normal_diag_lowrank" ]]; then
            for rank in ${MN_LOWRANK_RANKS}; do
              register_rep_case_common \
                "${base_tag}_rank-${rank}" \
                zero "${sigma}" "${prior_std}" "${kl}" \
                BAYES_MMRL.REP_MN_ENFORCE_TRACE "${MN_ENFORCE_TRACE}" \
                BAYES_MMRL.REP_MN_LOWRANK_RANK "${rank}"
            done
          elif [[ "${sigma}" == "matrix_normal_diag" ]]; then
            register_rep_case_common \
              "${base_tag}" \
              zero "${sigma}" "${prior_std}" "${kl}" \
              BAYES_MMRL.REP_MN_ENFORCE_TRACE "${MN_ENFORCE_TRACE}"
          else
            register_rep_case_common \
              "${base_tag}" \
              zero "${sigma}" "${prior_std}" "${kl}"
          fi
        done
      done
    done
  fi

  if [[ "${RUN_CLIP_JOINT_PRIOR}" == "1" ]]; then
    for sigma in ${REP_SIGMA_MODES}; do
      for prior_std in ${CLIP_PRIOR_STDS}; do
        for kl in ${CLIP_KL_LIST}; do
          for blend in ${CLIP_BLEND_LIST}; do
            for scale in ${CLIP_SCALE_LIST}; do
              base_tag="rep_clip_sig-${sigma}_pstd-${prior_std}_kl-${kl}_blend-${blend}_scale-${scale}"

              if [[ "${sigma}" == "matrix_normal_diag_lowrank" ]]; then
                for rank in ${MN_LOWRANK_RANKS}; do
                  register_rep_case_common \
                    "${base_tag}_rank-${rank}" \
                    clip_joint "${sigma}" "${prior_std}" "${kl}" \
                    BAYES_MMRL.CLIP_PRIOR_BLEND "${blend}" \
                    BAYES_MMRL.CLIP_PRIOR_SCALE "${scale}" \
                    BAYES_MMRL.REP_MN_ENFORCE_TRACE "${MN_ENFORCE_TRACE}" \
                    BAYES_MMRL.REP_MN_LOWRANK_RANK "${rank}"
                done
              elif [[ "${sigma}" == "matrix_normal_diag" ]]; then
                register_rep_case_common \
                  "${base_tag}" \
                  clip_joint "${sigma}" "${prior_std}" "${kl}" \
                  BAYES_MMRL.CLIP_PRIOR_BLEND "${blend}" \
                  BAYES_MMRL.CLIP_PRIOR_SCALE "${scale}" \
                  BAYES_MMRL.REP_MN_ENFORCE_TRACE "${MN_ENFORCE_TRACE}"
              else
                register_rep_case_common \
                  "${base_tag}" \
                  clip_joint "${sigma}" "${prior_std}" "${kl}" \
                  BAYES_MMRL.CLIP_PRIOR_BLEND "${blend}" \
                  BAYES_MMRL.CLIP_PRIOR_SCALE "${scale}"
              fi
            done
          done
        done
      done
    done
  fi

  echo "[search space] total rep_tokens cases: ${#REP_TAGS[@]}"
}

# ------------------------------------------------------------
# 通用执行函数：可用于搜索阶段或最终确认阶段
# ------------------------------------------------------------
run_rep_stage_with_tags() {
  local base_root=$1
  local global_summary=$2
  local stage_name=$3
  local scheme_name=$4
  local datasets_str=$5
  local shots_str=$6
  local seeds_str=$7
  shift 7
  local tags=("$@")

  print_banner "Start ${stage_name}: ${scheme_name}"
  init_slots

  local dataset shot tag seed slot gpu_id outdir logfile desc opts_str
  local tag_count=${#tags[@]}
  echo "[info] ${scheme_name} tag count: ${tag_count}"

  for dataset in ${datasets_str}; do
    for shot in ${shots_str}; do
      for tag in "${tags[@]}"; do
        opts_str="${REP_OPTS[$tag]}"
        read -r -a EXTRA_OPTS <<< "${opts_str}"

        for seed in ${seeds_str}; do
          outdir="$(build_outdir "${base_root}" "BayesMMRL" "${dataset}" "${shot}" "${tag}" "${seed}")"
          logfile="${outdir}/run.log"

          if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
            mkdir -p "${outdir}"
            cleanup_checkpoint_if_ready "${outdir}"
            echo "[skip] ${scheme_name} dataset=${dataset} shot=${shot} tag=${tag} seed=${seed}"
            continue
          fi

          wait_for_any_slot
          slot="${READY_SLOT}"
          gpu_id="${GPU_LIST[$slot]}"
          desc="${scheme_name} | dataset=${dataset} | shot=${shot} | seed=${seed} | tag=${tag}"

          (
            mkdir -p "${outdir}"
            : > "${logfile}"
            write_log_header "${logfile}" "${gpu_id}" "BayesMMRL" "${dataset}" "${shot}" "${seed}" "${tag}" "${stage_name}" "${base_root}"
            cleanup_broken_resume_if_needed "${outdir}" "${logfile}"

            echo "[run] ${desc}" >> "${logfile}"
            echo "[info] rep_tokens-only stage" >> "${logfile}"
            echo "[info] EVAL_MODE=${COMMON_EVAL_MODE}, EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION}" >> "${logfile}"
            echo "[info] KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS}" >> "${logfile}"

            (
              cd "${PROJECT_DIR}"
              CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
                --root "${ROOT}" \
                --dataset-config-file "configs/datasets/${dataset}.yaml" \
                --method-config-file "${METHOD_CFG}" \
                --protocol-config-file "${PROTOCOL_CFG}" \
                --runtime-config-file "${RUNTIME_CFG}" \
                --output-dir "${outdir}" \
                --method BayesMMRL \
                --protocol "${PROTOCOL}" \
                --exec-mode "${EXEC_MODE}" \
                --seed "${seed}" \
                DATASET.NUM_SHOTS "${shot}" \
                DATASET.SUBSAMPLE_CLASSES "${SUBSAMPLE}" \
                MODEL.BACKBONE.NAME "${BACKBONE}" \
                BAYES_MMRL.ALPHA "${PAPER_ALPHA}" \
                BAYES_MMRL.REG_WEIGHT "${PAPER_REG_WEIGHT}" \
                BAYES_MMRL.N_REP_TOKENS "${PAPER_N_REP_TOKENS}" \
                BAYES_MMRL.REP_LAYERS "${PAPER_REP_LAYERS}" \
                BAYES_MMRL.REP_DIM "${PAPER_REP_DIM}" \
                BAYES_MMRL.N_MC_TRAIN "${COMMON_N_MC_TRAIN}" \
                BAYES_MMRL.N_MC_TEST "${COMMON_N_MC_TEST}" \
                BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}" \
                BAYES_MMRL.EVAL_USE_POSTERIOR_MEAN False \
                BAYES_MMRL.EVAL_AGGREGATION "${COMMON_EVAL_AGGREGATION}" \
                BAYES_MMRL.KL_WARMUP_EPOCHS "${COMMON_KL_WARMUP_EPOCHS}" \
                "${EXTRA_OPTS[@]}"
            ) >> "${logfile}" 2>&1

            cleanup_checkpoint_if_ready "${outdir}" "${logfile}"
          ) &

          RUNNING_PIDS[$slot]=$!
          SLOT_GPU[$slot]="${gpu_id}"
          SLOT_LOG[$slot]="${logfile}"
          SLOT_DESC[$slot]="${desc}"
          announce_job_started "${slot}"
        done
      done
    done
  done

  wait_all_jobs

  local dataset2 shot2 tag2
  for dataset2 in ${datasets_str}; do
    for shot2 in ${shots_str}; do
      for tag2 in "${tags[@]}"; do
        summarize_one_tag "${base_root}" "${global_summary}" "BayesMMRL" "${stage_name}" "${scheme_name}" "${dataset2}" "${shot2}" "${tag2}"
      done
    done
  done
}

# ------------------------------------------------------------
# 选最佳参数：
# 规则：
# 1) 先看 ACC
# 2) 若 ACC 与最优只差 <= ACC_CLOSE_THRESHOLD，则优先选 ECE 更低的
# 3) 若没有 ECE，则按 ACC / F1 / NLL / Brier 兜底
# ------------------------------------------------------------
select_best_tag_from_summary() {
  local summary_csv=$1
  local best_summary=$2
  local best_env=$3
  local datasets_str=$4
  local shots_str=$5

  if [[ ! -f "${summary_csv}" ]]; then
    echo "[error] missing summary file: ${summary_csv}" >&2
    exit 1
  fi

  python - <<PY
import pandas as pd
from pathlib import Path

summary_csv = Path(r"${summary_csv}")
best_summary = Path(r"${best_summary}")
best_env = Path(r"${best_env}")

datasets = set("""${datasets_str}""".split())
shots = set("""${shots_str}""".split())
acc_close_threshold = float("""${ACC_CLOSE_THRESHOLD}""".strip())

df = pd.read_csv(summary_csv)
required_cols = {"scheme", "dataset", "shot", "tag", "accuracy_mean"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"summary missing columns: {missing}")

df["dataset"] = df["dataset"].astype(str)
df["shot"] = df["shot"].astype(str)

df = df[df["dataset"].isin(datasets) & df["shot"].isin(shots)]
if df.empty:
    raise SystemExit("no rows left after dataset/shot filtering")

agg_dict = {
    "accuracy_mean": ("accuracy_mean", "mean"),
    "row_count": ("accuracy_mean", "size"),
}
if "accuracy_std" in df.columns:
    agg_dict["accuracy_std_mean"] = ("accuracy_std", "mean")
if "macro_f1_mean" in df.columns:
    agg_dict["macro_f1_mean"] = ("macro_f1_mean", "mean")
if "ece_mean" in df.columns:
    agg_dict["ece_mean"] = ("ece_mean", "mean")
if "brier_mean" in df.columns:
    agg_dict["brier_mean"] = ("brier_mean", "mean")
if "nll_mean" in df.columns:
    agg_dict["nll_mean"] = ("nll_mean", "mean")

agg = (
    df.groupby(["scheme", "tag"], as_index=False)
      .agg(**agg_dict)
)

max_acc = agg["accuracy_mean"].max()

# ACC 足够接近最优的候选
acc_candidates = agg[agg["accuracy_mean"] >= max_acc - acc_close_threshold].copy()

# 如果存在 ece_mean，则在 ACC 接近时优先选更低 ECE
sort_cols = []
ascending = []

if "ece_mean" in acc_candidates.columns:
    sort_cols.append("ece_mean")
    ascending.append(True)

sort_cols.append("accuracy_mean")
ascending.append(False)

if "macro_f1_mean" in acc_candidates.columns:
    sort_cols.append("macro_f1_mean")
    ascending.append(False)

if "nll_mean" in acc_candidates.columns:
    sort_cols.append("nll_mean")
    ascending.append(True)

if "brier_mean" in acc_candidates.columns:
    sort_cols.append("brier_mean")
    ascending.append(True)

sort_cols.append("tag")
ascending.append(True)

best = acc_candidates.sort_values(sort_cols, ascending=ascending).head(1).copy()

best_summary.parent.mkdir(parents=True, exist_ok=True)
best.to_csv(best_summary, index=False)

best_tag = best.iloc[0]["tag"]
best_acc = best.iloc[0]["accuracy_mean"]

best_env.parent.mkdir(parents=True, exist_ok=True)
best_env.write_text(
    "\\n".join([
        f'REP_BEST_TAG="{best_tag}"',
        f'REP_BEST_ACCURACY_MEAN="{best_acc}"',
        f'ACC_CLOSE_THRESHOLD_USED="{acc_close_threshold}"',
        "",
    ]),
    encoding="utf-8",
)

print("=== aggregated candidates ===")
print(agg.sort_values(["accuracy_mean"], ascending=[False]).to_string(index=False))
print()
print(f"ACC close threshold used: {acc_close_threshold}")
print()
print("=== ACC-close candidates ===")
print(acc_candidates.sort_values(sort_cols, ascending=ascending).to_string(index=False))
print()
print("=== selected best ===")
print(best.to_string(index=False))
PY
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
main() {
  trap cleanup_children INT TERM

  resolve_stage_defaults
  resolve_confirm_defaults
  init_gpu_list

  SEARCH_ROOT="${OUTPUT_ROOT}/${STAGE}/search_stage"
  CONFIRM_ROOT="${OUTPUT_ROOT}/${STAGE}/final_confirm_stage"

  GLOBAL_SEARCH_SUMMARY="${SEARCH_ROOT}/global_search_summary.csv"
  GLOBAL_CONFIRM_SUMMARY="${CONFIRM_ROOT}/global_confirm_summary.csv"
  BEST_CONFIG_SUMMARY="${SEARCH_ROOT}/best_rep_config_summary.csv"
  BEST_CONFIG_ENV="${SEARCH_ROOT}/best_rep_config.env"

  mkdir -p "${SEARCH_ROOT}" "${CONFIRM_ROOT}"
  rm -f "${GLOBAL_SEARCH_SUMMARY}" "${GLOBAL_CONFIRM_SUMMARY}" "${BEST_CONFIG_SUMMARY}" "${BEST_CONFIG_ENV}"

  # 1) 构造搜索空间
  init_rep_search_space

  # 2) 搜索阶段
  run_rep_stage_with_tags \
    "${SEARCH_ROOT}" \
    "${GLOBAL_SEARCH_SUMMARY}" \
    "search_stage" \
    "REP_TOKENS_SEARCH" \
    "${DATASETS}" \
    "${SHOTS}" \
    "${SEEDS}" \
    "${REP_TAGS[@]}"

  print_summary_table "${GLOBAL_SEARCH_SUMMARY}" "rep_tokens search summary"

  # 3) 选最优 tag
  if [[ "${AUTO_SELECT_BEST}" == "1" ]]; then
    print_banner "Auto-select best rep_tokens config"
    select_best_tag_from_summary \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "${BEST_CONFIG_SUMMARY}" \
      "${BEST_CONFIG_ENV}" \
      "${DATASETS}" \
      "${SHOTS}"

    print_best_table "${BEST_CONFIG_SUMMARY}"
    echo
    echo "[best-env] ${BEST_CONFIG_ENV}"
  fi

  # 4) 最终确认：best tag × shots 1..16 × seeds 1..3
  if [[ "${AUTO_CONFIRM}" == "1" ]]; then
    if [[ ! -f "${BEST_CONFIG_ENV}" ]]; then
      echo "[error] missing best config env: ${BEST_CONFIG_ENV}" >&2
      exit 1
    fi

    # shellcheck disable=SC1090
    source "${BEST_CONFIG_ENV}"

    if [[ -z "${REP_BEST_TAG:-}" ]]; then
      echo "[error] REP_BEST_TAG is empty" >&2
      exit 1
    fi

    print_banner "Final confirm with best tag: ${REP_BEST_TAG}"
    run_rep_stage_with_tags \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "final_confirm_stage" \
      "REP_TOKENS_FINAL" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}" \
      "${REP_BEST_TAG}"

    print_summary_table "${GLOBAL_CONFIRM_SUMMARY}" "rep_tokens final confirm summary"
  fi

  if [[ "${FAILED_JOBS}" -ne 0 ]]; then
    echo
    echo "[warning] FAILED_JOBS=${FAILED_JOBS}"
    exit 1
  fi

  echo
  echo "Done."
}

main "$@"