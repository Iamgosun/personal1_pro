#!/bin/bash
set -euo pipefail

# ============================================================
# BayesMMRL rep_tokens-only sweep
#
# 改动目标：
# 1) 搜索阶段保持原逻辑；
# 2) 自动选择 best config 时，不再所有数据集共用同一参数，
#    也不只是每个 dataset 一个参数；
#    而是每个 dataset + sigma_mode 单独选自己的 best tag；
# 3) confirm 阶段遍历：dataset x sigma_mode x shot x seed；
# 4) B2N 阶段也遍历：dataset x sigma_mode x shot x seed：
#    - train_base/base: configs/protocols/b2n.yaml
#    - test_new/new  : configs/protocols/b2n_test_new.yaml + --eval-only --model-dir train_outdir
# 5) 支持每张 GPU 同时跑多个任务：JOBS_PER_GPU=2 表示一张卡两个并发任务。
#
# 用法示例：
#   cd MMRL
#   bash sweep.sh GPU_IDS="0 1 2 3" JOBS_PER_GPU=2
#
# 常用开关：
#   AUTO_CONFIRM=1
#   AUTO_CONFIRM_B2N=1
#   DELETE_CKPT_AFTER_TEST=1
#   JOBS_PER_GPU=2
#
# 说明：
#   请在 MMRL 根目录运行，或显式设置 PROJECT_DIR=/path/to/MMRL
# ============================================================

# ------------------------------------------------------------
# 支持 bash sweep.sh KEY=VALUE 形式传参
# ------------------------------------------------------------
apply_kv_args() {
  local arg key val
  for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
      key="${arg%%=*}"
      val="${arg#*=}"
      case "${key}" in
        PROJECT_DIR|ROOT|PROTOCOL|EXEC_MODE|BACKBONE|STAGE|DATASETS|SHOTS|SEEDS|CONFIRM_DATASETS|CONFIRM_SHOTS|CONFIRM_SEEDS|OUTPUT_ROOT|NGPU|GPU_IDS|JOBS_PER_GPU|SKIP_EXISTING|SLEEP_SEC|RUN_ZERO_PRIOR|RUN_CLIP_JOINT_PRIOR|DELETE_CKPT_AFTER_TEST|KEEP_CONFIRM_CKPT_SHOTS|AUTO_SELECT_BEST|AUTO_CONFIRM|AUTO_CONFIRM_B2N|ACC_CLOSE_THRESHOLD|PAPER_ALPHA|PAPER_REG_WEIGHT|PAPER_N_REP_TOKENS|PAPER_REP_LAYERS|PAPER_REP_DIM|COMMON_N_MC_TRAIN|COMMON_N_MC_TEST|COMMON_EVAL_MODE|COMMON_EVAL_AGGREGATION|COMMON_KL_WARMUP_EPOCHS|REP_SIGMA_MODES|ZERO_PRIOR_STDS|ZERO_KL_LIST|CLIP_PRIOR_STDS|CLIP_KL_LIST|CLIP_BLEND_LIST|CLIP_SCALE_LIST|MN_ENFORCE_TRACE|MN_LOWRANK_RANKS)
          printf -v "${key}" '%s' "${val}"
          export "${key}"
          ;;
        *)
          echo "[warn] unknown KEY=VALUE argument ignored: ${arg}" >&2
          ;;
      esac
    else
      echo "[warn] non KEY=VALUE argument ignored: ${arg}" >&2
    fi
  done
}

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

OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayes_mmrl_rep_tokens}

# GPU 调度
NGPU=${NGPU:-}
GPU_IDS=${GPU_IDS:-}

# 每张物理 GPU 同时跑几个任务。
# JOBS_PER_GPU=1：每张卡一个任务，保持原行为。
# JOBS_PER_GPU=2：每张卡两个并发任务。
JOBS_PER_GPU=${JOBS_PER_GPU:-2}

SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# 开关
RUN_ZERO_PRIOR=${RUN_ZERO_PRIOR:-1}
RUN_CLIP_JOINT_PRIOR=${RUN_CLIP_JOINT_PRIOR:-0}
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}
AUTO_SELECT_BEST=${AUTO_SELECT_BEST:-1}
AUTO_CONFIRM=${AUTO_CONFIRM:-1}
AUTO_CONFIRM_B2N=${AUTO_CONFIRM_B2N:-1}

# confirm 阶段需要保留 checkpoint 的 shots
# 默认保留 best config confirm 阶段的 16-shot 和 32-shot 权重
KEEP_CONFIRM_CKPT_SHOTS=${KEEP_CONFIRM_CKPT_SHOTS:-"16 32"}

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
COMMON_KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS:-6}

# ---------------------------
# rep_tokens 搜索空间
# ---------------------------
REP_SIGMA_MODES=${REP_SIGMA_MODES:-"global per_token diagonal matrix_normal_diag matrix_normal_diag_lowrank"}

# zero prior
ZERO_PRIOR_STDS=${ZERO_PRIOR_STDS:-"0.01 0.05 0.1 0.5 1.0"}
ZERO_KL_LIST=${ZERO_KL_LIST:-"1e-6 1e-5 1e-4 5e-4 1e-3 1e-2 5e-2"}

# clip_joint prior（默认关闭）
CLIP_PRIOR_STDS=${CLIP_PRIOR_STDS:-"0.05"}
CLIP_KL_LIST=${CLIP_KL_LIST:-"1e-4 5e-4 1e-3"}
CLIP_BLEND_LIST=${CLIP_BLEND_LIST:-"0.2 0.5"}
CLIP_SCALE_LIST=${CLIP_SCALE_LIST:-"0.02 0.05"}

# matrix normal 相关
MN_ENFORCE_TRACE=${MN_ENFORCE_TRACE:-True}
MN_LOWRANK_RANKS=${MN_LOWRANK_RANKS:-"2 4 8"}

apply_kv_args "$@"

METHOD_CFG="configs/methods/bayesmmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

SEARCH_ROOT=""
CONFIRM_ROOT=""
B2N_CONFIRM_ROOT=""

GLOBAL_SEARCH_SUMMARY=""
GLOBAL_CONFIRM_SUMMARY=""
GLOBAL_B2N_CONFIRM_SUMMARY=""

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

PHASE=""
SUBSAMPLE=""
PROTOCOL_CFG=""

set_protocol_context() {
  PROTOCOL="$1"
  read -r PHASE SUBSAMPLE PROTOCOL_CFG <<< "$(resolve_phase_semantics "${PROTOCOL}")"
}

set_b2n_test_new_context() {
  PROTOCOL="B2N"
  PHASE="test_new"
  SUBSAMPLE="new"
  PROTOCOL_CFG="configs/protocols/b2n_test_new.yaml"
}

set_protocol_context "${PROTOCOL}"

# ------------------------------------------------------------
# 第一阶段默认规模
# ------------------------------------------------------------
resolve_stage_defaults() {
  if [[ -z "${DATASETS}" ]]; then
    case "${STAGE}" in
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

  if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid JOBS_PER_GPU=${JOBS_PER_GPU}. It must be a positive integer." >&2
    exit 1
  fi

  # 先保存物理 GPU 列表。
  local -a PHYSICAL_GPU_LIST
  PHYSICAL_GPU_LIST=("${GPU_LIST[@]}")

  if [[ -z "${NGPU}" ]]; then
    NGPU=${#PHYSICAL_GPU_LIST[@]}
  fi

  if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="${PHYSICAL_GPU_LIST[*]}"
  fi

  # 将物理 GPU 扩展为调度 slot。
  # 例如 GPU_IDS="0 1", JOBS_PER_GPU=2
  # 则 GPU_LIST 变成：0 0 1 1
  # 后续调度器不需要修改，仍然按 GPU_LIST[$slot] 取卡号。
  GPU_LIST=()

  local gpu_id j
  for gpu_id in "${PHYSICAL_GPU_LIST[@]}"; do
    for ((j=0; j<JOBS_PER_GPU; j++)); do
      GPU_LIST+=("${gpu_id}")
    done
  done

  echo "[GPU] physical GPUs: ${GPU_IDS}"
  echo "[GPU] jobs per GPU: ${JOBS_PER_GPU}"
  echo "[GPU] slots: ${#GPU_LIST[@]}"
  echo "[GPU] slot mapping: ${GPU_LIST[*]}"
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

build_b2n_train_outdir() {
  local base_root=$1
  local dataset=$2
  local shot=$3
  local tag=$4
  local seed=$5
  local backbone_tag="${BACKBONE//\//-}"
  echo "${base_root}/BayesMMRL/B2N/train_base/${dataset}/shots_${shot}/${backbone_tag}/${tag}/seed${seed}"
}

build_b2n_test_new_outdir() {
  local base_root=$1
  local dataset=$2
  local shot=$3
  local tag=$4
  local seed=$5
  local backbone_tag="${BACKBONE//\//-}"
  echo "${base_root}/BayesMMRL/B2N/test_new/${dataset}/shots_${shot}/${backbone_tag}/${tag}/seed${seed}"
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
    echo "PHASE: ${PHASE}"
    echo "SUBSAMPLE: ${SUBSAMPLE}"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "DATA_ROOT: ${ROOT}"
    echo "OUTPUT_ROOT: ${base_root}"
    echo "BACKBONE: ${BACKBONE}"
    echo "PROTOCOL_CFG: ${PROTOCOL_CFG}"
    echo "JOBS_PER_GPU: ${JOBS_PER_GPU}"
    echo "============================================================"
  } >> "${logfile}"
}

write_b2n_new_eval_log_header() {
  local logfile=$1
  local gpu_id=$2
  local dataset=$3
  local shot=$4
  local seed=$5
  local tag=$6
  local model_dir=$7
  local base_root=$8

  {
    echo "============================================================"
    echo "START: $(date '+%F %T')"
    echo "STAGE: B2N test_new"
    echo "GPU: ${gpu_id}"
    echo "METHOD: BayesMMRL"
    echo "PROTOCOL: B2N"
    echo "PHASE: test_new"
    echo "SUBSAMPLE: new"
    echo "EXEC_MODE: ${EXEC_MODE}"
    echo "DATASET: ${dataset}"
    echo "SHOTS: ${shot}"
    echo "SEED: ${seed}"
    echo "TAG: ${tag}"
    echo "DATA_ROOT: ${ROOT}"
    echo "OUTPUT_ROOT: ${base_root}"
    echo "BACKBONE: ${BACKBONE}"
    echo "MODEL_DIR: ${model_dir}"
    echo "PROTOCOL_CFG: configs/protocols/b2n_test_new.yaml"
    echo "JOBS_PER_GPU: ${JOBS_PER_GPU}"
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

should_keep_confirm_checkpoint() {
  local stage_name=${1:-}
  local shot=${2:-}

  # 只保护 confirm 阶段的 best config 权重。
  # 注意：B2N train_base 默认仍会在 test_new 成功后按 DELETE_CKPT_AFTER_TEST 清理。
  if [[ "${stage_name}" != "confirm_best_per_dataset" && "${stage_name}" != "confirm_best_per_dataset_sigma" ]]; then
    return 1
  fi

  local keep_shot
  for keep_shot in ${KEEP_CONFIRM_CKPT_SHOTS}; do
    if [[ "${shot}" == "${keep_shot}" ]]; then
      return 0
    fi
  done

  return 1
}

cleanup_checkpoint_if_ready() {
  local outdir=$1
  local logfile=${2:-}
  local stage_name=${3:-}
  local shot=${4:-}

  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi

  if should_keep_confirm_checkpoint "${stage_name}" "${shot}"; then
    if [[ -n "${logfile}" ]]; then
      echo "[cleanup] kept checkpoint dir for confirm stage shot=${shot}: ${outdir}/refactor_model" >> "${logfile}"
    fi
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

b2n_train_checkpoint_ready() {
  local outdir=$1
  local ckpt_dir="${outdir}/refactor_model"

  [[ -f "${ckpt_dir}/checkpoint" ]] && compgen -G "${ckpt_dir}/model*.pth.tar*" > /dev/null
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
    mkdir -p "$(dirname "${global_summary}")"
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
    if ! python evaluation/result_parser.py "${case_root}" --split test >/dev/null 2>&1; then
      echo "[warn] result_parser failed: ${case_root}" >&2
    fi
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
    sort_cols = [c for c in ["stage_name", "scheme", "dataset", "shot", "accuracy_mean"] if c in df.columns]
    ascending = [False if c == "accuracy_mean" else True for c in sort_cols]
    print(df[keep].sort_values(by=sort_cols, ascending=ascending).to_string(index=False))
else:
    print(df.to_string(index=False))
PY
  else
    echo "[warn] summary not found."
  fi
}

print_best_table() {
  local summary_csv=$1
  echo
  echo "============================================================"
  echo "Auto-selected per-dataset + per-sigma best configs"
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
# 通用执行函数：搜索阶段
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
  local -a EXTRA_OPTS
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
            cleanup_checkpoint_if_ready "${outdir}" "" "${stage_name}" "${shot}"
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

            cleanup_checkpoint_if_ready "${outdir}" "${logfile}" "${stage_name}" "${shot}"
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
# 选最佳参数：每个 dataset + sigma_mode 单独选择 best tag。
#
# 规则：
# 1) 先看 ACC；
# 2) 若 ACC 与该 dataset/sigma_mode 最优只差 <= ACC_CLOSE_THRESHOLD，优先选 ECE 更低的；
# 3) 若没有 ECE，则按 ACC / F1 / NLL / Brier 兜底。
# ------------------------------------------------------------
select_best_tag_from_summary() {
  local summary_csv=$1
  local best_summary=$2
  local best_env=$3
  local datasets_str=$4
  local shots_str=$5
  local sigma_modes_str=${6:-"${REP_SIGMA_MODES}"}

  if [[ ! -f "${summary_csv}" ]]; then
    echo "[error] missing summary file: ${summary_csv}" >&2
    exit 1
  fi

  python - <<PY
import re
import pandas as pd
from pathlib import Path

summary_csv = Path(r"${summary_csv}")
best_summary = Path(r"${best_summary}")
best_env = Path(r"${best_env}")

datasets_order = """${datasets_str}""".split()
datasets = set(datasets_order)
shots = set("""${shots_str}""".split())
sigma_modes_order = """${sigma_modes_str}""".split()
sigma_modes = set(sigma_modes_order)
acc_close_threshold = float("""${ACC_CLOSE_THRESHOLD}""".strip())

df = pd.read_csv(summary_csv)
required_cols = {"scheme", "dataset", "shot", "tag", "accuracy_mean"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"summary missing columns: {missing}")

df["dataset"] = df["dataset"].astype(str)
df["shot"] = df["shot"].astype(str)
df["tag"] = df["tag"].astype(str)

def parse_sigma_mode(tag: str) -> str:
    # 支持：
    # rep_zero_sig-global_pstd-...
    # rep_zero_sig-matrix_normal_diag_lowrank_pstd-...
    # rep_clip_sig-diagonal_pstd-...
    m = re.search(r"_sig-(.+?)_pstd-", tag)
    if not m:
        raise ValueError(f"cannot parse sigma_mode from tag: {tag}")
    return m.group(1)

df["sigma_mode"] = df["tag"].map(parse_sigma_mode)

df = df[
    df["dataset"].isin(datasets)
    & df["shot"].isin(shots)
    & df["sigma_mode"].isin(sigma_modes)
]

if df.empty:
    raise SystemExit("no rows left after dataset/shot/sigma_mode filtering")

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
    df.groupby(["dataset", "sigma_mode", "scheme", "tag"], as_index=False)
      .agg(**agg_dict)
)

best_rows = []

for dataset in datasets_order:
    for sigma_mode in sigma_modes_order:
        sub = agg[(agg["dataset"] == dataset) & (agg["sigma_mode"] == sigma_mode)].copy()
        if sub.empty:
            raise SystemExit(f"no candidate rows for dataset={dataset}, sigma_mode={sigma_mode}")

        max_acc = sub["accuracy_mean"].max()
        candidates = sub[sub["accuracy_mean"] >= max_acc - acc_close_threshold].copy()

        sort_cols = []
        ascending = []

        if "ece_mean" in candidates.columns:
            sort_cols.append("ece_mean")
            ascending.append(True)

        sort_cols.append("accuracy_mean")
        ascending.append(False)

        if "macro_f1_mean" in candidates.columns:
            sort_cols.append("macro_f1_mean")
            ascending.append(False)

        if "nll_mean" in candidates.columns:
            sort_cols.append("nll_mean")
            ascending.append(True)

        if "brier_mean" in candidates.columns:
            sort_cols.append("brier_mean")
            ascending.append(True)

        sort_cols.append("tag")
        ascending.append(True)

        best = candidates.sort_values(sort_cols, ascending=ascending).head(1).copy()
        best_rows.append(best)

best_df = pd.concat(best_rows, ignore_index=True)

best_summary.parent.mkdir(parents=True, exist_ok=True)
best_df.to_csv(best_summary, index=False)

def bash_quote(s):
    s = str(s)
    return "'" + s.replace("'", "'\"'\"'") + "'"

best_env.parent.mkdir(parents=True, exist_ok=True)

lines = []
lines.append("declare -ga REP_BEST_DATASETS")
lines.append("REP_BEST_DATASETS=(" + " ".join(bash_quote(x) for x in datasets_order) + ")")
lines.append("declare -ga REP_BEST_SIGMA_MODES")
lines.append("REP_BEST_SIGMA_MODES=(" + " ".join(bash_quote(x) for x in sigma_modes_order) + ")")
lines.append("declare -gA REP_BEST_TAG_BY_DATASET_SIGMA")
lines.append("declare -gA REP_BEST_SCHEME_BY_DATASET_SIGMA")
lines.append("declare -gA REP_BEST_ACCURACY_BY_DATASET_SIGMA")

for _, row in best_df.iterrows():
    dataset = str(row["dataset"])
    sigma_mode = str(row["sigma_mode"])
    key = f"{dataset}|{sigma_mode}"
    tag = str(row["tag"])
    scheme = str(row["scheme"])
    acc = str(row["accuracy_mean"])

    lines.append(f"REP_BEST_TAG_BY_DATASET_SIGMA[{bash_quote(key)}]={bash_quote(tag)}")
    lines.append(f"REP_BEST_SCHEME_BY_DATASET_SIGMA[{bash_quote(key)}]={bash_quote(scheme)}")
    lines.append(f"REP_BEST_ACCURACY_BY_DATASET_SIGMA[{bash_quote(key)}]={bash_quote(acc)}")

lines.append(f"ACC_CLOSE_THRESHOLD_USED={bash_quote(acc_close_threshold)}")
lines.append("")

best_env.write_text("\n".join(lines), encoding="utf-8")

print("=== aggregated candidates by dataset + sigma_mode ===")
print(
    agg.sort_values(
        ["dataset", "sigma_mode", "accuracy_mean"],
        ascending=[True, True, False],
    ).to_string(index=False)
)
print()
print(f"ACC close threshold used: {acc_close_threshold}")
print()
print("=== selected best by dataset + sigma_mode ===")
print(best_df.to_string(index=False))
PY
}

load_best_tags_by_dataset_sigma() {
  local best_env=$1
  local datasets_str=$2
  local sigma_modes_str=$3

  if [[ ! -f "${best_env}" ]]; then
    echo "[error] missing best env file: ${best_env}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${best_env}"

  local dataset sigma_mode key tag

  for dataset in ${datasets_str}; do
    for sigma_mode in ${sigma_modes_str}; do
      key="${dataset}|${sigma_mode}"
      tag="${REP_BEST_TAG_BY_DATASET_SIGMA[$key]:-}"

      if [[ -z "${tag}" ]]; then
        echo "[error] no selected best tag for dataset=${dataset}, sigma_mode=${sigma_mode}" >&2
        exit 1
      fi

      if [[ -z "${REP_OPTS[${tag}]:-}" ]]; then
        echo "[error] selected tag not found in REP_OPTS: dataset=${dataset}, sigma_mode=${sigma_mode}, tag=${tag}" >&2
        exit 1
      fi
    done
  done
}

# ------------------------------------------------------------
# Confirm：按 dataset + sigma_mode 对应 best tag 跑
# ------------------------------------------------------------
run_rep_stage_with_dataset_sigma_best_tags() {
  local base_root=$1
  local global_summary=$2
  local stage_name=$3
  local scheme_name=$4
  local datasets_str=$5
  local sigma_modes_str=$6
  local shots_str=$7
  local seeds_str=$8

  print_banner "Start ${stage_name}: ${scheme_name} using per-dataset + per-sigma best tags"

  local dataset sigma_mode key tag

  for dataset in ${datasets_str}; do
    for sigma_mode in ${sigma_modes_str}; do
      key="${dataset}|${sigma_mode}"
      tag="${REP_BEST_TAG_BY_DATASET_SIGMA[$key]:-}"

      if [[ -z "${tag}" ]]; then
        echo "[error] no best tag for dataset=${dataset}, sigma_mode=${sigma_mode}" >&2
        exit 1
      fi

      if [[ -z "${REP_OPTS[${tag}]:-}" ]]; then
        echo "[error] selected tag not found in REP_OPTS: dataset=${dataset}, sigma_mode=${sigma_mode}, tag=${tag}" >&2
        exit 1
      fi

      echo "[best] dataset=${dataset} sigma_mode=${sigma_mode} tag=${tag} acc=${REP_BEST_ACCURACY_BY_DATASET_SIGMA[$key]:-NA}"
    done
  done

  init_slots

  local shot seed slot gpu_id outdir logfile desc opts_str
  local -a EXTRA_OPTS

  # 调度维度：shot x seed x dataset x sigma_mode
  # 最终等价于 dataset x sigma_mode x shot x seed 全覆盖。
  for shot in ${shots_str}; do
    for seed in ${seeds_str}; do
      for dataset in ${datasets_str}; do
        for sigma_mode in ${sigma_modes_str}; do
          key="${dataset}|${sigma_mode}"
          tag="${REP_BEST_TAG_BY_DATASET_SIGMA[$key]}"
          opts_str="${REP_OPTS[$tag]}"
          read -r -a EXTRA_OPTS <<< "${opts_str}"

          outdir="$(build_outdir "${base_root}" "BayesMMRL" "${dataset}" "${shot}" "${tag}" "${seed}")"
          logfile="${outdir}/run.log"

          if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
            mkdir -p "${outdir}"
            cleanup_checkpoint_if_ready "${outdir}" "" "${stage_name}" "${shot}"
            echo "[skip] ${scheme_name} dataset=${dataset} sigma_mode=${sigma_mode} shot=${shot} tag=${tag} seed=${seed}"
            continue
          fi

          wait_for_any_slot
          slot="${READY_SLOT}"
          gpu_id="${GPU_LIST[$slot]}"
          desc="${scheme_name} | dataset=${dataset} | sigma_mode=${sigma_mode} | shot=${shot} | seed=${seed} | tag=${tag}"

          (
            mkdir -p "${outdir}"
            : > "${logfile}"

            write_log_header \
              "${logfile}" \
              "${gpu_id}" \
              "BayesMMRL" \
              "${dataset}" \
              "${shot}" \
              "${seed}" \
              "${tag}" \
              "${stage_name}" \
              "${base_root}"

            cleanup_broken_resume_if_needed "${outdir}" "${logfile}"

            echo "[run] ${desc}" >> "${logfile}"
            echo "[info] rep_tokens-only confirm stage with per-dataset + per-sigma best tag" >> "${logfile}"
            echo "[info] dataset-specific sigma best tag: dataset=${dataset}, sigma_mode=${sigma_mode}, tag=${tag}" >> "${logfile}"
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

            cleanup_checkpoint_if_ready "${outdir}" "${logfile}" "${stage_name}" "${shot}"
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

  local dataset2 sigma_mode2 shot2 tag2 key2
  for dataset2 in ${datasets_str}; do
    for sigma_mode2 in ${sigma_modes_str}; do
      key2="${dataset2}|${sigma_mode2}"
      tag2="${REP_BEST_TAG_BY_DATASET_SIGMA[$key2]}"

      for shot2 in ${shots_str}; do
        summarize_one_tag \
          "${base_root}" \
          "${global_summary}" \
          "BayesMMRL" \
          "${stage_name}" \
          "${scheme_name}_${sigma_mode2}" \
          "${dataset2}" \
          "${shot2}" \
          "${tag2}"
      done
    done
  done
}

# ------------------------------------------------------------
# B2N：按 dataset + sigma_mode 对应 best tag 跑 train_base + test_new
# ------------------------------------------------------------
run_b2n_one_case_with_tag() {
  local gpu_id=$1
  local base_root=$2
  local dataset=$3
  local shot=$4
  local seed=$5
  local tag=$6
  local sigma_mode=${7:-}

  local opts_str
  local -a EXTRA_OPTS

  opts_str="${REP_OPTS[${tag}]}"
  read -r -a EXTRA_OPTS <<< "${opts_str}"

  local train_outdir train_log eval_outdir eval_log
  train_outdir="$(build_b2n_train_outdir "${base_root}" "${dataset}" "${shot}" "${tag}" "${seed}")"
  eval_outdir="$(build_b2n_test_new_outdir "${base_root}" "${dataset}" "${shot}" "${tag}" "${seed}")"
  train_log="${train_outdir}/run.log"
  eval_log="${eval_outdir}/run.log"

  mkdir -p "${train_outdir}" "${eval_outdir}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    echo "[skip] b2n_confirm dataset=${dataset} sigma_mode=${sigma_mode:-NA} shot=${shot} tag=${tag} seed=${seed}"

    if [[ "${DELETE_CKPT_AFTER_TEST}" == "1" ]]; then
      cleanup_checkpoint_if_ready "${train_outdir}" "${train_log}" "b2n_confirm_train_base" "${shot}"
      cleanup_checkpoint_if_ready "${eval_outdir}" "${eval_log}" "b2n_confirm_test_new" "${shot}"
    fi

    return 0
  fi

  local need_train=1
  if [[ -f "${train_outdir}/test_metrics.json" ]] && b2n_train_checkpoint_ready "${train_outdir}"; then
    need_train=0
  fi

  if [[ "${need_train}" == "1" ]]; then
    set_protocol_context B2N

    : > "${train_log}"

    write_log_header \
      "${train_log}" \
      "${gpu_id}" \
      "BayesMMRL" \
      "${dataset}" \
      "${shot}" \
      "${seed}" \
      "${tag}" \
      "b2n_confirm_train_base" \
      "${base_root}"

    cleanup_broken_resume_if_needed "${train_outdir}" "${train_log}"

    {
      echo "[run] B2N train_base | dataset=${dataset} | sigma_mode=${sigma_mode:-NA} | shot=${shot} | seed=${seed} | tag=${tag}"
      echo "[info] B2N train_base uses configs/protocols/b2n.yaml and DATASET.SUBSAMPLE_CLASSES=base"
      echo "[info] sigma_mode=${sigma_mode:-NA}"
      echo "[info] train_outdir=${train_outdir}"
      echo "[info] eval_outdir=${eval_outdir}"
      echo "[info] EVAL_MODE=${COMMON_EVAL_MODE}, EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION}"
      echo "[info] KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS}"
    } >> "${train_log}"

    (
      cd "${PROJECT_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${ROOT}" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --method-config-file "${METHOD_CFG}" \
        --protocol-config-file "configs/protocols/b2n.yaml" \
        --runtime-config-file "${RUNTIME_CFG}" \
        --output-dir "${train_outdir}" \
        --method BayesMMRL \
        --protocol B2N \
        --exec-mode "${EXEC_MODE}" \
        --seed "${seed}" \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES base \
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
    ) >> "${train_log}" 2>&1
  else
    touch "${train_log}"
    {
      echo "============================================================"
      echo "SKIP_B2N_TRAIN_BASE: existing metrics and checkpoint found"
      echo "TIME: $(date '+%F %T')"
      echo "DATASET: ${dataset}"
      echo "SIGMA_MODE: ${sigma_mode:-NA}"
      echo "SHOT: ${shot}"
      echo "SEED: ${seed}"
      echo "TAG: ${tag}"
      echo "TRAIN_OUTDIR: ${train_outdir}"
      echo "EVAL_OUTDIR: ${eval_outdir}"
      echo "============================================================"
    } >> "${train_log}"
  fi

  if ! b2n_train_checkpoint_ready "${train_outdir}"; then
    {
      echo
      echo "============================================================"
      echo "ERROR: B2N train checkpoint missing; cannot run test_new"
      echo "DATASET: ${dataset}"
      echo "SIGMA_MODE: ${sigma_mode:-NA}"
      echo "SHOT: ${shot}"
      echo "SEED: ${seed}"
      echo "TAG: ${tag}"
      echo "TRAIN_OUTDIR: ${train_outdir}"
      echo "EXPECTED: ${train_outdir}/refactor_model/checkpoint and model*.pth.tar*"
      echo "============================================================"
    } >> "${train_log}"

    return 1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    echo "[skip] B2N test_new existing dataset=${dataset} sigma_mode=${sigma_mode:-NA} shot=${shot} tag=${tag} seed=${seed}"
  else
    : > "${eval_log}"

    write_b2n_new_eval_log_header \
      "${eval_log}" \
      "${gpu_id}" \
      "${dataset}" \
      "${shot}" \
      "${seed}" \
      "${tag}" \
      "${train_outdir}" \
      "${base_root}"

    {
      echo "[run] B2N test_new | dataset=${dataset} | sigma_mode=${sigma_mode:-NA} | shot=${shot} | seed=${seed} | tag=${tag}"
      echo "[info] B2N test_new uses configs/protocols/b2n_test_new.yaml, --eval-only, and --model-dir=${train_outdir}"
      echo "[info] sigma_mode=${sigma_mode:-NA}"
      echo "[info] train_outdir=${train_outdir}"
      echo "[info] eval_outdir=${eval_outdir}"
      echo "[info] EVAL_MODE=${COMMON_EVAL_MODE}, EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION}"
      echo "[info] KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS}"
    } >> "${eval_log}"

    (
      cd "${PROJECT_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
        --root "${ROOT}" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --method-config-file "${METHOD_CFG}" \
        --protocol-config-file "configs/protocols/b2n_test_new.yaml" \
        --runtime-config-file "${RUNTIME_CFG}" \
        --output-dir "${eval_outdir}" \
        --model-dir "${train_outdir}" \
        --method BayesMMRL \
        --protocol B2N \
        --exec-mode "${EXEC_MODE}" \
        --seed "${seed}" \
        --eval-only \
        DATASET.NUM_SHOTS "${shot}" \
        DATASET.SUBSAMPLE_CLASSES new \
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
    ) >> "${eval_log}" 2>&1
  fi

  if [[ "${DELETE_CKPT_AFTER_TEST}" == "1" && -f "${eval_outdir}/test_metrics.json" ]]; then
    cleanup_checkpoint_if_ready "${train_outdir}" "${train_log}" "b2n_confirm_train_base" "${shot}"
    cleanup_checkpoint_if_ready "${eval_outdir}" "${eval_log}" "b2n_confirm_test_new" "${shot}"
  fi
}

run_b2n_with_dataset_sigma_best_tags() {
  local base_root=$1
  local global_summary=$2
  local datasets_str=$3
  local sigma_modes_str=$4
  local shots_str=$5
  local seeds_str=$6

  print_banner "Start b2n_confirm: per-dataset + per-sigma best tags"

  local dataset sigma_mode key tag

  # 先完整检查，避免跑到一半才发现某个 dataset/sigma 没有 best tag
  for dataset in ${datasets_str}; do
    for sigma_mode in ${sigma_modes_str}; do
      key="${dataset}|${sigma_mode}"
      tag="${REP_BEST_TAG_BY_DATASET_SIGMA[$key]:-}"

      if [[ -z "${tag}" ]]; then
        echo "[error] no best tag for dataset=${dataset}, sigma_mode=${sigma_mode}" >&2
        exit 1
      fi

      if [[ -z "${REP_OPTS[${tag}]:-}" ]]; then
        echo "[error] selected tag not found in REP_OPTS: dataset=${dataset}, sigma_mode=${sigma_mode}, tag=${tag}" >&2
        exit 1
      fi

      echo "[best-b2n] dataset=${dataset} sigma_mode=${sigma_mode} tag=${tag} acc=${REP_BEST_ACCURACY_BY_DATASET_SIGMA[$key]:-NA}"
    done
  done

  init_slots

  local shot seed slot gpu_id desc logfile train_log eval_log

  # 调度维度：shot x seed x dataset x sigma_mode
  # 最终等价于 dataset x sigma_mode x shot x seed 全覆盖。
  for shot in ${shots_str}; do
    for seed in ${seeds_str}; do
      for dataset in ${datasets_str}; do
        for sigma_mode in ${sigma_modes_str}; do
          key="${dataset}|${sigma_mode}"
          tag="${REP_BEST_TAG_BY_DATASET_SIGMA[$key]}"

          wait_for_any_slot
          slot="${READY_SLOT}"
          gpu_id="${GPU_LIST[$slot]}"

          train_log="$(build_b2n_train_outdir "${base_root}" "${dataset}" "${shot}" "${tag}" "${seed}")/run.log"
          eval_log="$(build_b2n_test_new_outdir "${base_root}" "${dataset}" "${shot}" "${tag}" "${seed}")/run.log"
          logfile="${train_log}"

          desc="b2n_confirm | dataset=${dataset} | sigma_mode=${sigma_mode} | shot=${shot} | seed=${seed} | tag=${tag} | eval_log=${eval_log}"

          (
            run_b2n_one_case_with_tag \
              "${gpu_id}" \
              "${base_root}" \
              "${dataset}" \
              "${shot}" \
              "${seed}" \
              "${tag}" \
              "${sigma_mode}"
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

  local dataset2 sigma_mode2 shot2 tag2 key2

  # 汇总 B2N train_base/base
  set_protocol_context B2N

  for dataset2 in ${datasets_str}; do
    for sigma_mode2 in ${sigma_modes_str}; do
      key2="${dataset2}|${sigma_mode2}"
      tag2="${REP_BEST_TAG_BY_DATASET_SIGMA[$key2]}"

      for shot2 in ${shots_str}; do
        summarize_one_tag \
          "${base_root}" \
          "${global_summary}" \
          "BayesMMRL" \
          "b2n_confirm_train_base" \
          "rep_tokens_best_per_dataset_sigma_${sigma_mode2}" \
          "${dataset2}" \
          "${shot2}" \
          "${tag2}"
      done
    done
  done

  # 汇总 B2N test_new/new
  set_b2n_test_new_context

  for dataset2 in ${datasets_str}; do
    for sigma_mode2 in ${sigma_modes_str}; do
      key2="${dataset2}|${sigma_mode2}"
      tag2="${REP_BEST_TAG_BY_DATASET_SIGMA[$key2]}"

      for shot2 in ${shots_str}; do
        summarize_one_tag \
          "${base_root}" \
          "${global_summary}" \
          "BayesMMRL" \
          "b2n_confirm_test_new" \
          "rep_tokens_best_per_dataset_sigma_${sigma_mode2}" \
          "${dataset2}" \
          "${shot2}" \
          "${tag2}"
      done
    done
  done
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
main() {
  trap cleanup_children INT TERM

  local ORIGINAL_PROTOCOL
  ORIGINAL_PROTOCOL="${PROTOCOL}"

  set_protocol_context "${ORIGINAL_PROTOCOL}"

  resolve_stage_defaults
  resolve_confirm_defaults
  init_gpu_list

  SEARCH_ROOT="${OUTPUT_ROOT}/${STAGE}/search_stage"
  CONFIRM_ROOT="${OUTPUT_ROOT}/${STAGE}/confirm_stage"
  B2N_CONFIRM_ROOT="${OUTPUT_ROOT}/${STAGE}/b2n_confirm_stage"

  GLOBAL_SEARCH_SUMMARY="${SEARCH_ROOT}/global_test_summary.csv"
  GLOBAL_CONFIRM_SUMMARY="${CONFIRM_ROOT}/global_test_summary.csv"
  GLOBAL_B2N_CONFIRM_SUMMARY="${B2N_CONFIRM_ROOT}/global_test_summary.csv"

  BEST_CONFIG_SUMMARY="${OUTPUT_ROOT}/${STAGE}/best_config_per_dataset_sigma.csv"
  BEST_CONFIG_ENV="${OUTPUT_ROOT}/${STAGE}/best_config_per_dataset_sigma.env"

  print_banner "BayesMMRL rep_tokens-only sweep"
  echo "[project] ${PROJECT_DIR}"
  echo "[root] ${ROOT}"
  echo "[protocol] ${PROTOCOL}"
  echo "[phase] ${PHASE}"
  echo "[subsample] ${SUBSAMPLE}"
  echo "[stage] ${STAGE}"
  echo "[datasets] ${DATASETS}"
  echo "[shots] ${SHOTS}"
  echo "[seeds] ${SEEDS}"
  echo "[confirm datasets] ${CONFIRM_DATASETS}"
  echo "[confirm shots] ${CONFIRM_SHOTS}"
  echo "[confirm seeds] ${CONFIRM_SEEDS}"
  echo "[sigma modes] ${REP_SIGMA_MODES}"
  echo "[output root] ${OUTPUT_ROOT}"
  echo "[jobs per GPU] ${JOBS_PER_GPU}"
  echo "[auto select best] ${AUTO_SELECT_BEST}"
  echo "[auto confirm] ${AUTO_CONFIRM}"
  echo "[auto confirm b2n] ${AUTO_CONFIRM_B2N}"

  init_rep_search_space

  rm -f "${GLOBAL_SEARCH_SUMMARY}"

  run_rep_stage_with_tags \
    "${SEARCH_ROOT}" \
    "${GLOBAL_SEARCH_SUMMARY}" \
    "search" \
    "rep_tokens" \
    "${DATASETS}" \
    "${SHOTS}" \
    "${SEEDS}" \
    "${REP_TAGS[@]}"

  print_summary_table "${GLOBAL_SEARCH_SUMMARY}" "Search summary"

  if [[ "${AUTO_SELECT_BEST}" == "1" ]]; then
    select_best_tag_from_summary \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "${BEST_CONFIG_SUMMARY}" \
      "${BEST_CONFIG_ENV}" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${REP_SIGMA_MODES}"

    print_best_table "${BEST_CONFIG_SUMMARY}"
  else
    echo "[info] AUTO_SELECT_BEST=0, stop after search."
  fi

  if [[ "${AUTO_SELECT_BEST}" == "1" && "${AUTO_CONFIRM}" == "1" ]]; then
    load_best_tags_by_dataset_sigma \
      "${BEST_CONFIG_ENV}" \
      "${CONFIRM_DATASETS}" \
      "${REP_SIGMA_MODES}"

    set_protocol_context "${ORIGINAL_PROTOCOL}"

    rm -f "${GLOBAL_CONFIRM_SUMMARY}"

    run_rep_stage_with_dataset_sigma_best_tags \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm_best_per_dataset_sigma" \
      "rep_tokens_best_per_dataset_sigma" \
      "${CONFIRM_DATASETS}" \
      "${REP_SIGMA_MODES}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"

    print_summary_table "${GLOBAL_CONFIRM_SUMMARY}" "Confirm summary using per-dataset + per-sigma best tags"
  fi

  if [[ "${AUTO_SELECT_BEST}" == "1" && "${AUTO_CONFIRM_B2N}" == "1" ]]; then
    load_best_tags_by_dataset_sigma \
      "${BEST_CONFIG_ENV}" \
      "${CONFIRM_DATASETS}" \
      "${REP_SIGMA_MODES}"

    rm -f "${GLOBAL_B2N_CONFIRM_SUMMARY}"

    run_b2n_with_dataset_sigma_best_tags \
      "${B2N_CONFIRM_ROOT}" \
      "${GLOBAL_B2N_CONFIRM_SUMMARY}" \
      "${CONFIRM_DATASETS}" \
      "${REP_SIGMA_MODES}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"

    print_summary_table "${GLOBAL_B2N_CONFIRM_SUMMARY}" "B2N confirm summary using per-dataset + per-sigma best tags"
  fi

  set_protocol_context "${ORIGINAL_PROTOCOL}"

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo "[DONE] finished with ${FAILED_JOBS} failed job(s)."
    exit 1
  fi

  echo "[DONE] all jobs finished successfully."
}

main "$@"
