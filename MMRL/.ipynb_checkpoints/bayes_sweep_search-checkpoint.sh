#!/bin/bash
set -euo pipefail

# ============================================================
# BayesMMRL 两阶段搜索 + 自动选优 + 自动确认脚本
#
# 设计目标：
# 1. 第一阶段：只在 coarse shots = 1/4/16 上做搜索
# 2. 自动执行顺序：baseline -> C -> B -> A
# 3. 第一阶段结束后，自动从 A/B/C 各选一个 best tag
#    选优规则：以 accuracy_mean 为主，其它指标仅做 tie-break
# 4. 第二阶段：自动跑 baseline + A_best + B_best + C_best
# 5. 第二阶段 shots：1/2/4/8/16/32
# 6. 第二阶段 seeds：1/2/3
# 7. 输出独立写到 output_sweeps，不污染 output_refactor
# 8. 训练时正常保存 checkpoint；测试完成后可自动删除
# 9. 自动清理残缺的 resume 状态，避免 FileNotFoundError: refactor_model/checkpoint
# ============================================================

# ---------------------------
# 用户可改环境变量
# ---------------------------
ROOT=${ROOT:-DATASETS}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}

# 搜索阶段数据集规模：
#   coarse3 -> caltech101 / oxford_pets / ucf101
#   full11  -> 11 datasets
STAGE=${STAGE:-coarse3}

# 第一阶段：如果手动指定 DATASETS / SHOTS / SEEDS，则覆盖默认
DATASETS=${DATASETS:-}
SHOTS=${SHOTS:-}
SEEDS=${SEEDS:-}

# 第二阶段：如果不指定，则默认沿用第一阶段数据集，
# shots 使用完整 1/2/4/8/16/32，seeds 使用 1/2/3
CONFIRM_DATASETS=${CONFIRM_DATASETS:-}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-}

# 输出目录
OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayes_mmrl_search_seq_v3_two_stage}

# GPU 调度
NGPU=${NGPU:-3}
GPU_IDS=${GPU_IDS:-"0 1 2"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# 开关
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_SCHEME_C=${RUN_SCHEME_C:-1}
RUN_SCHEME_B=${RUN_SCHEME_B:-1}
RUN_SCHEME_A=${RUN_SCHEME_A:-1}

AUTO_SELECT_BEST=${AUTO_SELECT_BEST:-1}
AUTO_CONFIRM=${AUTO_CONFIRM:-1}

# 训练完成并测试完成后，是否自动删除该 case 的 checkpoint 目录
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}

# ---------------------------
# 固定使用 MMRL 论文主体超参
# 不再当作 sweep 维度搜索
# ---------------------------
PAPER_ALPHA=0.7
PAPER_REG_WEIGHT=0.5
PAPER_N_REP_TOKENS=5
PAPER_REP_LAYERS='[6,7,8,9,10,11,12]'
PAPER_REP_DIM=512

# Bayes 固定 MC 设置
COMMON_N_MC_TRAIN=3
COMMON_N_MC_TEST=10
COMMON_EVAL_MODE=mc_predictive

METHOD_CFG="configs/methods/bayesmmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

# ---------------------------
# 路径
# ---------------------------
SEARCH_ROOT=""
CONFIRM_ROOT=""
GLOBAL_SEARCH_SUMMARY=""
GLOBAL_CONFIRM_SUMMARY=""
BEST_CONFIG_SUMMARY=""
BEST_CONFIG_ENV=""

# ------------------------------------------------------------
# 根据协议解析 phase / subsample / protocol config
# ------------------------------------------------------------
resolve_phase_semantics() {
  case "$1" in
    B2N) echo "train_base base configs/protocols/b2n.yaml" ;;
    FS)  echo "fewshot_train all configs/protocols/fs.yaml" ;;
    CD)  echo "cross_train all configs/protocols/cd.yaml" ;;
    *)
      echo "未知 PROTOCOL=$1" >&2
      exit 1
      ;;
  esac
}

read -r PHASE SUBSAMPLE PROTOCOL_CFG <<< "$(resolve_phase_semantics "$PROTOCOL")"

# ------------------------------------------------------------
# 根据阶段自动给出第一阶段默认数据集 / shots / seeds
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
        echo "未知 STAGE=${STAGE}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ -z "${SHOTS}" ]]; then
    SHOTS="1 4 16"
  fi

  if [[ -z "${SEEDS}" ]]; then
    SEEDS="1 2"
  fi
}

# ------------------------------------------------------------
# 第二阶段默认设置
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
# GPU 列表初始化
# ------------------------------------------------------------
init_gpu_list() {
  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a GPU_LIST <<< "${GPU_IDS}"
  else
    GPU_LIST=()
    local i
    for ((i=0; i<NGPU; i++)); do
      GPU_LIST+=("$i")
    done
  fi

  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "没有可用 GPU，请设置 NGPU 或 GPU_IDS" >&2
    exit 1
  fi
}

# ------------------------------------------------------------
# 字符串清洗：用于目录命名
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

# ------------------------------------------------------------
# 输出目录生成
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 日志头
# ------------------------------------------------------------
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
    echo "开始时间: $(date '+%F %T')"
    echo "当前阶段: ${stage_name}"
    echo "GPU: ${gpu_id}"
    echo "方法: ${method}"
    echo "协议: ${PROTOCOL}"
    echo "执行模式: ${EXEC_MODE}"
    echo "数据集: ${dataset}"
    echo "shots: ${shot}"
    echo "seed: ${seed}"
    echo "标签: ${tag}"
    echo "数据根目录: ${ROOT}"
    echo "阶段输出根目录: ${base_root}"
    echo "Backbone: ${BACKBONE}"
    echo "============================================================"
  } >> "${logfile}"
}

# ------------------------------------------------------------
# 状态提示
# ------------------------------------------------------------
print_banner() {
  local msg="$1"
  echo
  echo "################################################################"
  echo "# ${msg}"
  echo "################################################################"
}

# ------------------------------------------------------------
# checkpoint 清理 / resume 修复
# ------------------------------------------------------------
cleanup_checkpoint_if_ready() {
  local outdir=$1
  local logfile=${2:-}

  if [[ "${DELETE_CKPT_AFTER_TEST}" != "1" ]]; then
    return 0
  fi

  if [[ -f "${outdir}/test_metrics.json" ]]; then
    if [[ -d "${outdir}/refactor_model" ]]; then
      rm -rf "${outdir}/refactor_model"
      if [[ -n "${logfile}" ]]; then
        echo "[清理] 测试已完成，已删除 checkpoint 目录: ${outdir}/refactor_model" >> "${logfile}"
      else
        echo "[清理] 测试已完成，已删除 checkpoint 目录: ${outdir}/refactor_model"
      fi
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
      echo "[清理] 发现残缺 resume 状态，已删除: ${ckpt_dir}" >> "${logfile}"
    else
      echo "[清理] 发现残缺 resume 状态，已删除: ${ckpt_dir}"
    fi
  fi
}

# ------------------------------------------------------------
# 汇总表
# ------------------------------------------------------------
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

  python evaluation/result_parser.py "${case_root}" --split test >/dev/null 2>&1 || true

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
    "stage_name",
    "scheme",
    "dataset",
    "shot",
    "tag",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "ece_mean",
    "brier_mean",
    "nll_mean",
]
keep = [c for c in preferred_cols if c in df.columns]
if keep:
    sort_cols = [c for c in ["stage_name", "scheme", "dataset", "shot", "accuracy_mean"] if c in df.columns]
    ascending = []
    for c in sort_cols:
        if c == "accuracy_mean":
            ascending.append(False)
        else:
            ascending.append(True)
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
  echo "自动选优结果"
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
          echo "[完成] ${SLOT_DESC[$idx]}"
        else
          echo "[失败] ${SLOT_DESC[$idx]}  日志: ${SLOT_LOG[$idx]}" >&2
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
        echo "[完成] ${SLOT_DESC[$idx]}"
      else
        echo "[失败] ${SLOT_DESC[$idx]}  日志: ${SLOT_LOG[$idx]}" >&2
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

# ------------------------------------------------------------
# baseline：固定论文配置的 MMRL
# ------------------------------------------------------------
run_baseline_stage() {
  local base_root=$1
  local global_summary=$2
  local stage_name=$3
  local scheme_name=$4
  local tag_name=$5
  local datasets_str=$6
  local shots_str=$7
  local seeds_str=$8

  if [[ "${RUN_BASELINE}" != "1" ]]; then
    echo "[跳过] baseline 阶段"
    return 0
  fi

  print_banner "开始执行 ${stage_name}：固定论文配置的 MMRL baseline"

  init_slots

  local dataset shot seed slot gpu_id outdir logfile statusfile desc
  for dataset in ${datasets_str}; do
    for shot in ${shots_str}; do
      for seed in ${seeds_str}; do
        outdir="$(build_outdir "${base_root}" "MMRL" "${dataset}" "${shot}" "${tag_name}" "${seed}")"
        logfile="${outdir}/run.log"
        statusfile="${outdir}/job_status.txt"

        if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
          mkdir -p "${outdir}"
          cleanup_checkpoint_if_ready "${outdir}"
          echo "SKIP" > "${statusfile}"
          echo "[跳过] baseline dataset=${dataset} shot=${shot} seed=${seed}"
          continue
        fi

        wait_for_any_slot
        slot="${READY_SLOT}"
        gpu_id="${GPU_LIST[$slot]}"
        desc="baseline | dataset=${dataset} | shot=${shot} | seed=${seed}"

        (
          mkdir -p "${outdir}"
          : > "${logfile}"
          write_log_header "${logfile}" "${gpu_id}" "MMRL" "${dataset}" "${shot}" "${seed}" "${tag_name}" "${stage_name}" "${base_root}"
          cleanup_broken_resume_if_needed "${outdir}" "${logfile}"

          echo "[当前执行] ${desc}"

          CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
            --root "${ROOT}" \
            --dataset-config-file "configs/datasets/${dataset}.yaml" \
            --method-config-file "configs/methods/mmrl.yaml" \
            --protocol-config-file "${PROTOCOL_CFG}" \
            --runtime-config-file "${RUNTIME_CFG}" \
            --output-dir "${outdir}" \
            --method MMRL \
            --protocol "${PROTOCOL}" \
            --exec-mode "${EXEC_MODE}" \
            --seed "${seed}" \
            DATASET.NUM_SHOTS "${shot}" \
            DATASET.SUBSAMPLE_CLASSES "${SUBSAMPLE}" \
            MODEL.BACKBONE.NAME "${BACKBONE}" \
            MMRL.ALPHA "${PAPER_ALPHA}" \
            MMRL.REG_WEIGHT "${PAPER_REG_WEIGHT}" \
            MMRL.N_REP_TOKENS "${PAPER_N_REP_TOKENS}" \
            MMRL.REP_LAYERS "${PAPER_REP_LAYERS}" \
            MMRL.REP_DIM "${PAPER_REP_DIM}" \
            >> "${logfile}" 2>&1

          cleanup_checkpoint_if_ready "${outdir}" "${logfile}"
        ) &

        RUNNING_PIDS[$slot]=$!
        SLOT_GPU[$slot]="${gpu_id}"
        SLOT_LOG[$slot]="${logfile}"
        SLOT_DESC[$slot]="${desc}"
      done
    done
  done

  wait_all_jobs

  local dataset2 shot2
  for dataset2 in ${datasets_str}; do
    for shot2 in ${shots_str}; do
      summarize_one_tag "${base_root}" "${global_summary}" "MMRL" "${stage_name}" "${scheme_name}" "${dataset2}" "${shot2}" "${tag_name}"
    done
  done
}

# ------------------------------------------------------------
# 各方案的候选配置
# 执行顺序：C -> B -> A
# ------------------------------------------------------------
declare -a C_TAGS
declare -A C_OPTS

declare -a B_TAGS
declare -A B_OPTS

declare -a A_TAGS
declare -A A_OPTS

declare -a SELECTED_C_TAGS
declare -a SELECTED_B_TAGS
declare -a SELECTED_A_TAGS

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

init_scheme_C() {
  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-row_kl-5e-7_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-7 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-row_kl-1e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-row_kl-2e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 2e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-row_kl-5e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-global_kl-5e-7_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-7 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-global_kl-1e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-global_kl-2e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 2e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-global_kl-5e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-col_kl-1e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE col \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-col_kl-5e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE col \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"
}

init_scheme_B() {
  register_case B_TAGS B_OPTS \
    "stage20_B_rep_clipjoint_blend-0.2_scale-0.05_kl-1e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 1e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.2 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_clipjoint_blend-0.2_scale-0.05_kl-5e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.2 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_clipjoint_blend-0.5_scale-0.05_kl-1e-4_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 1e-4 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_cliptext_blend-0.5_scale-0.05_kl-1e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_text \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 1e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_cliptext_blend-0.5_scale-0.05_kl-5e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_text \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_clipjoint_blend-0.2_scale-0.02_kl-5e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.2 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_cliptext_blend-0.2_scale-0.05_kl-5e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_text \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.2 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case B_TAGS B_OPTS \
    "stage20_B_rep_clipjoint_blend-0.5_scale-0.02_kl-5e-5_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE prior_mean_noise \
    BAYES_MMRL.REP_INIT_STD 0.01 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"
}

init_scheme_A() {
  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-global_kl-1e-5_prior-0.02_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE global \
    BAYES_MMRL.REP_KL_WEIGHT 1e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.02 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-global_kl-5e-5_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE global \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-global_kl-1e-4_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE global \
    BAYES_MMRL.REP_KL_WEIGHT 1e-4 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-per_token_kl-1e-5_prior-0.02_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 1e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.02 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-per_token_kl-5e-5_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-per_token_kl-1e-4_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_KL_WEIGHT 1e-4 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-per_dim_kl-5e-5_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_dim \
    BAYES_MMRL.REP_KL_WEIGHT 5e-5 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case A_TAGS A_OPTS \
    "stage30_A_rep_zero_sig-per_dim_kl-1e-4_prior-0.05_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_dim \
    BAYES_MMRL.REP_KL_WEIGHT 1e-4 \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_INIT_MODE normal \
    BAYES_MMRL.REP_INIT_STD 0.02 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"
}

# ------------------------------------------------------------
# 统一执行某一个方案阶段
# ------------------------------------------------------------
run_scheme_stage() {
  local base_root=$1
  local global_summary=$2
  local stage_name=$3
  local scheme_name=$4
  local tags_array_name=$5
  local opts_assoc_name=$6
  local datasets_str=$7
  local shots_str=$8
  local seeds_str=$9

  print_banner "开始执行 ${stage_name}：方案 ${scheme_name}"

  init_slots

  local dataset shot tag seed slot gpu_id outdir logfile statusfile desc opts_str
  eval "local tags=(\"\${${tags_array_name}[@]}\")"

  for dataset in ${datasets_str}; do
    for shot in ${shots_str}; do
      for tag in "${tags[@]}"; do
        eval "opts_str=\"\${${opts_assoc_name}[${tag}]}\""
        read -r -a EXTRA_OPTS <<< "${opts_str}"

        for seed in ${seeds_str}; do
          outdir="$(build_outdir "${base_root}" "BayesMMRL" "${dataset}" "${shot}" "${tag}" "${seed}")"
          logfile="${outdir}/run.log"
          statusfile="${outdir}/job_status.txt"

          if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
            mkdir -p "${outdir}"
            cleanup_checkpoint_if_ready "${outdir}"
            echo "SKIP" > "${statusfile}"
            echo "[跳过] ${scheme_name} dataset=${dataset} shot=${shot} tag=${tag} seed=${seed}"
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

            echo "[当前执行] ${desc}"
            echo "[说明] 当前使用固定 MMRL 论文主体超参，只搜索 Bayes 新引入参数"
            echo "[说明] 当前评估模式固定为 ${COMMON_EVAL_MODE}"

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
              "${EXTRA_OPTS[@]}" \
              >> "${logfile}" 2>&1

            cleanup_checkpoint_if_ready "${outdir}" "${logfile}"
          ) &

          RUNNING_PIDS[$slot]=$!
          SLOT_GPU[$slot]="${gpu_id}"
          SLOT_LOG[$slot]="${logfile}"
          SLOT_DESC[$slot]="${desc}"
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
# 自动选优：
# 规则：以 mean accuracy_mean 为主
# tie-break: mean macro_f1 desc, mean ece asc, mean nll asc, tag asc
# ------------------------------------------------------------
select_best_tags() {
  if [[ "${AUTO_SELECT_BEST}" != "1" ]]; then
    echo "[跳过] 自动选优"
    return 0
  fi

  if [[ ! -f "${GLOBAL_SEARCH_SUMMARY}" ]]; then
    echo "[错误] 找不到第一阶段全局汇总文件: ${GLOBAL_SEARCH_SUMMARY}" >&2
    exit 1
  fi

  print_banner "开始自动选优：从 A / B / C 各选一个 best tag"

  python - <<PY
import pandas as pd
from pathlib import Path

global_summary = Path(r"${GLOBAL_SEARCH_SUMMARY}")
best_summary = Path(r"${BEST_CONFIG_SUMMARY}")
best_env = Path(r"${BEST_CONFIG_ENV}")

datasets = """${DATASETS}""".split()
shots = """${SHOTS}""".split()
required_n = len(datasets) * len(shots)

df = pd.read_csv(global_summary)

required_cols = {"scheme", "dataset", "shot", "tag", "accuracy_mean"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"global_sweep_summary.csv 缺少必要列: {missing}")

df["dataset"] = df["dataset"].astype(str)
df["shot"] = df["shot"].astype(str)
df["scheme"] = df["scheme"].astype(str)
df["tag"] = df["tag"].astype(str)

rows = []
env_lines = []

for scheme in ["A", "B", "C"]:
    sdf = df[df["scheme"] == scheme].copy()
    sdf = sdf[sdf["dataset"].isin(datasets) & sdf["shot"].isin(shots)].copy()

    if sdf.empty:
        raise SystemExit(f"scheme={scheme} 在搜索汇总中没有可用结果")

    agg_kwargs = {
        "n_cells": ("accuracy_mean", "count"),
        "mean_accuracy": ("accuracy_mean", "mean"),
    }
    if "macro_f1_mean" in sdf.columns:
        agg_kwargs["mean_macro_f1"] = ("macro_f1_mean", "mean")
    if "ece_mean" in sdf.columns:
        agg_kwargs["mean_ece"] = ("ece_mean", "mean")
    if "nll_mean" in sdf.columns:
        agg_kwargs["mean_nll"] = ("nll_mean", "mean")

    grouped = sdf.groupby("tag", dropna=False).agg(**agg_kwargs).reset_index()

    if "mean_macro_f1" not in grouped.columns:
        grouped["mean_macro_f1"] = float("-inf")
    if "mean_ece" not in grouped.columns:
        grouped["mean_ece"] = float("inf")
    if "mean_nll" not in grouped.columns:
        grouped["mean_nll"] = float("inf")

    complete = grouped[grouped["n_cells"] == required_n].copy()
    if complete.empty:
        use = grouped.copy()
        complete_flag = "incomplete_fallback"
    else:
        use = complete.copy()
        complete_flag = "complete"

    use = use.sort_values(
        by=["n_cells", "mean_accuracy", "mean_macro_f1", "mean_ece", "mean_nll", "tag"],
        ascending=[False, False, False, True, True, True]
    ).reset_index(drop=True)

    best = use.iloc[0]

    rows.append({
        "scheme": scheme,
        "best_tag": best["tag"],
        "mean_accuracy": best["mean_accuracy"],
        "mean_macro_f1": best["mean_macro_f1"],
        "mean_ece": best["mean_ece"],
        "mean_nll": best["mean_nll"],
        "num_cells": int(best["n_cells"]),
        "required_cells": required_n,
        "selection_mode": complete_flag,
        "selection_rule": "accuracy_mean desc > macro_f1_mean desc > ece_mean asc > nll_mean asc > tag asc",
    })

    env_lines.append(f"{scheme}_BEST_TAG={best['tag']}")

best_df = pd.DataFrame(rows)
best_df.to_csv(best_summary, index=False)

with best_env.open("w", encoding="utf-8") as f:
    for line in env_lines:
        f.write(line + "\\n")

print("[OK] saved:", best_summary)
print("[OK] saved:", best_env)
print(best_df.to_string(index=False))
PY
}

# ------------------------------------------------------------
# 第二阶段：自动确认
# baseline + A_best + B_best + C_best
# shots: 1 2 4 8 16 32
# seeds: 1 2 3
# ------------------------------------------------------------
run_confirm_stage() {
  if [[ "${AUTO_CONFIRM}" != "1" ]]; then
    echo "[跳过] 第二阶段自动确认"
    return 0
  fi

  if [[ ! -f "${BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到 best 配置文件: ${BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${BEST_CONFIG_ENV}"

  if [[ -z "${A_BEST_TAG:-}" || -z "${B_BEST_TAG:-}" || -z "${C_BEST_TAG:-}" ]]; then
    echo "[错误] 自动选优结果不完整，请检查: ${BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  SELECTED_A_TAGS=("${A_BEST_TAG}")
  SELECTED_B_TAGS=("${B_BEST_TAG}")
  SELECTED_C_TAGS=("${C_BEST_TAG}")

  print_banner "开始第二阶段自动确认"
  echo "确认数据集: ${CONFIRM_DATASETS}"
  echo "确认 shots: ${CONFIRM_SHOTS}"
  echo "确认 seeds: ${CONFIRM_SEEDS}"
  echo "A_best: ${A_BEST_TAG}"
  echo "B_best: ${B_BEST_TAG}"
  echo "C_best: ${C_BEST_TAG}"

  rm -f "${GLOBAL_CONFIRM_SUMMARY}"

  run_baseline_stage \
    "${CONFIRM_ROOT}" \
    "${GLOBAL_CONFIRM_SUMMARY}" \
    "confirm00_baseline_mmrl_paper_fixed" \
    "baseline" \
    "confirm00_baseline_mmrl_paper_fixed" \
    "${CONFIRM_DATASETS}" \
    "${CONFIRM_SHOTS}" \
    "${CONFIRM_SEEDS}"

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm10_schemeC_best" \
      "C_best" \
      "SELECTED_C_TAGS" \
      "C_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm20_schemeB_best" \
      "B_best" \
      "SELECTED_B_TAGS" \
      "B_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm30_schemeA_best" \
      "A_best" \
      "SELECTED_A_TAGS" \
      "A_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  print_summary_table "${GLOBAL_CONFIRM_SUMMARY}" "第二阶段确认汇总"
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
main() {
  resolve_stage_defaults
  resolve_confirm_defaults
  init_gpu_list

  SEARCH_ROOT="${OUTPUT_ROOT}/${STAGE}/search_stage"
  CONFIRM_ROOT="${OUTPUT_ROOT}/${STAGE}/confirm_stage"

  GLOBAL_SEARCH_SUMMARY="${SEARCH_ROOT}/global_sweep_summary.csv"
  GLOBAL_CONFIRM_SUMMARY="${CONFIRM_ROOT}/confirm_global_summary.csv"
  BEST_CONFIG_SUMMARY="${SEARCH_ROOT}/best_config_summary.csv"
  BEST_CONFIG_ENV="${SEARCH_ROOT}/best_config.env"

  mkdir -p "${SEARCH_ROOT}"
  mkdir -p "${CONFIRM_ROOT}"

  rm -f "${GLOBAL_SEARCH_SUMMARY}"
  rm -f "${BEST_CONFIG_SUMMARY}"
  rm -f "${BEST_CONFIG_ENV}"

  init_scheme_C
  init_scheme_B
  init_scheme_A

  print_banner "BayesMMRL 两阶段搜索开始"
  echo "第一阶段数据集: ${DATASETS}"
  echo "第一阶段 shots: ${SHOTS}"
  echo "第一阶段 seeds: ${SEEDS}"
  echo "第二阶段数据集: ${CONFIRM_DATASETS}"
  echo "第二阶段 shots: ${CONFIRM_SHOTS}"
  echo "第二阶段 seeds: ${CONFIRM_SEEDS}"
  echo "GPU: ${GPU_IDS}"
  echo "搜索输出目录: ${SEARCH_ROOT}"
  echo "确认输出目录: ${CONFIRM_ROOT}"
  echo "Backbone: ${BACKBONE}"
  echo "评估模式(固定): ${COMMON_EVAL_MODE}"
  echo "测试后自动删 checkpoint: ${DELETE_CKPT_AFTER_TEST}"
  echo "执行顺序: baseline -> 方案C -> 方案B -> 方案A -> 自动选优 -> 自动确认"

  trap 'echo "[中断] 正在停止子任务..."; cleanup_children; exit 130' INT TERM

  run_baseline_stage \
    "${SEARCH_ROOT}" \
    "${GLOBAL_SEARCH_SUMMARY}" \
    "stage00_baseline_mmrl_paper_fixed" \
    "baseline" \
    "stage00_baseline_mmrl_paper_fixed" \
    "${DATASETS}" \
    "${SHOTS}" \
    "${SEEDS}"

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${SEARCH_ROOT}" \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "stage10_schemeC_projrep_first" \
      "C" \
      "C_TAGS" \
      "C_OPTS" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${SEEDS}"
  else
    echo "[跳过] 方案 C"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage \
      "${SEARCH_ROOT}" \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "stage20_schemeB_rep_clip_prior" \
      "B" \
      "B_TAGS" \
      "B_OPTS" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${SEEDS}"
  else
    echo "[跳过] 方案 B"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage \
      "${SEARCH_ROOT}" \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "stage30_schemeA_rep_zero_prior" \
      "A" \
      "A_TAGS" \
      "A_OPTS" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${SEEDS}"
  else
    echo "[跳过] 方案 A"
  fi

  print_summary_table "${GLOBAL_SEARCH_SUMMARY}" "第一阶段搜索汇总"

  select_best_tags
  print_best_table "${BEST_CONFIG_SUMMARY}"

  run_confirm_stage

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo
    echo "[结束] 有 ${FAILED_JOBS} 个任务失败，请检查对应 run.log"
    exit 1
  fi

  echo
  echo "[结束] 所有任务执行完成"
}

main "$@"
