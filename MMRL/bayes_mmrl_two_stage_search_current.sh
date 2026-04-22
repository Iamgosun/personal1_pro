#!/bin/bash
set -euo pipefail

# ============================================================
# BayesMMRL：ALPHA 优先搜索 -> 主参数搜索 -> 16-shot N_MC_TRAIN -> 最终确认
#
# 已实际核对过的代码实现要点：
# 1) 训练入口使用 MMRL/run.py，并通过命令行 opts 覆盖 YACS 配置。
# 2) 结果汇总使用 MMRL/evaluation/result_parser.py，生成 test_summary.csv。
# 3) KL warmup 在 online_executor.py 中生效：
#    - 前 KL_WARMUP_EPOCHS 个 epoch：kl_beta = 0
#    - 之后线性从 0 拉升到 1
# 4) 当前有效参数空间（不是旧 checkpoint 里的历史字段）：
#    - A/B: REP_SIGMA_MODE 只能是 global / per_token
#    - A/B: REP_PRIOR_MODE 只能是 zero / clip_joint
#    - C  : PROJ_REP_SIGMA_MODE 只能是 global / row
#    - C  : PROJ_REP_PRIOR_MODE 只能是 self_proj_rep / clip_proj
#    - 当前代码已支持 EVAL_AGGREGATION: prob_mean / logit_mean
#
# 当前流程策略：
#   0) 可选 baseline：固定论文配置的 MMRL（默认关闭）
#   1) ALPHA 搜索：先在 16-shot 上用每个 scheme 的锚点配置搜索 alpha
#   2) 主参数搜索：固定 alpha_best 后，再搜索 A/B/C 的主参数空间
#   3) N_MC_TRAIN 搜索：固定 alpha_best + 主参数 best，只在 16-shot 搜索 {3,5,10,20}
#   4) 最终确认：只跑 A/B/C 的最终 best，不再做搜索，也默认不跑 baseline
#
# 注意：
#   1) 请在 MMRL 根目录运行，或显式设置 PROJECT_DIR=/path/to/MMRL
#   2) 本脚本固定 MMRL 主体超参，只搜索 Bayes 新引入的参数
#   3) 本脚本显式传入 EVAL_AGGREGATION，避免默认值歧义
# ============================================================

# ---------------------------
# 基础环境
# ---------------------------
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
ROOT=${ROOT:-DATASETS}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}

# 第一阶段规模：
#   coarse3 -> caltech101 / oxford_pets / ucf101
#   full11  -> 11 个常用数据集
STAGE=${STAGE:-coarse3}

# 第一阶段如果手动指定，则覆盖默认
DATASETS=${DATASETS:-}
SHOTS=${SHOTS:-}
SEEDS=${SEEDS:-}

# alpha-first 阶段：默认沿用第一阶段数据集 / seeds，只在 16-shot 上跑
ALPHA_DATASETS=${ALPHA_DATASETS:-}
ALPHA_SHOTS=${ALPHA_SHOTS:-}
ALPHA_SEEDS=${ALPHA_SEEDS:-}

# N_MC_TRAIN 搜索阶段：默认沿用第一阶段数据集 / seeds，只在 16-shot 上跑
NMC_DATASETS=${NMC_DATASETS:-}
NMC_SHOTS=${NMC_SHOTS:-}
NMC_SEEDS=${NMC_SEEDS:-}

# 最终确认阶段：只跑最终 best，不再做任何搜索
CONFIRM_DATASETS=${CONFIRM_DATASETS:-}
CONFIRM_SHOTS=${CONFIRM_SHOTS:-}
CONFIRM_SEEDS=${CONFIRM_SEEDS:-}

# 输出目录
OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayes_mmrl_search_current}

# GPU 调度
# 规则：
# 1) 如果显式设置了 GPU_IDS，则优先按 GPU_IDS 使用
# 2) 否则如果显式设置了 NGPU，则自动使用 0..NGPU-1
# 3) 否则自动探测当前机器可见 GPU 数量
NGPU=${NGPU:-}
GPU_IDS=${GPU_IDS:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# 开关
RUN_BASELINE=${RUN_BASELINE:-0}
RUN_SCHEME_C=${RUN_SCHEME_C:-1}
RUN_SCHEME_B=${RUN_SCHEME_B:-1}
RUN_SCHEME_A=${RUN_SCHEME_A:-1}
AUTO_SELECT_BEST=${AUTO_SELECT_BEST:-1}
AUTO_CONFIRM=${AUTO_CONFIRM:-1}
DELETE_CKPT_AFTER_TEST=${DELETE_CKPT_AFTER_TEST:-1}

# ---------------------------
# 固定的主体超参（与当前 BayesMMRL 默认配置对齐）
# ---------------------------
PAPER_ALPHA=${PAPER_ALPHA:-0.7}
PAPER_REG_WEIGHT=${PAPER_REG_WEIGHT:-0.5}
PAPER_N_REP_TOKENS=${PAPER_N_REP_TOKENS:-5}
PAPER_REP_LAYERS=${PAPER_REP_LAYERS:-'[6,7,8,9,10,11,12]'}
PAPER_REP_DIM=${PAPER_REP_DIM:-512}

COMMON_N_MC_TRAIN=${COMMON_N_MC_TRAIN:-3}
COMMON_N_MC_TEST=${COMMON_N_MC_TEST:-10}
COMMON_EVAL_MODE=${COMMON_EVAL_MODE:-mc_predictive}
# 当前 bayesmmrl.yaml 写的是 logit_mean；config.py 的默认值是 prob_mean。
# 为避免歧义，脚本里显式传值。
COMMON_EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION:-logit_mean}
COMMON_KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS:-5}

METHOD_CFG="configs/methods/bayesmmrl.yaml"
MMRL_METHOD_CFG="configs/methods/mmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

# ---------------------------
# 输出路径变量
# ---------------------------
SEARCH_ROOT=""
NMC_ROOT=""
ALPHA_ROOT=""
CONFIRM_ROOT=""
GLOBAL_SEARCH_SUMMARY=""
GLOBAL_NMC_SUMMARY=""
GLOBAL_ALPHA_SUMMARY=""
GLOBAL_CONFIRM_SUMMARY=""
BEST_CONFIG_SUMMARY=""
BEST_CONFIG_ENV=""
NMC_BEST_CONFIG_SUMMARY=""
NMC_BEST_CONFIG_ENV=""
ALPHA_BEST_CONFIG_SUMMARY=""
ALPHA_BEST_CONFIG_ENV=""

# ------------------------------------------------------------
# 协议对应的阶段语义
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
        echo "未知 STAGE=${STAGE}" >&2
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
# alpha / nmc 轻量搜索阶段默认规模
# ------------------------------------------------------------
resolve_tune_defaults() {
  if [[ -z "${ALPHA_DATASETS}" ]]; then
    ALPHA_DATASETS="${DATASETS}"
  fi
  if [[ -z "${ALPHA_SHOTS}" ]]; then
    ALPHA_SHOTS="16"
  fi
  if [[ -z "${ALPHA_SEEDS}" ]]; then
    ALPHA_SEEDS="${SEEDS}"
  fi

  if [[ -z "${NMC_DATASETS}" ]]; then
    NMC_DATASETS="${DATASETS}"
  fi
  if [[ -z "${NMC_SHOTS}" ]]; then
    NMC_SHOTS="16"
  fi
  if [[ -z "${NMC_SEEDS}" ]]; then
    NMC_SEEDS="${SEEDS}"
  fi
}

# ------------------------------------------------------------
# 最终确认阶段默认规模
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

  # 情况 1：用户显式指定 GPU_IDS，优先使用
  if [[ -n "${GPU_IDS}" ]]; then
    read -r -a GPU_LIST <<< "${GPU_IDS}"

  # 情况 2：用户显式指定 NGPU，则自动映射为 0..NGPU-1
  elif [[ -n "${NGPU}" ]]; then
    local i
    for ((i=0; i<NGPU; i++)); do
      GPU_LIST+=("$i")
    done

  # 情况 3：都没指定，则自动探测当前可见 GPU
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      while IFS= read -r idx; do
        [[ -n "${idx}" ]] && GPU_LIST+=("${idx}")
      done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    fi
  fi

  if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
    echo "没有可用 GPU，请设置 GPU_IDS 或 NGPU" >&2
    exit 1
  fi

  # 如果 NGPU 没显式设置，则根据最终 GPU_LIST 自动回填
  if [[ -z "${NGPU}" ]]; then
    NGPU=${#GPU_LIST[@]}
  fi

  # 如果 GPU_IDS 没显式设置，则根据最终 GPU_LIST 自动回填
  if [[ -z "${GPU_IDS}" ]]; then
    GPU_IDS="${GPU_LIST[*]}"
  fi

  echo "[GPU 调度] 使用 GPU: ${GPU_IDS}"
  echo "[GPU 调度] 槽位数: ${#GPU_LIST[@]}"
}

# ------------------------------------------------------------
# 工具函数：字符串清洗 / 路径构造
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

  if [[ -f "${outdir}/test_metrics.json" && -d "${outdir}/refactor_model" ]]; then
    rm -rf "${outdir}/refactor_model"
    if [[ -n "${logfile}" ]]; then
      echo "[清理] 测试完成，已删除 checkpoint 目录: ${outdir}/refactor_model" >> "${logfile}"
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
    fi
  fi
}

# ------------------------------------------------------------
# 汇总工具：对单个 tag 做聚合，并追加到全局表
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
    sort_cols = [c for c in ["stage_name", "scheme", "dataset", "shot", "accuracy_mean"] if c in df.columns]
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

announce_job_started() {
  local slot=$1
  echo "[启动] [slot=${slot}] [gpu=${SLOT_GPU[$slot]}] [pid=${RUNNING_PIDS[$slot]}] ${SLOT_DESC[$slot]}"
  echo "       log: ${SLOT_LOG[$slot]}"
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

  local dataset shot seed slot gpu_id outdir logfile desc
  for dataset in ${datasets_str}; do
    for shot in ${shots_str}; do
      for seed in ${seeds_str}; do
        outdir="$(build_outdir "${base_root}" "MMRL" "${dataset}" "${shot}" "${tag_name}" "${seed}")"
        logfile="${outdir}/run.log"

        if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
          mkdir -p "${outdir}"
          cleanup_checkpoint_if_ready "${outdir}"
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

          echo "[当前执行] ${desc}" >> "${logfile}"
          (
            cd "${PROJECT_DIR}"
            CUDA_VISIBLE_DEVICES="${gpu_id}" python run.py \
              --root "${ROOT}" \
              --dataset-config-file "configs/datasets/${dataset}.yaml" \
              --method-config-file "${MMRL_METHOD_CFG}" \
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
              MMRL.REP_DIM "${PAPER_REP_DIM}"
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

  wait_all_jobs

  local dataset2 shot2
  for dataset2 in ${datasets_str}; do
    for shot2 in ${shots_str}; do
      summarize_one_tag "${base_root}" "${global_summary}" "MMRL" "${stage_name}" "${scheme_name}" "${dataset2}" "${shot2}" "${tag_name}"
    done
  done
}

# ------------------------------------------------------------
# 三个方案的当前有效搜索空间
# ------------------------------------------------------------
declare -a C_TAGS
declare -A C_OPTS

declare -a B_TAGS
declare -A B_OPTS

declare -a A_TAGS
declare -A A_OPTS

# alpha-first 后，主参数搜索要使用“固定 alpha”的搜索空间
declare -a C_SEARCH_TAGS
declare -A C_SEARCH_OPTS
declare -a B_SEARCH_TAGS
declare -A B_SEARCH_OPTS
declare -a A_SEARCH_TAGS
declare -A A_SEARCH_OPTS

declare -a SELECTED_C_TAGS
declare -a SELECTED_B_TAGS
declare -a SELECTED_A_TAGS

declare -a A_NMC_TAGS
declare -A A_NMC_OPTS
declare -a B_NMC_TAGS
declare -A B_NMC_OPTS
declare -a C_NMC_TAGS
declare -A C_NMC_OPTS

declare -a A_ALPHA_TAGS
declare -A A_ALPHA_OPTS
declare -a B_ALPHA_TAGS
declare -A B_ALPHA_OPTS
declare -a C_ALPHA_TAGS
declare -A C_ALPHA_OPTS

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
  # 方案 C：Bayes on proj_rep
  local sigma prior_mode kl prior_std

  for sigma in row global; do
    for prior_mode in self_proj_rep clip_proj; do
      for kl in 1e-6 5e-6  1e-5 1e-4 ; do
        prior_std=0.01
        register_case C_TAGS C_OPTS \
          "C_sig-${sigma}_pmode-${prior_mode}_pstd-${prior_std}_kl-${kl}" \
          BAYES_MMRL.BAYES_TARGET proj_rep \
          BAYES_MMRL.PROJ_REP_SIGMA_MODE "${sigma}" \
          BAYES_MMRL.PROJ_REP_PRIOR_MODE "${prior_mode}" \
          BAYES_MMRL.PROJ_REP_PRIOR_STD "${prior_std}" \
          BAYES_MMRL.PROJ_REP_KL_WEIGHT "${kl}"
      done
    done
  done

  for prior_mode in self_proj_rep clip_proj; do
    register_case C_TAGS C_OPTS \
      "C_sig-row_pmode-${prior_mode}_pstd-0.005_kl-1e-5" \
      BAYES_MMRL.BAYES_TARGET proj_rep \
      BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
      BAYES_MMRL.PROJ_REP_PRIOR_MODE "${prior_mode}" \
      BAYES_MMRL.PROJ_REP_PRIOR_STD 0.005 \
      BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-5

    register_case C_TAGS C_OPTS \
      "C_sig-row_pmode-${prior_mode}_pstd-0.02_kl-1e-6" \
      BAYES_MMRL.BAYES_TARGET proj_rep \
      BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
      BAYES_MMRL.PROJ_REP_PRIOR_MODE "${prior_mode}" \
      BAYES_MMRL.PROJ_REP_PRIOR_STD 0.02 \
      BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-6
  done
}

init_scheme_B() {
  # 方案 B：Bayes on R，CLIP joint prior
  local sigma kl blend scale

  for sigma in global per_token; do
    for kl in 2e-4 5e-4 1e-3; do
      for blend in 0.2 0.5; do
        for scale in 0.02 0.05; do
          register_case B_TAGS B_OPTS \
            "B_sig-${sigma}_pstd-0.05_kl-${kl}_blend-${blend}_scale-${scale}" \
            BAYES_MMRL.BAYES_TARGET rep_tokens \
            BAYES_MMRL.REP_PRIOR_MODE clip_joint \
            BAYES_MMRL.REP_SIGMA_MODE "${sigma}" \
            BAYES_MMRL.REP_PRIOR_STD 0.05 \
            BAYES_MMRL.REP_KL_WEIGHT "${kl}" \
            BAYES_MMRL.CLIP_PRIOR_BLEND "${blend}" \
            BAYES_MMRL.CLIP_PRIOR_SCALE "${scale}"
        done
      done
    done
  done

  register_case B_TAGS B_OPTS \
    "B_sig-per_token_pstd-0.02_kl-5e-4_blend-0.5_scale-0.05" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_PRIOR_STD 0.02 \
    BAYES_MMRL.REP_KL_WEIGHT 5e-4 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05

  register_case B_TAGS B_OPTS \
    "B_sig-per_token_pstd-0.1_kl-5e-4_blend-0.5_scale-0.05" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE clip_joint \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_PRIOR_STD 0.1 \
    BAYES_MMRL.REP_KL_WEIGHT 5e-4 \
    BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
    BAYES_MMRL.CLIP_PRIOR_SCALE 0.05
}

init_scheme_A() {
  # 方案 A：Bayes on R，zero prior
  local sigma prior_std kl

  for sigma in global per_token; do
    for prior_std in 0.02 0.05 0.1; do
      for kl in 1e-5 5e-4 1e-4 1e-3 ; do
        register_case A_TAGS A_OPTS \
          "A_sig-${sigma}_pstd-${prior_std}_kl-${kl}" \
          BAYES_MMRL.BAYES_TARGET rep_tokens \
          BAYES_MMRL.REP_PRIOR_MODE zero \
          BAYES_MMRL.REP_SIGMA_MODE "${sigma}" \
          BAYES_MMRL.REP_PRIOR_STD "${prior_std}" \
          BAYES_MMRL.REP_KL_WEIGHT "${kl}"
      done
    done
  done

  register_case A_TAGS A_OPTS \
    "A_sig-per_token_pstd-0.05_kl-3e-3" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE per_token \
    BAYES_MMRL.REP_PRIOR_STD 0.05 \
    BAYES_MMRL.REP_KL_WEIGHT 3e-3

  register_case A_TAGS A_OPTS \
    "A_sig-global_pstd-0.1_kl-1e-3" \
    BAYES_MMRL.BAYES_TARGET rep_tokens \
    BAYES_MMRL.REP_PRIOR_MODE zero \
    BAYES_MMRL.REP_SIGMA_MODE global \
    BAYES_MMRL.REP_PRIOR_STD 0.1 \
    BAYES_MMRL.REP_KL_WEIGHT 1e-3
}

# ------------------------------------------------------------
# ALPHA 优先搜索：使用各 scheme 的固定锚点，仅搜索 alpha
# ------------------------------------------------------------
init_alpha_stage_cases() {
  A_ALPHA_TAGS=(); B_ALPHA_TAGS=(); C_ALPHA_TAGS=()
  A_ALPHA_OPTS=(); B_ALPHA_OPTS=(); C_ALPHA_OPTS=()

  local alpha

  # A: zero prior 的代表性锚点
  for alpha in 0.0 0.3 0.5 0.7 1.0; do
    register_case A_ALPHA_TAGS A_ALPHA_OPTS \
      "A_anchor_sig-per_token_pstd-0.05_kl-1e-3_alpha-${alpha}" \
      BAYES_MMRL.BAYES_TARGET rep_tokens \
      BAYES_MMRL.REP_PRIOR_MODE zero \
      BAYES_MMRL.REP_SIGMA_MODE per_token \
      BAYES_MMRL.REP_PRIOR_STD 0.05 \
      BAYES_MMRL.REP_KL_WEIGHT 1e-3 \
      BAYES_MMRL.ALPHA "${alpha}"
  done

  # B: clip_joint prior 的代表性锚点
  for alpha in 0.0 0.3 0.5 0.7 1.0; do
    register_case B_ALPHA_TAGS B_ALPHA_OPTS \
      "B_anchor_sig-per_token_pstd-0.05_kl-5e-4_blend-0.5_scale-0.05_alpha-${alpha}" \
      BAYES_MMRL.BAYES_TARGET rep_tokens \
      BAYES_MMRL.REP_PRIOR_MODE clip_joint \
      BAYES_MMRL.REP_SIGMA_MODE per_token \
      BAYES_MMRL.REP_PRIOR_STD 0.05 \
      BAYES_MMRL.REP_KL_WEIGHT 5e-4 \
      BAYES_MMRL.CLIP_PRIOR_BLEND 0.5 \
      BAYES_MMRL.CLIP_PRIOR_SCALE 0.05 \
      BAYES_MMRL.ALPHA "${alpha}"
  done

  # C: proj_rep 的代表性锚点
  for alpha in 0.0 0.3 0.5 0.7 1.0; do
    register_case C_ALPHA_TAGS C_ALPHA_OPTS \
      "C_anchor_sig-row_pmode-self_proj_rep_pstd-0.01_kl-1e-5_alpha-${alpha}" \
      BAYES_MMRL.BAYES_TARGET proj_rep \
      BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
      BAYES_MMRL.PROJ_REP_PRIOR_MODE self_proj_rep \
      BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
      BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-5 \
      BAYES_MMRL.ALPHA "${alpha}"
  done
}

# ------------------------------------------------------------
# 固定 alpha_best 后，构造主参数搜索空间
# ------------------------------------------------------------
init_search_stage_cases_with_alpha() {
  if [[ ! -f "${ALPHA_BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到 alpha best 配置文件: ${ALPHA_BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${ALPHA_BEST_CONFIG_ENV}"

  A_SEARCH_TAGS=(); B_SEARCH_TAGS=(); C_SEARCH_TAGS=()
  A_SEARCH_OPTS=(); B_SEARCH_OPTS=(); C_SEARCH_OPTS=()

  local tag base_opts

  for tag in "${A_TAGS[@]}"; do
    base_opts="${A_OPTS[$tag]}"
    register_case A_SEARCH_TAGS A_SEARCH_OPTS \
      "${tag}" \
      ${base_opts} \
      BAYES_MMRL.ALPHA "${A_ALPHA_BEST_VALUE}"
  done

  for tag in "${B_TAGS[@]}"; do
    base_opts="${B_OPTS[$tag]}"
    register_case B_SEARCH_TAGS B_SEARCH_OPTS \
      "${tag}" \
      ${base_opts} \
      BAYES_MMRL.ALPHA "${B_ALPHA_BEST_VALUE}"
  done

  for tag in "${C_TAGS[@]}"; do
    base_opts="${C_OPTS[$tag]}"
    register_case C_SEARCH_TAGS C_SEARCH_OPTS \
      "${tag}" \
      ${base_opts} \
      BAYES_MMRL.ALPHA "${C_ALPHA_BEST_VALUE}"
  done
}

# ------------------------------------------------------------
# 统一执行某一方案阶段
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

  local dataset shot tag seed slot gpu_id outdir logfile desc opts_str
  eval "local tags=(\"\${${tags_array_name}[@]}\")"

  for dataset in ${datasets_str}; do
    for shot in ${shots_str}; do
      for tag in "${tags[@]}"; do
        eval "opts_str=\"\${${opts_assoc_name}[${tag}]}\""
        read -r -a EXTRA_OPTS <<< "${opts_str}"

        for seed in ${seeds_str}; do
          outdir="$(build_outdir "${base_root}" "BayesMMRL" "${dataset}" "${shot}" "${tag}" "${seed}")"
          logfile="${outdir}/run.log"

          if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
            mkdir -p "${outdir}"
            cleanup_checkpoint_if_ready "${outdir}"
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

            echo "[当前执行] ${desc}" >> "${logfile}"
            echo "[说明] 当前使用固定 MMRL 主体超参，只搜索 Bayes 参数" >> "${logfile}"
            echo "[说明] EVAL_MODE=${COMMON_EVAL_MODE}, EVAL_AGGREGATION=${COMMON_EVAL_AGGREGATION}" >> "${logfile}"
            echo "[说明] KL_WARMUP_EPOCHS=${COMMON_KL_WARMUP_EPOCHS}" >> "${logfile}"

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
# 自动选优：通用版
# ------------------------------------------------------------
select_best_tags_from_summary() {
  local summary_csv=$1
  local best_summary=$2
  local best_env=$3
  local env_suffix=$4
  local datasets_str=$5
  local shots_str=$6
  local title=$7

  if [[ ! -f "${summary_csv}" ]]; then
    echo "[错误] 找不到汇总文件: ${summary_csv}" >&2
    exit 1
  fi

  print_banner "${title}"

  python - <<PY
import pandas as pd
from pathlib import Path

global_summary = Path(r"${summary_csv}")
best_summary = Path(r"${best_summary}")
best_env = Path(r"${best_env}")
env_suffix = "${env_suffix}"

datasets = """${datasets_str}""".split()
shots = """${shots_str}""".split()
required_n = len(datasets) * len(shots)

df = pd.read_csv(global_summary)
required_cols = {"scheme", "dataset", "shot", "tag", "accuracy_mean"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"汇总文件缺少必要列: {missing}")

for c in ["dataset", "shot", "scheme", "tag"]:
    df[c] = df[c].astype(str)

rows = []
env_lines = []
for scheme in ["A", "B", "C"]:
    sdf = df[df["scheme"] == scheme].copy()
    sdf = sdf[sdf["dataset"].isin(datasets) & sdf["shot"].isin(shots)].copy()
    if sdf.empty:
        raise SystemExit(f"scheme={scheme} 在汇总中没有可用结果")

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
        mode = "incomplete_fallback"
    else:
        use = complete.copy()
        mode = "complete"

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
        "selection_mode": mode,
        "selection_rule": "accuracy_mean desc > macro_f1_mean desc > ece_mean asc > nll_mean asc > tag asc",
    })
    env_lines.append(f"{scheme}_{env_suffix}_TAG={best['tag']}")

best_df = pd.DataFrame(rows)
best_df.to_csv(best_summary, index=False)
with best_env.open("w", encoding="utf-8") as f:
    for line in env_lines:
        f.write(line + "\n")

print("[OK] saved:", best_summary)
print("[OK] saved:", best_env)
print(best_df.to_string(index=False))
PY
}

select_best_tags() {
  if [[ "${AUTO_SELECT_BEST}" != "1" ]]; then
    echo "[跳过] 自动选优"
    return 0
  fi

  select_best_tags_from_summary \
    "${GLOBAL_SEARCH_SUMMARY}" \
    "${BEST_CONFIG_SUMMARY}" \
    "${BEST_CONFIG_ENV}" \
    "BEST" \
    "${DATASETS}" \
    "${SHOTS}" \
    "开始自动选优：从 A / B / C 各选一个主参数 best tag"
}

select_best_nmc_tags() {
  select_best_tags_from_summary \
    "${GLOBAL_NMC_SUMMARY}" \
    "${NMC_BEST_CONFIG_SUMMARY}" \
    "${NMC_BEST_CONFIG_ENV}" \
    "NMC_BEST" \
    "${NMC_DATASETS}" \
    "${NMC_SHOTS}" \
    "开始自动选优：从 A / B / C 各选一个 N_MC_TRAIN best tag"
}

select_best_alpha_tags() {
  if [[ ! -f "${GLOBAL_ALPHA_SUMMARY}" ]]; then
    echo "[错误] 找不到 ALPHA 汇总文件: ${GLOBAL_ALPHA_SUMMARY}" >&2
    exit 1
  fi

  print_banner "开始自动选优：从 A / B / C 各选一个 alpha best"

  python - <<PY
import re
import pandas as pd
from pathlib import Path

global_summary = Path(r"${GLOBAL_ALPHA_SUMMARY}")
best_summary = Path(r"${ALPHA_BEST_CONFIG_SUMMARY}")
best_env = Path(r"${ALPHA_BEST_CONFIG_ENV}")

datasets = """${ALPHA_DATASETS}""".split()
shots = """${ALPHA_SHOTS}""".split()
required_n = len(datasets) * len(shots)

df = pd.read_csv(global_summary)
required_cols = {"scheme", "dataset", "shot", "tag", "accuracy_mean"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"汇总文件缺少必要列: {missing}")

for c in ["dataset", "shot", "scheme", "tag"]:
    df[c] = df[c].astype(str)

rows = []
env_lines = []

for scheme in ["A", "B", "C"]:
    sdf = df[df["scheme"] == scheme].copy()
    sdf = sdf[sdf["dataset"].isin(datasets) & sdf["shot"].isin(shots)].copy()
    if sdf.empty:
        raise SystemExit(f"scheme={scheme} 在 alpha 汇总中没有可用结果")

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
        mode = "incomplete_fallback"
    else:
        use = complete.copy()
        mode = "complete"

    use = use.sort_values(
        by=["n_cells", "mean_accuracy", "mean_macro_f1", "mean_ece", "mean_nll", "tag"],
        ascending=[False, False, False, True, True, True]
    ).reset_index(drop=True)

    best = use.iloc[0]
    tag = str(best["tag"])
    m = re.search(r"_alpha-([0-9.]+)$", tag)
    if not m:
        raise SystemExit(f"无法从 tag 解析 alpha: {tag}")
    alpha_value = m.group(1)

    rows.append({
        "scheme": scheme,
        "best_tag": tag,
        "best_alpha": alpha_value,
        "mean_accuracy": best["mean_accuracy"],
        "mean_macro_f1": best["mean_macro_f1"],
        "mean_ece": best["mean_ece"],
        "mean_nll": best["mean_nll"],
        "num_cells": int(best["n_cells"]),
        "required_cells": required_n,
        "selection_mode": mode,
        "selection_rule": "accuracy_mean desc > macro_f1_mean desc > ece_mean asc > nll_mean asc > tag asc",
    })

    env_lines.append(f"{scheme}_ALPHA_BEST_TAG={tag}")
    env_lines.append(f"{scheme}_ALPHA_BEST_VALUE={alpha_value}")

best_df = pd.DataFrame(rows)
best_df.to_csv(best_summary, index=False)
with best_env.open("w", encoding="utf-8") as f:
    for line in env_lines:
        f.write(line + "\n")

print("[OK] saved:", best_summary)
print("[OK] saved:", best_env)
print(best_df.to_string(index=False))
PY
}

init_nmc_stage_cases() {
  if [[ ! -f "${BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到主参数 best 配置文件: ${BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "${BEST_CONFIG_ENV}"

  A_NMC_TAGS=(); B_NMC_TAGS=(); C_NMC_TAGS=()
  A_NMC_OPTS=(); B_NMC_OPTS=(); C_NMC_OPTS=()

  local nmc base_opts

  base_opts="${A_SEARCH_OPTS[$A_BEST_TAG]}"
  for nmc in 3 5 8 ; do
    register_case A_NMC_TAGS A_NMC_OPTS \
      "${A_BEST_TAG}_nmc-${nmc}" \
      ${base_opts} \
      BAYES_MMRL.N_MC_TRAIN "${nmc}"
  done

  base_opts="${B_SEARCH_OPTS[$B_BEST_TAG]}"
  for nmc in  3 5 8 ; do
    register_case B_NMC_TAGS B_NMC_OPTS \
      "${B_BEST_TAG}_nmc-${nmc}" \
      ${base_opts} \
      BAYES_MMRL.N_MC_TRAIN "${nmc}"
  done

  base_opts="${C_SEARCH_OPTS[$C_BEST_TAG]}"
  for nmc in  3 5 8 ; do
    register_case C_NMC_TAGS C_NMC_OPTS \
      "${C_BEST_TAG}_nmc-${nmc}" \
      ${base_opts} \
      BAYES_MMRL.N_MC_TRAIN "${nmc}"
  done
}

run_alpha_search_stage() {
  print_banner "开始第一阶段：先在 16-shot 上搜索 ALPHA"
  rm -f "${GLOBAL_ALPHA_SUMMARY}"
  init_alpha_stage_cases

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${ALPHA_ROOT}" \
      "${GLOBAL_ALPHA_SUMMARY}" \
      "stage10_schemeC_alpha_first" \
      "C" \
      "C_ALPHA_TAGS" \
      "C_ALPHA_OPTS" \
      "${ALPHA_DATASETS}" \
      "${ALPHA_SHOTS}" \
      "${ALPHA_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage \
      "${ALPHA_ROOT}" \
      "${GLOBAL_ALPHA_SUMMARY}" \
      "stage20_schemeB_alpha_first" \
      "B" \
      "B_ALPHA_TAGS" \
      "B_ALPHA_OPTS" \
      "${ALPHA_DATASETS}" \
      "${ALPHA_SHOTS}" \
      "${ALPHA_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage \
      "${ALPHA_ROOT}" \
      "${GLOBAL_ALPHA_SUMMARY}" \
      "stage30_schemeA_alpha_first" \
      "A" \
      "A_ALPHA_TAGS" \
      "A_ALPHA_OPTS" \
      "${ALPHA_DATASETS}" \
      "${ALPHA_SHOTS}" \
      "${ALPHA_SEEDS}"
  fi

  print_summary_table "${GLOBAL_ALPHA_SUMMARY}" "ALPHA 搜索汇总（第一阶段，仅 16-shot）"
  select_best_alpha_tags
  print_best_table "${ALPHA_BEST_CONFIG_SUMMARY}"
}

run_nmc_search_stage() {
  print_banner "开始第三阶段：在 16-shot 上搜索 N_MC_TRAIN"
  rm -f "${GLOBAL_NMC_SUMMARY}"
  init_nmc_stage_cases

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${NMC_ROOT}" \
      "${GLOBAL_NMC_SUMMARY}" \
      "stage70_schemeC_nmc16" \
      "C" \
      "C_NMC_TAGS" \
      "C_NMC_OPTS" \
      "${NMC_DATASETS}" \
      "${NMC_SHOTS}" \
      "${NMC_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage \
      "${NMC_ROOT}" \
      "${GLOBAL_NMC_SUMMARY}" \
      "stage80_schemeB_nmc16" \
      "B" \
      "B_NMC_TAGS" \
      "B_NMC_OPTS" \
      "${NMC_DATASETS}" \
      "${NMC_SHOTS}" \
      "${NMC_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage \
      "${NMC_ROOT}" \
      "${GLOBAL_NMC_SUMMARY}" \
      "stage90_schemeA_nmc16" \
      "A" \
      "A_NMC_TAGS" \
      "A_NMC_OPTS" \
      "${NMC_DATASETS}" \
      "${NMC_SHOTS}" \
      "${NMC_SEEDS}"
  fi

  print_summary_table "${GLOBAL_NMC_SUMMARY}" "N_MC_TRAIN 搜索汇总（第三阶段，仅 16-shot）"
  select_best_nmc_tags
  print_best_table "${NMC_BEST_CONFIG_SUMMARY}"
}

# ------------------------------------------------------------
# 最终确认：只跑最终 A/B/C best，不跑 baseline
# ------------------------------------------------------------
run_confirm_stage() {
  if [[ "${AUTO_CONFIRM}" != "1" ]]; then
    echo "[跳过] 最终确认阶段"
    return 0
  fi

  if [[ ! -f "${ALPHA_BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到 alpha best 配置文件: ${ALPHA_BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  if [[ ! -f "${BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到主参数 best 配置文件: ${BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  if [[ ! -f "${NMC_BEST_CONFIG_ENV}" ]]; then
    echo "[错误] 找不到 N_MC_TRAIN best 配置文件: ${NMC_BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  # 为了保证 A_NMC_OPTS / B_NMC_OPTS / C_NMC_OPTS 在确认阶段一定可用，重新构造一次
  # shellcheck disable=SC1090
  source "${ALPHA_BEST_CONFIG_ENV}"
  init_search_stage_cases_with_alpha
  init_nmc_stage_cases

  # shellcheck disable=SC1090
  source "${NMC_BEST_CONFIG_ENV}"

  if [[ -z "${A_NMC_BEST_TAG:-}" || -z "${B_NMC_BEST_TAG:-}" || -z "${C_NMC_BEST_TAG:-}" ]]; then
    echo "[错误] N_MC_TRAIN best 结果不完整，请检查: ${NMC_BEST_CONFIG_ENV}" >&2
    exit 1
  fi

  SELECTED_A_TAGS=("${A_NMC_BEST_TAG}")
  SELECTED_B_TAGS=("${B_NMC_BEST_TAG}")
  SELECTED_C_TAGS=("${C_NMC_BEST_TAG}")

  print_banner "开始最终确认：只跑 A/B/C 最终 best"
  echo "确认数据集: ${CONFIRM_DATASETS}"
  echo "确认 shots: ${CONFIRM_SHOTS}"
  echo "确认 seeds: ${CONFIRM_SEEDS}"
  echo "A_final_best: ${A_NMC_BEST_TAG}"
  echo "B_final_best: ${B_NMC_BEST_TAG}"
  echo "C_final_best: ${C_NMC_BEST_TAG}"

  rm -f "${GLOBAL_CONFIRM_SUMMARY}"

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm10_schemeC_final_best" \
      "C" \
      "SELECTED_C_TAGS" \
      "C_NMC_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm20_schemeB_final_best" \
      "B" \
      "SELECTED_B_TAGS" \
      "B_NMC_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage \
      "${CONFIRM_ROOT}" \
      "${GLOBAL_CONFIRM_SUMMARY}" \
      "confirm30_schemeA_final_best" \
      "A" \
      "SELECTED_A_TAGS" \
      "A_NMC_OPTS" \
      "${CONFIRM_DATASETS}" \
      "${CONFIRM_SHOTS}" \
      "${CONFIRM_SEEDS}"
  fi

  print_summary_table "${GLOBAL_CONFIRM_SUMMARY}" "最终确认汇总"
}

# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
main() {
  resolve_stage_defaults
  resolve_tune_defaults
  resolve_confirm_defaults
  init_gpu_list

  SEARCH_ROOT="${OUTPUT_ROOT}/${STAGE}/search_stage"
  NMC_ROOT="${OUTPUT_ROOT}/${STAGE}/nmc_stage"
  ALPHA_ROOT="${OUTPUT_ROOT}/${STAGE}/alpha_stage"
  CONFIRM_ROOT="${OUTPUT_ROOT}/${STAGE}/confirm_stage"

  GLOBAL_SEARCH_SUMMARY="${SEARCH_ROOT}/global_sweep_summary.csv"
  GLOBAL_NMC_SUMMARY="${NMC_ROOT}/global_nmc_summary.csv"
  GLOBAL_ALPHA_SUMMARY="${ALPHA_ROOT}/global_alpha_summary.csv"
  GLOBAL_CONFIRM_SUMMARY="${CONFIRM_ROOT}/confirm_global_summary.csv"
  BEST_CONFIG_SUMMARY="${SEARCH_ROOT}/best_config_summary.csv"
  BEST_CONFIG_ENV="${SEARCH_ROOT}/best_config.env"
  NMC_BEST_CONFIG_SUMMARY="${NMC_ROOT}/best_nmc_config_summary.csv"
  NMC_BEST_CONFIG_ENV="${NMC_ROOT}/best_nmc_config.env"
  ALPHA_BEST_CONFIG_SUMMARY="${ALPHA_ROOT}/best_alpha_config_summary.csv"
  ALPHA_BEST_CONFIG_ENV="${ALPHA_ROOT}/best_alpha_config.env"

  mkdir -p "${SEARCH_ROOT}" "${NMC_ROOT}" "${ALPHA_ROOT}" "${CONFIRM_ROOT}"
  rm -f \
    "${GLOBAL_SEARCH_SUMMARY}" "${BEST_CONFIG_SUMMARY}" "${BEST_CONFIG_ENV}" \
    "${GLOBAL_NMC_SUMMARY}" "${NMC_BEST_CONFIG_SUMMARY}" "${NMC_BEST_CONFIG_ENV}" \
    "${GLOBAL_ALPHA_SUMMARY}" "${ALPHA_BEST_CONFIG_SUMMARY}" "${ALPHA_BEST_CONFIG_ENV}" \
    "${GLOBAL_CONFIRM_SUMMARY}"

  init_scheme_C
  init_scheme_B
  init_scheme_A

  print_banner "BayesMMRL：ALPHA 优先搜索 -> 主参数搜索 -> N_MC_TRAIN -> 最终确认"
  echo "PROJECT_DIR: ${PROJECT_DIR}"
  echo "主参数搜索数据集: ${DATASETS}"
  echo "主参数搜索 shots: ${SHOTS}"
  echo "主参数搜索 seeds: ${SEEDS}"
  echo "ALPHA 搜索数据集: ${ALPHA_DATASETS}"
  echo "ALPHA 搜索 shots: ${ALPHA_SHOTS}"
  echo "ALPHA 搜索 seeds: ${ALPHA_SEEDS}"
  echo "N_MC_TRAIN 搜索数据集: ${NMC_DATASETS}"
  echo "N_MC_TRAIN 搜索 shots: ${NMC_SHOTS}"
  echo "N_MC_TRAIN 搜索 seeds: ${NMC_SEEDS}"
  echo "最终确认数据集: ${CONFIRM_DATASETS}"
  echo "最终确认 shots: ${CONFIRM_SHOTS}"
  echo "最终确认 seeds: ${CONFIRM_SEEDS}"
  echo "GPU: ${GPU_IDS}"
  echo "搜索输出目录: ${SEARCH_ROOT}"
  echo "ALPHA 输出目录: ${ALPHA_ROOT}"
  echo "N_MC_TRAIN 输出目录: ${NMC_ROOT}"
  echo "确认输出目录: ${CONFIRM_ROOT}"
  echo "Backbone: ${BACKBONE}"
  echo "固定评估模式: ${COMMON_EVAL_MODE}"
  echo "固定聚合方式: ${COMMON_EVAL_AGGREGATION}"
  echo "固定 KL warmup: ${COMMON_KL_WARMUP_EPOCHS}"
  echo "默认不跑 baseline: RUN_BASELINE=${RUN_BASELINE}"
  echo "测试后自动删 checkpoint: ${DELETE_CKPT_AFTER_TEST}"
  echo "执行顺序: ALPHA(16-shot) -> 自动选优 -> 主参数搜索(C->B->A) -> 自动选优 -> N_MC_TRAIN(16-shot) -> 自动选优 -> 最终确认(best only)"
  echo "方案 C 候选数: ${#C_TAGS[@]}"
  echo "方案 B 候选数: ${#B_TAGS[@]}"
  echo "方案 A 候选数: ${#A_TAGS[@]}"

  trap 'echo "[中断] 正在停止子任务..."; cleanup_children; exit 130' INT TERM

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    run_baseline_stage \
      "${SEARCH_ROOT}" \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "stage00_baseline_mmrl_fixed" \
      "baseline" \
      "stage00_baseline_mmrl_fixed" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${SEEDS}"
  else
    echo "[跳过] baseline（已按你的要求默认关闭）"
  fi

  # 第一阶段：先搜 alpha
  run_alpha_search_stage

  # 第二阶段：固定 alpha 后再搜主参数
  init_search_stage_cases_with_alpha

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage \
      "${SEARCH_ROOT}" \
      "${GLOBAL_SEARCH_SUMMARY}" \
      "stage40_schemeC_projrep_after_alpha" \
      "C" \
      "C_SEARCH_TAGS" \
      "C_SEARCH_OPTS" \
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
      "stage50_schemeB_clip_joint_prior_after_alpha" \
      "B" \
      "B_SEARCH_TAGS" \
      "B_SEARCH_OPTS" \
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
      "stage60_schemeA_zero_prior_after_alpha" \
      "A" \
      "A_SEARCH_TAGS" \
      "A_SEARCH_OPTS" \
      "${DATASETS}" \
      "${SHOTS}" \
      "${SEEDS}"
  else
    echo "[跳过] 方案 A"
  fi

  print_summary_table "${GLOBAL_SEARCH_SUMMARY}" "主参数搜索汇总（第二阶段，alpha 已固定）"
  select_best_tags
  print_best_table "${BEST_CONFIG_SUMMARY}"

  # 第三阶段：搜 N_MC_TRAIN
  run_nmc_search_stage

  # 第四阶段：最终确认
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