#!/bin/bash
set -euo pipefail

# ============================================================
# BayesMMRL 顺序化搜索脚本（新接口版，中文注释）
#
# 设计目标：
# 1. 不写入 output_refactor，统一写到 output_sweeps
# 2. 自动执行顺序：baseline -> 方案C -> 方案B -> 方案A
# 3. 终端持续打印“当前在干什么”
# 4. 目录命名清晰，方便后续筛选
# ============================================================

# ---------------------------
# 用户可改环境变量
# ---------------------------
ROOT=${ROOT:-DATASETS}
PROTOCOL=${PROTOCOL:-FS}
EXEC_MODE=${EXEC_MODE:-online}
BACKBONE=${BACKBONE:-ViT-B/16}

# 阶段：
#   coarse3 -> 先用 3 个数据集筛方向
#   full11  -> 再上 11 个数据集
STAGE=${STAGE:-coarse3}

# 如果手动指定 DATASETS/SHOTS/SEEDS，则覆盖默认
DATASETS=${DATASETS:-}
SHOTS=${SHOTS:-}
SEEDS=${SEEDS:-}

# 输出目录：单独走 output_sweeps，避免污染正式结果
OUTPUT_ROOT=${OUTPUT_ROOT:-output_sweeps/bayes_mmrl_search_seq_v2}

# GPU 调度
NGPU=${NGPU:-2}
GPU_IDS=${GPU_IDS:-"0 1"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
SLEEP_SEC=${SLEEP_SEC:-2}

# 是否先跑固定论文配置的 MMRL baseline
RUN_BASELINE=${RUN_BASELINE:-1}

# 是否执行各方案
RUN_SCHEME_C=${RUN_SCHEME_C:-1}
RUN_SCHEME_B=${RUN_SCHEME_B:-1}
RUN_SCHEME_A=${RUN_SCHEME_A:-1}

# ---------------------------
# 固定使用 MMRL 论文提供的最佳配置
# 不再当作 sweep 维度搜索
# ---------------------------
PAPER_ALPHA=0.7
PAPER_REG_WEIGHT=0.5
PAPER_N_REP_TOKENS=5
PAPER_REP_LAYERS='[6,7,8,9,10,11,12]'
PAPER_REP_DIM=512

# 统一的 MC 设置
COMMON_N_MC_TRAIN=3
COMMON_N_MC_TEST=10

# 新接口：正式 sweep 固定使用 mc_predictive
COMMON_EVAL_MODE=mc_predictive

METHOD_CFG="configs/methods/bayesmmrl.yaml"
RUNTIME_CFG="configs/runtime/default.yaml"

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
# 根据阶段自动给出默认数据集 / shots / seeds
# ------------------------------------------------------------
resolve_stage_defaults() {
  if [[ -z "${DATASETS}" ]]; then
    case "$STAGE" in
      coarse3)
        DATASETS="caltech101 oxfordpets ucf101"
        ;;
      full11)
        DATASETS="caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxfordpets sun397 stanford_cars ucf101"
        ;;
      *)
        echo "未知 STAGE=${STAGE}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ -z "${SHOTS}" ]]; then
    case "$STAGE" in
      coarse3) SHOTS="1 4 16" ;;
      full11)  SHOTS="1 4 16" ;;
      *)
        echo "未知 STAGE=${STAGE}" >&2
        exit 1
        ;;
    esac
  fi

  if [[ -z "${SEEDS}" ]]; then
    case "$STAGE" in
      coarse3) SEEDS="1 2" ;;
      full11)  SEEDS="1 2 3" ;;
      *)
        echo "未知 STAGE=${STAGE}" >&2
        exit 1
        ;;
    esac
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
build_outdir() {
  local method=$1
  local dataset=$2
  local shot=$3
  local tag=$4
  local seed=$5
  local backbone_tag="${BACKBONE//\//-}"
  echo "${OUTPUT_ROOT}/${STAGE}/${method}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/${backbone_tag}/${tag}/seed${seed}"
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
    echo "输出根目录: ${OUTPUT_ROOT}"
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
# 汇总表
# ------------------------------------------------------------
GLOBAL_SUMMARY=""

append_summary_to_global() {
  local summary_csv=$1
  local stage_name=$2
  local scheme=$3
  local dataset=$4
  local shot=$5
  local tag=$6

  local header
  header="$(head -n 1 "${summary_csv}")"

  if [[ ! -f "${GLOBAL_SUMMARY}" ]]; then
    echo "stage_name,scheme,dataset,shot,tag,${header}" > "${GLOBAL_SUMMARY}"
  fi

  tail -n +2 "${summary_csv}" | while IFS= read -r line; do
    echo "${stage_name},${scheme},${dataset},${shot},${tag},${line}" >> "${GLOBAL_SUMMARY}"
  done
}

summarize_one_tag() {
  local method=$1
  local stage_name=$2
  local scheme=$3
  local dataset=$4
  local shot=$5
  local tag=$6

  local backbone_tag tag_root summary_csv
  backbone_tag="${BACKBONE//\//-}"
  tag_root="${OUTPUT_ROOT}/${STAGE}/${method}/${PROTOCOL}/${PHASE}/${dataset}/shots_${shot}/${backbone_tag}/${tag}"

  python evaluation/result_parser.py "${tag_root}" --split test >/dev/null 2>&1 || true

  summary_csv="${tag_root}/test_summary.csv"
  if [[ -f "${summary_csv}" ]]; then
    append_summary_to_global "${summary_csv}" "${stage_name}" "${scheme}" "${dataset}" "${shot}" "${tag}"
  fi
}

print_global_summary() {
  echo
  echo "============================================================"
  echo "全局汇总文件:"
  echo "${GLOBAL_SUMMARY}"
  echo "============================================================"

  if [[ -f "${GLOBAL_SUMMARY}" ]]; then
    python - <<PY
import pandas as pd
from pathlib import Path

path = Path(r"${GLOBAL_SUMMARY}")
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
    view = df[keep].sort_values(
        by=["stage_name", "dataset", "shot", "accuracy_mean"],
        ascending=[True, True, True, False],
    )
    print(view.to_string(index=False))
else:
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
  if [[ "${RUN_BASELINE}" != "1" ]]; then
    echo "[跳过] baseline 阶段"
    return 0
  fi

  local stage_name="stage00_baseline_mmrl_paper_fixed"
  print_banner "开始执行 ${stage_name}：固定论文配置的 MMRL baseline"

  init_slots

  local dataset shot seed slot gpu_id outdir logfile statusfile desc
  for dataset in ${DATASETS}; do
    for shot in ${SHOTS}; do
      for seed in ${SEEDS}; do
        outdir="$(build_outdir "MMRL" "${dataset}" "${shot}" "${stage_name}" "${seed}")"
        logfile="${outdir}/run.log"
        statusfile="${outdir}/job_status.txt"

        if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
          mkdir -p "${outdir}"
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
          write_log_header "${logfile}" "${gpu_id}" "MMRL" "${dataset}" "${shot}" "${seed}" "${stage_name}" "${stage_name}"

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
  for dataset2 in ${DATASETS}; do
    for shot2 in ${SHOTS}; do
      summarize_one_tag "MMRL" "${stage_name}" "baseline" "${dataset2}" "${shot2}" "${stage_name}"
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
  # 方案 C：Bayes 放在 proj_rep
  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-row_kl-1e-7_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-7 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

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
    "stage10_C_projrep_sig-row_kl-5e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE row \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"

  register_case C_TAGS C_OPTS \
    "stage10_C_projrep_sig-global_kl-1e-7_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 1e-7 \
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
    "stage10_C_projrep_sig-global_kl-5e-6_prior-0.01_eval-mcpred" \
    BAYES_MMRL.BAYES_TARGET proj_rep \
    BAYES_MMRL.PROJ_REP_SIGMA_MODE global \
    BAYES_MMRL.PROJ_REP_KL_WEIGHT 5e-6 \
    BAYES_MMRL.PROJ_REP_PRIOR_STD 0.01 \
    BAYES_MMRL.PROJ_REP_INIT_MODE pretrained_mean \
    BAYES_MMRL.PROJ_REP_INIT_STD 0.0 \
    BAYES_MMRL.EVAL_MODE "${COMMON_EVAL_MODE}"
}

init_scheme_B() {
  # 方案 B：Bayes 放在 R，并使用 CLIP 先验
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
}

init_scheme_A() {
  # 方案 A：Bayes 放在 R，零中心先验
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
}

# ------------------------------------------------------------
# 统一执行某一个方案阶段
# ------------------------------------------------------------
run_scheme_stage() {
  local stage_name=$1
  local scheme_name=$2
  local tags_array_name=$3
  local opts_assoc_name=$4

  print_banner "开始执行 ${stage_name}：方案 ${scheme_name}"

  init_slots

  local dataset shot tag seed slot gpu_id outdir logfile statusfile desc opts_str
  eval "local tags=(\"\${${tags_array_name}[@]}\")"

  for dataset in ${DATASETS}; do
    for shot in ${SHOTS}; do
      for tag in "${tags[@]}"; do
        eval "opts_str=\"\${${opts_assoc_name}[${tag}]}\""
        read -r -a EXTRA_OPTS <<< "${opts_str}"

        for seed in ${SEEDS}; do
          outdir="$(build_outdir "BayesMMRL" "${dataset}" "${shot}" "${tag}" "${seed}")"
          logfile="${outdir}/run.log"
          statusfile="${outdir}/job_status.txt"

          if [[ "${SKIP_EXISTING}" == "1" && -f "${outdir}/test_metrics.json" ]]; then
            mkdir -p "${outdir}"
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
            write_log_header "${logfile}" "${gpu_id}" "BayesMMRL" "${dataset}" "${shot}" "${seed}" "${tag}" "${stage_name}"

            echo "[当前执行] ${desc}"
            echo "[说明] 当前使用固定 MMRL 论文超参，只搜索 Bayes 新引入的参数"
            echo "[说明] 当前评估模式固定为 ${COMMON_EVAL_MODE}"
            echo "[说明] 当前输出目录为独立 sweep 目录，不会写入 output_refactor"

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
  for dataset2 in ${DATASETS}; do
    for shot2 in ${SHOTS}; do
      for tag2 in "${tags[@]}"; do
        summarize_one_tag "BayesMMRL" "${stage_name}" "${scheme_name}" "${dataset2}" "${shot2}" "${tag2}"
      done
    done
  done
}

# ------------------------------------------------------------
# 主流程：
# 自动执行顺序：
#   1) baseline
#   2) 方案 C
#   3) 方案 B
#   4) 方案 A
# ------------------------------------------------------------
main() {
  resolve_stage_defaults
  init_gpu_list

  mkdir -p "${OUTPUT_ROOT}/${STAGE}"
  GLOBAL_SUMMARY="${OUTPUT_ROOT}/${STAGE}/global_sweep_summary.csv"
  rm -f "${GLOBAL_SUMMARY}"

  init_scheme_C
  init_scheme_B
  init_scheme_A

  print_banner "BayesMMRL 顺序化搜索开始（新接口版）"
  echo "阶段: ${STAGE}"
  echo "数据集: ${DATASETS}"
  echo "shots: ${SHOTS}"
  echo "seeds: ${SEEDS}"
  echo "GPU: ${GPU_IDS}"
  echo "输出目录: ${OUTPUT_ROOT}"
  echo "Backbone: ${BACKBONE}"
  echo "评估模式(固定): ${COMMON_EVAL_MODE}"
  echo "执行顺序: baseline -> 方案C -> 方案B -> 方案A"

  trap 'echo "[中断] 正在停止子任务..."; cleanup_children; exit 130' INT TERM

  run_baseline_stage

  if [[ "${RUN_SCHEME_C}" == "1" ]]; then
    run_scheme_stage "stage10_schemeC_projrep_first" "C" "C_TAGS" "C_OPTS"
  else
    echo "[跳过] 方案 C"
  fi

  if [[ "${RUN_SCHEME_B}" == "1" ]]; then
    run_scheme_stage "stage20_schemeB_rep_clip_prior" "B" "B_TAGS" "B_OPTS"
  else
    echo "[跳过] 方案 B"
  fi

  if [[ "${RUN_SCHEME_A}" == "1" ]]; then
    run_scheme_stage "stage30_schemeA_rep_zero_prior" "A" "A_TAGS" "A_OPTS"
  else
    echo "[跳过] 方案 A"
  fi

  print_global_summary

  if [[ "${FAILED_JOBS}" -gt 0 ]]; then
    echo
    echo "[结束] 有 ${FAILED_JOBS} 个任务失败，请检查对应 run.log"
    exit 1
  fi

  echo
  echo "[结束] 所有任务执行完成"
}

main "$@"