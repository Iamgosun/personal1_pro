# Objective Fidelity V3 — DAVI / Branch-MC / Original-MC

本版本仅新增一种训练目标，**不重写 Branch-MC**，也不修改主仓库文件：

| 模式 | 实际方法类 | 实际模型类 | 损失 |
|---|---|---|---|
| `davi` | 原版 `DetBayesRTMMRLMethod` | 原版 `DetBayesRTMMRLModel` | 确定性 Jensen–moment |
| `branch_mc` | **原版 `BayesRTMMRLMethod`** | **原版 `BayesRTMMRLModel`** | 分支 MC CE |
| `original_mc` | 仅新增 `FidelityOriginalMC(BayesRTMMRLMethod)` | **原版 `BayesRTMMRLModel`** | Eq.(3) 概率混合 MC NLL |

`FidelityOriginalMC` 只通过在已有 `losses['total']`、`losses['data_term']` 上加上 `original_NLL - branch_NLL` 更换训练数据项。现有 KL、cosine alignment、AMP、优化器、调度、checkpoint、test 完整复用。

## 重要：native 和 controlled 是两种不同问题

### `PROFILE=native`（默认，忠实使用各方法原始 YAML）

- DAVI → `configs/methods/det_bayesrt_mmrl_clamp.yaml`，其 `R_PRIOR_MODE=random_mmrl`，`DET_VAR_SCALE=1.0`，`DET_VAR_CLAMP=50.0`。
- Branch-MC 和 Original-MC → `configs/methods/bayesrt_mmrl.yaml`，其 `R_PRIOR_MODE=zero`，`N_MC_TRAIN=50`。
- 所有方法均由原生类构建，**没有 `FidelityBranchMC`**。
- 只统一 16-shot / dataset / seed / 运行时代码；为避免 HPO 导致额外实验，明确禁用 HPO。
- 由于 DAVI 和 AVI **初始化、prior_std、KL weights 不同**，三种模型在 native 模式的最终损失差异不能单独归因于近似目标。`original_nll`、后验指标可以描述实际结果；不同方法的 `original_reference_loss` 不是同一个带 KL 权重的目标，汇总不会计算它们的跨方法差值。
- 每一个 checkpoint 内的三种 loss/gap 使用同一批数据与同一批 MC 采样，仍然是有意义的 approximation-fidelity 诊断。

### `PROFILE=controlled`（可选，真正控制变量）

- 三种方法均读取 **同一份** `det_bayesrt_mmrl_clamp.yaml`。
- 唯一必要的初始化覆盖：`R_PRIOR_MODE=self_proj_rep`，这是 DAVI 和原版 AVI **都原生支持**的模式，无需增加模型子类。后验/先验初始化均使用 `proj_rep`，避免零均值矩传播退化。
- CLAMP 仍为 50.0，先验尺度、KL 权重、学习率等对齐。MC 的采样训练次数按 YAML 默认 3；建议正式训练时设 `MC_TRAIN=50`，DAVI 训练不使用 MC。
- 这是科学上更合适的 **controlled objective comparison**，但因 DAVI 不再使用主实验的 `random_mmrl` 初始化，**不能当作原 DAVI 主实验的完全复现**。

**不要将 `random_mmrl` 直接赋给原版 `BayesRTMMRLModel`**：它本来不支持该模式。这个版本完全没有这样做。

## 文件部署

解压文件夹并重命名为 `objective_fidelity`，放在如下位置：

```text
MMRL/experiments/objective_fidelity/
  train.py
  evaluate.py
  summarize.py
  run_study.sh
  README.md
```

使用 `python experiments/objective_fidelity/train.py ...` 启动。脚本自动识别 MMRL 项目根目录，所以不受 Shell 当前工作目录影响。

## 最小测试

```bash
cd ~/autodl-tmp/MMRL
python experiments/objective_fidelity/train.py --mode davi --profile native \
  --dataset caltech101 --shot 16 --seed 1 --epochs 1 --gpu 0
python experiments/objective_fidelity/evaluate.py --mode davi --profile native \
  --dataset caltech101 --shot 16 --seed 1 --samples 16 --chunk 4 \
  --limit-batches 1 --gpu 0
```

`--limit-batches` 仅用于接口/显存冒烟测试，生成 `fidelity_smoke.json`，**不生成正式汇总输入**。需要正式 `fidelity.json`，去掉 `--limit-batches`。

## 完整运行

原生配置（原主实验方法实现不修改）：

```bash
GPU=0 PROFILE=native DATASETS="fgvc_aircraft eurosat food101" SEEDS="1 2 3" \
  bash experiments/objective_fidelity/run_study.sh
```

控制变量配置（建议用于训练目标因果归因）：

```bash
GPU=0 PROFILE=controlled MC_TRAIN=50 DATASETS="fgvc_aircraft eurosat food101" SEEDS="1 2 3" \
  bash experiments/objective_fidelity/run_study.sh
```

只测试 Caltech101 一个 seed：

```bash
GPU=0 PROFILE=controlled MC_TRAIN=50 DATASETS="caltech101" SEEDS="1" SAMPLES=128 CHUNK=8 \
  bash experiments/objective_fidelity/run_study.sh
```

`run_study.sh` 对每个设置依次训练并诊断；若发现已经完成的训练（有 `elapsed_wall_seconds.txt`）或诊断（有 `fidelity.json`），则自动跳过，便于故障后续跑。直接调用 `train.py` 遇到已有输出目录仍会报错以防覆盖。要重做请备份/移开旧目录，或指定新 `OUT`。

## 数学定义（同一个 checkpoint）

采样索引 `s` 和图像索引 `i`，标签 `y_i`：

```text
Original NLL = mean_(s,i) [ -log(alpha * p_c(y_i|s) + (1-alpha)*p_r(y_i|s)) ]
Branch NLL   = mean_(s,i) [ -alpha*log p_c(y_i|s) - (1-alpha)*log p_r(y_i|s) ]
DAVI NLL     = alpha * det_jensen_ce(mu_c,var_c,y) + (1-alpha)*det_jensen_ce(mu_r,var_r,y)
mixture_gap  = Branch NLL - Original NLL >= 0
moment_gap   = DAVI NLL - Branch NLL   (可正可负)
total_gap    = DAVI NLL - Original NLL
```

`det_jensen_ce` 总是使用 **DAVI 的 CLAMP 配置 (eta=1.0, clamp=50.0)**，不使用其函数默认 `0.5/20.0`，且不按 AVI YAML 中的默认 clamp=0.0 更换诊断定义。

`original_reference_loss` / `branch_reference_loss` / `deterministic_reference_loss` = 对应 NLL + alignment + 当前方法自带归一化 KL（以 KL beta=1 报告）。该 KL 部分不是新的 ELBO 公式；native profile 的不同方法使用各自 KL 超参数，所以 **reference loss 跨方法不可直接作为等目标优化比较**。

其它指标：`r_trace`、`t_trace` 为后验协方差迹；`visual_mean_distance_to_original_mc` 为视觉后验均值的相对 Frobenius 距离；`train_and_test_wall_seconds` 包括初始化与测试时间，不是严格 matched-time 优化预算。

## 配置与 checkpoint 可靠性

- 训练存 `effective_config.yaml`、`experiment_manifest.json`（含配置 SHA256）和原训练器的 `refactor_model` checkpoint。
- 评估从 manifest 原路径重建配置并校验 SHA256；配置被改动后 **拒绝**用新参数评价旧 checkpoint。
- 评估直接用该次模式对应的方法类构造模型，并核对关键后验 tensor 与 checkpoint 完全相同。
- 对 AVI/Original-MC 的 **DAVI moments 诊断**，通过调用已经存在的 `DetBayesRTMMRLModel.forward_joint_moments(model, images)` 复用矩公式；使用的仍是原生 AVI 模型及已加载参数，并未替换 checkpoint 的模型类，也不会影响训练。
- 同一 batch 内三个目标共享图像和 MC 样本。跨 checkpoint 的训练 loader 使用同 seed；不能未经核验断言逐像素增强完全一致。
- 本实验为相同 epoch 训练；**没有实现 matched-wall-clock**。

## 核心文件职责

- `train.py`：仅注册 `FidelityOriginalMC`，按 mode/profile 构建原仓库 args，调用 `run.main`。
- `evaluate.py`：按原生类构建/加载模型，在同一 checkpoint 算三种目标与后验统计。
- `summarize.py`：生成分 profile CSV；仅在 controlled 模式计算完整参考目标的跨方法差。
- `run_study.sh`：依次运行训练、诊断和汇总。

**运行验证状态：** 代码在本地通过 Python 语法、shell 语法和 Original-MC 目标数值测试；尚未在具有 Dassl、CLIP 权重及数据集的 GPU 环境完成端到端训练。
