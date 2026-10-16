# Objective Fidelity V5 — 固定训练集 + 测试集双协议诊断

本版本**只修改实验目录**，不改原始模型、训练器、KL、优化器和正式 `static_logit` 推理。以 V4 三方法包为基础，恢复并保留：

| 方法 | 原有 / 实验类 | 用途 |
|---|---|---|
| `original_mc` | 小型 `FidelityOriginalMC(BayesRTMMRLMethod)` 子类 | 原始 Eq. (3) 概率混合目标的 Jensen 分解消融 |
| `branch_mc` | 原生 `BayesRTMMRLMethod` | 两分支 MC 损失参照 |
| `davi` | 原生 `DetBayesRTMMRLMethod` | 确定性矩近似 |

## 最重要的 V5 修正

1. `evaluate.py` **不再**把含随机裁剪、翻转的 `train_loader_x` 直接当参考集。它构造 `train_fixed`：保留 few-shot 训练图像和标签，但**强制使用原始 test loader 的确定性预处理**，顺序固定，无随机增广。`test` 直接取原始正式测试集图像/标签，使用相同确定性预处理。两个 split 都只用于 **checkpoint 离线评估**；不参与训练、超参选择或模型更新。
2. 对 `train_fixed` 和 `test` **分别计算全部三个目标**：`original_nll`、`branch_nll`、`det_nll`，并计算 `mix_gap`、`det_gap`、`total_gap`。每个 gap 都是在**同一个 checkpoint、同一批输入、同一分布**上计算，不能跨方法相减解释为近似误差。
3. 两个 split 都输出 `main_nll`、`rep_nll`、`main_accuracy`、`rep_accuracy`、`static_prob_accuracy`、`static_logit_accuracy`，使用足量的 MC 后验样本。`static_prob` 只用于比较，三种方法**正式测试仍用 `static_logit`**。
4. 每个 split 记录 `input_sha256`（按固定遍历顺序对输入张量和标签哈希）。`summarize.py` 在计算跨 checkpoint 差异前核对同一数据集/seed/split 的哈希。不一致直接报错，不再默认为可比。
5. 图表分别展示训练集和测试集的三目标、Main/Rep、误差分解、融合对照，不会把训练 NLL 当作测试指标。
6. 修正 V4 的一个标注错误：`representation_learner()` 实际返回 **(text tokens, visual tokens)**；旧快照曾把这两类有效提示误标。V5 评估会重新生成正确标注的 `posterior_diagnostic.pt`。

**注意**：`train_fixed` 的固定测试风格预处理与训练时的随机增广不是同一个输入分布。因此 `train_fixed` 指标用于可复现的后验和目标函数诊断，不应描述为“原始训练期随机增强目标的精确平均值”。`test` 参考目标是 held-out 数据上的估计，既不是训练过程使用的目标，也不是原生训练器输出的 `test_nll`。后者按最终 `static_logit` 预测计算。

## 部署（完整覆盖 `experiments/objective_fidelity/`）

将 `objective_fidelity_v5_fixed_train_test.zip` 上传到 `MMRL/` 根目录：

```bash
cd ~/autodl-tmp/MMRL
cp -r experiments/objective_fidelity experiments/objective_fidelity_backup_v4
unzip -o objective_fidelity_v5_fixed_train_test.zip -d experiments/
```

包含 6 个文件：`train.py`、`evaluate.py`、`summarize.py`、`plot_results.py`、`run_study.sh`、`README.md`。

## 用已有 checkpoint 重新评估（无需重训）

已有训练完成标志与 `experiment_manifest.json` 时，会自动跳过训练；**旧版 V4 fidelity.json 会强制重评**，因为数据协议已经改变。

```bash
GPU=0 PROFILE=native DATASETS="caltech101" SEEDS="1" \
  SAMPLES=128 CHUNK=8 REF_BATCH=32 FEATURES=128 \
  bash experiments/objective_fidelity/run_study.sh
```

用于最终论文统计时，再改为较大参考采样数：

```bash
GPU=0 PROFILE=native DATASETS="fgvc_aircraft eurosat food101" SEEDS="1 2 3" \
  SAMPLES=4096 CHUNK=16 REF_BATCH=32 FEATURES=128 \
  bash experiments/objective_fidelity/run_study.sh
```

建议先从少量采样调试并确认可运行，再提高到 4096。**双 split 都进行 MC 评估，总耗时将明显高于 V4**。不建议在最终论文参考数值使用 `--limit-batches`，其输出仅保存为 `fidelity_smoke.json`，不会被汇总。

可以单独调用评估：

```bash
python experiments/objective_fidelity/evaluate.py \
  --mode original_mc --profile native --dataset caltech101 \
  --shot 16 --seed 1 --gpu 0 --samples 128 --chunk 8 \
  --reference-batch-size 32 --features 128
```

## 输出与字段

输出目录沿用 `output_objective_fidelity_v3/`（或自定义 `OUT`），原有三个 checkpoint 保留。

```text
OUT/native/original_mc/caltech101/shot_16/seed1/fidelity.json
OUT/native/branch_mc/caltech101/shot_16/seed1/fidelity.json
OUT/native/davi/caltech101/shot_16/seed1/fidelity.json
OUT/objective_fidelity_three_method_native.csv
OUT/plots/native/fig01_reference_objectives.png
OUT/plots/native/fig02_branch_supervision_train_fixed.png
OUT/plots/native/fig02_branch_supervision_test.png
OUT/plots/native/fig03_gap_decomposition.png
OUT/plots/native/fig04_fusion_diagnostic.png
OUT/plots/native/fig05_test_accuracy_nll.png
...
```

字段命名前缀严格区分 split：

- `train_fixed_original_nll` / `test_original_nll`
- `train_fixed_branch_nll` / `test_branch_nll`
- `train_fixed_det_nll` / `test_det_nll`
- `train_fixed_mix_gap` / `test_mix_gap`
- `train_fixed_det_gap` / `test_det_gap`
- `train_fixed_total_gap` / `test_total_gap`
- `train_fixed_main_nll` / `test_main_nll`，`train_fixed_rep_nll` / `test_rep_nll`
- `train_fixed_main_accuracy` / `test_main_accuracy`，`train_fixed_rep_accuracy` / `test_rep_accuracy`
- `train_fixed_static_prob_accuracy` / `test_static_prob_accuracy`
- `train_fixed_static_logit_accuracy` / `test_static_logit_accuracy`
- `train_fixed_input_sha256` / `test_input_sha256`，`*_num_examples`

`test_nll` 和 `test_accuracy` **不带“reference”语义**，是汇总时从原生训练器正式测试报告读取的指标；不要与新字段 `test_original_nll`、`test_static_logit_accuracy` 混淆（采样数和测试评估策略可能不同）。

三个参考目标在同一个 checkpoint 上分别是：

\[
L_o=\mathbb E_{q}[-\log(\alpha p_c(y)+(1-\alpha)p_r(y))],
\quad
L_b=\alpha\mathbb E_q[-\log p_c(y)]+(1-\alpha)\mathbb E_q[-\log p_r(y)],
\]

\[
L_d=\alpha\,\mathrm{detCE}(\mu_c,v_c)+(1-\alpha)\,\mathrm{detCE}(\mu_r,v_r).
\]

\[
\mathrm{mix\_gap}=L_b-L_o\ge 0,\quad
\mathrm{det\_gap}=L_d-L_b,\quad
\mathrm{total\_gap}=L_d-L_o.
\]

`det_gap`、`total_gap` 在当前近似矩和方差截断下不保证非负。KL 和 alignment 作为共同参考项添加到各目标，但 **test split 的 KL 使用原训练批次数归一化**，不按测试集 batch 数重新定义。只有 NLL 才是严格的 per-example 预测损失。

## 原生配置与控制变量

`PROFILE=native`：DAVI 使用 `R_PRIOR_MODE=random_mmrl`，另外两种 MC 方法使用 `zero`。因此 DAVI 与 Branch-MC 的后验差异不能全部归因于目标函数。

`PROFILE=controlled`：三者统一使用原生两模型都支持的 `self_proj_rep`，并共享先验、KL 等配置。这是有意改变 DAVI 原生初始化的控制变量实验；不要与 native 结果混为一谈。

## 检查结果与限制

提交前进行了 Python 编译、Shell 语法、固定输入 loader 单元测试和模拟前向参考损失单元测试，以及汇总与绘图的离线烟雾测试。**由于当前环境没有 Dassl 和对应 CLIP checkpoint，没有声称运行了 GPU 端到端诊断。** 旧 V4 的训练集 CSV 数据不可直接和新 V5 的 `train_fixed` 数值比较，因为输入增广协议已经改变。
