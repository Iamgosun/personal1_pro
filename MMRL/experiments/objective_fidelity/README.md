# DAVI Objective Fidelity V5 — 第一阶段核心修订（native-only）

此文件夹**完整保留 V5 的六个入口文件**：`train.py`、`evaluate.py`、`summarize.py`、`plot_results.py`、`run_study.sh`、`README.md`。保留 `original_mc`、`branch_mc`、`davi` 三种训练方法及原 V5 八类图，**不修改** `MMRL/methods/`、`MMRL/core/`、模型配置、先验、KL、优化器、checkpoint 或原来的 `fidelity.json`。

## 第一阶段到底要回答什么

在**同一个已经训练好的 native checkpoint** 及同一批输入上，计算三个分类目标：

\[
L_O=\mathbb E_{q(P_r,P_t)}[-\log(\alpha p_m(y)+(1-\alpha)p_r(y))],
\quad L_B=\alpha\mathbb E_q[-\log p_m(y)]+(1-\alpha)\mathbb E_q[-\log p_r(y)],
\]
\[
L_D=\alpha\operatorname{detCE}(\widetilde\mu_m,\widetilde v_m)
+(1-\alpha)\operatorname{detCE}(\widetilde\mu_r,\widetilde v_r).
\]

核心只报告：

- **Mix Gap** = `Branch - Original`，严格非负，反映第一次 Jensen 松弛。
- **Det Gap** = `DAVI - Branch`，允许正负，反映 DAVI 实际确定性代理相对原生 Branch 目标的净误差。
- **Total Gap** = `DAVI - Original`，允许正负。再报告逐图像的 `E|DAVI-Branch|` 避免正负抵消。

**特别重要**：两个严格 Jensen 不等式不代表代码中的 DAVI（归一化的 Taylor 近似 + `DET_VAR_CLAMP=50`）必然对原始 AVI 构成严格上界。附录的六项分解仅用来定位异常，**不是论文主表的六个中心问题**。

## 与真实模型和配置的关系

- 使用你仓库的 **native 配置**：DAVI 的 `R_PRIOR_MODE=random_mmrl`；Branch/Original 为 `zero`；两侧 `SIGMA_MODE=output`；`T_TRAIN_MEAN=False`；`ALPHA=0.7`；`DET_VAR_SCALE=1.0`、`DET_VAR_CLAMP=50.0`。
- 采样使用原始 `text_sample_features`、`rep_samples`、L2 归一化和共享文本投影样本。确定性部分复用原生 `forward_joint_moments` 的数学函数；在第一批数据中验证缓存前向的 logits/矩与原生前向是否一致，逐批核对 `det_jensen_ce`。
- **第一阶段不训练新模型。** `stage1_evaluate.py` 从现有 `output_objective_fidelity_v3/native/<mode>/<dataset>/shot_<K>/seed<N>/` 读取 manifest 和 checkpoint，不会修改训练结果。原 V5 实验的 `--profile native` 兼容保留，`controlled` 不支持。
- `train_fixed` 为训练图片但统一应用测试式确定性预处理；`test` 为正式 held-out 数据。均不是随机增广的训练 ELBO，也不同于 `static_logit` 推理指标。KL 和 cosine alignment 在同一 checkpoint 的三目标差分中相同而抵消，故 Table A 仅比较分类 NLL。

## 直接替换

将 ZIP 放在 `MMRL/` 下并备份旧目录：

```bash
mv experiments/objective_fidelity experiments/objective_fidelity_backup_$(date +%Y%m%d_%H%M%S)
unzip -o DAVI_objective_fidelity_Phase1_CORE_FIXED.zip -d experiments/
```

## 推荐顺序（只需两步）

首先执行 CPU 数值单元测试；此项不要求你重训模型：

```bash
python -m unittest discover -s experiments/objective_fidelity/tests -p 'test_stage1_*.py' -v
```

然后使用已有 Caltech101 seed1 **native checkpoint** 进行低成本检查：

```bash
GPU=0 DATASETS='caltech101' SEEDS='1' MODES='davi branch_mc' \
SAMPLES=128 REPEATS=1 GRID='32,128' \
bash experiments/objective_fidelity/run_phase1_native.sh
```

通过后，可用相同模型增大参考 MC 样本数，并逐步扩充数据集/seed：

```bash
GPU=0 DATASETS='caltech101' SEEDS='1' MODES='davi branch_mc' \
SAMPLES=4096 REPEATS=1 GRID='64,256,1024,4096' \
bash experiments/objective_fidelity/run_phase1_native.sh
```

论文报告建议在确认 S 收敛后使用 Aircraft、EuroSAT、Food101 × seed1,2,3。若怀疑小概率 tail 导致 MC 不稳定，可设 `REPEATS=3`，但这不是必要的默认实验。**不同 S 及 repeats 保存为不同文件**，不会覆盖旧 `fidelity.json`。已有相同文件时会拒绝静默覆盖。

所有汇总输出统一位于 `OUT/phase1_summary/<split>/`，不再散落在 `MMRL/` 根目录。目录内包含：

- `table_a_native_<split>.csv`、`.md`：主表。
- `table_a_native_<split>_raw.csv`：完整数值诊断。
- `table_a_native_<split>_convergence.csv`：不同 MC 样本数的汇总。
- `figures/figure_A_det_gap_vs_variance.pdf` 和 `.png`：Det Gap 与近似 logit 方差的关系。
- 只有存在至少两个不同的 MC 采样数，才绘制 `figures/figure_A_mc_convergence_<dataset>.pdf` 和 `.png`。

每个 checkpoint 的原始 JSON 仍保存在 `OUT/native/<mode>/<dataset>/shot_<K>/seed<N>/phase1/`。图中全部采用 ASCII 英文标签，PDF 嵌入 TrueType 字体，并同时输出 PNG，避免部分服务器或 PDF 阅读器的符号乱码问题。

**无需重跑 GPU 的重新汇总/绘图：** 若已有同一 S/R 的 `stage1_*.json`，替换代码后只执行：

```bash
OUT=output_objective_fidelity_v3 DATASETS=caltech101 SEEDS=1 MODES='branch_mc davi' \
SAMPLES=10 REPEATS=1 RENDER_ONLY=1 \
bash experiments/objective_fidelity/run_phase1_native.sh
```

上面的 `SAMPLES`、`REPEATS` 必须与原 JSON 文件名一致。原有散落在 `MMRL/` 根目录的旧 CSV/Markdown/图不会自动删除，可在核对新输出后自行清理。

## 结果的学术边界

同一 checkpoint 的 Gap 是目标近似关系的证据；不同 checkpoint 之间的 Gap 大小还受 learned posterior 所在区域影响。跨方法的后验距离、梯度、深度提示与预测校准属于下一阶段，不应该由 Table A 单独推出因果结论。

本包已通过本地数学和脚本测试，但**尚未在你的完整 Dassl/CLIP/GPU/checkpoint 环境中运行**；真实运行中的 cache/native parity 和生产损失比较不通过时，评估会立即停止。
