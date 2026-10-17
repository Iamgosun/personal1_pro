# 修订说明（第一阶段核心版）

保持完整 V5 目录、三种 native 训练方式、原评估/绘图与现有 checkpoint 不变，避免增加与第一阶段目标无关的工程化实验。

本次主要修订：

1. `stage1_gap.py`：保持真实分支混合、分支期望、生产 DAVI 矩代理的同 checkpoint 比较。新增 `E|D-B|` 和 P90；可选六项分解仍在 raw 诊断中，不占主表。
2. `stage1_make_table.py`：主表只报告 Original/Branch/DAVI NLL、Mix/Det/Total Gap 与逐图像 `E|D-B|`；严格验证同一 checkpoint 内部的恒等式及跨模型共同输入。附加六项与 SNR 留在 raw CSV。
3. `stage1_plot.py`：第一张图聚焦 signed Det Gap / `E|D-B|` 与投影方差的观测关联；收敛图按**同一 checkpoint**画 Det Gap 对 S 的变化，不混合训练种子 SD 和 MC 重复误差。
4. `stage1_evaluate.py`、`run_phase1_native.sh`：MC repeats 默认 1；如有疑问可选 repeats=3。保留已实现的缓存一致性验证和生产损失核验，不改变模型公式。
5. `docs` 与 `README`：明确六项为诊断、三目标为主问题，且 Taylor+方差截断不自动获得严格上界。

**未修改** `MMRL/methods/` 或配置（DAVI native: `random_mmrl`，`DET_VAR_CLAMP=50`；原生 `N_MC_TEST=10`，与论文报告使用的 MC 测试样本数可能不同）。没有声明已完成真实 GPU 验证。


## 输出路径与绘图修复

- `run_phase1_native.sh` 将汇总、表格与图片集中写入 `OUT/phase1_summary/<split>/`，不再放在 `MMRL/` 根目录；每个原生 checkpoint 的 `phase1/*.json` 位置不变。
- `stage1_make_table.py` 相对 `OUT` 的解析以 `MMRL/` 根目录为基准；默认输出目录与 `run_phase1_native.sh` 一致。
- `stage1_plot.py` 改为全部 ASCII 英文标签，强制 PDF 嵌入 Type 42/TrueType 字体，同时生成 PNG；单一采样规模不绘制收敛曲线。
- 可以设置 `RENDER_ONLY=1` 从已有 JSON 重新汇总绘图，不加载模型、不重新跑 GPU，不改变 O/B/D 三目标与任何 Gap 的计算。
