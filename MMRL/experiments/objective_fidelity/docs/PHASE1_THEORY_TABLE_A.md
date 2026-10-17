# DAVI 第一阶段：目标近似的严格关系与 Table A（Native-only，六项版本）

> 与 `stage1_gap.py`, `stage1_cached.py`, `stage1_evaluate.py` 一致。正式测试必须先在真实 checkpoint 上通过缓存/原生路径一致性检查。

## 1. 符号

在固定的后验 $q(P_r,P_t)$、确定性提示参数 $\Phi$ 和样本 $(x,y)$ 上，令 $Z_m,Z_r$ 是经过真实 L2 归一化的随机类别 logits。两分支的正确类别概率是 $a=\mathrm{softmax}(Z_m)_y$ 与 $b=\mathrm{softmax}(Z_r)_y$；$\pi=(\alpha,1-\alpha)$。

定义七个中间目标（前两个 Jensen 是严格数学界，其余是近似或实现细节）：

1. $L_O=\mathbb E_q[-\log(\alpha a+(1-\alpha)b)]$：论文原始混合分类 NLL。
2. $L_B=\alpha\mathbb E_q[-\log a]+(1-\alpha)\mathbb E_q[-\log b]$：原生 Branch-MC 的分类期望。
3. $L_J=\sum_j\pi_j[\mathrm{LSE}(\log\mathbb E_q e^{Z_j})-\mathbb E_q Z_{j,y}]$：真实 log-MGF 的 Jensen 目标。
4. $L_G=\sum_j\pi_j[\mathrm{LSE}(\mu_j+v_j/2)-\mu_{j,y}]$：真实 logit 均值方差的高斯矩替换；对归一化投影不是精确高斯界。
5. $L_P=\sum_j\pi_j[\mathrm{LSE}(\widetilde\mu_j+\widetilde v_j/2)-\widetilde\mu_{j,y}]$：DAVI 的一阶近似矩，**未**施加缩放与截断。
6. $L_S=\sum_j\pi_j[\mathrm{LSE}(\widetilde\mu_j+\eta\widetilde v_j/2)-\widetilde\mu_{j,y}]$：显式方差系数 $\eta$。
7. $L_D$：再按代码把非负近似方差截断到上限 $C$ 后的实际 DAVI 目标；$C\le0$ 表示关闭截断。

## 2. 三目标关系（Table A 主体），六项恒等式（可选诊断）

$$
L_D-L_O=(L_B-L_O)+(L_J-L_B)+(L_G-L_J)+(L_P-L_G)+(L_S-L_P)+(L_D-L_S).
$$

- **Mixture Jensen gap**：$L_B-L_O=\mathbb E_q[\mathrm{KL}(\pi\Vert r)]\ge0$，$r_j=\pi_j p_j(y)/(\sum_h\pi_h p_h(y))$。这个恒等式直接联系“固定分支权重”和“后验责任权重”。
- **Softmax Jensen gap**：$L_J-L_B\ge0$。令 $T_j=\mathrm{LSE}(Z_j)$，则每个分支间隙是 $\log\mathbb E e^{T_j}-\mathbb E T_j\ge0$。
- **Gaussian replacement error**：$L_G-L_J$，有符号；精确归一化 logits 一般非高斯。
- **Moment propagation error**：$L_P-L_G$，有符号，包含归一化的一阶 Taylor 近似。
- **Scale impact**：$L_S-L_P$，当 $\eta=1$ 为 0；当 $\eta>1$ 且方差非负时非负。
- **Clamp impact**：$L_D-L_S\le0$，需 $\eta\ge0$ 且方差被向下截断。

严格界并**不**意味着 $L_D$ 对 $L_O$ 永远是单侧上界。对最大化的对数似然，对应 Jensen 不等式方向反转；但实现所优化的含缩放 KL 与正则项的总损失不应未经证明直接称为标准负 ELBO。

## 3. 数学界与近似误差的作用范围

$\mathrm{LSE}$ 关于无穷范数是 1-Lipschitz，故

$$ |L_G-L_J|\le\sum_j\pi_j\|\log\mathbb E e^{Z_j}-\mu_j-v_j/2\|_\infty . $$

而一阶矩误差满足条件界

$$ |L_P-L_G|\le\sum_j\pi_j[2\|\widetilde\mu_j-\mu_j\|_\infty+\tfrac12\|\widetilde v_j-v_j\|_\infty]. $$

这是关于**给定矩误差**的界，不是对任意后验都紧的全局误差界。建议同时报告预归一化投影 SNR，识别均值范数很小的一阶近似不可靠区域。

## 4. 统计估计口径

- 只读取 `native/<mode>/<dataset>/shot_<K>/seed<N>/` 的现有 checkpoint；不改训练。
- 固定 `train_fixed`（few-shot 训练图像 + 测试式确定性预处理）和 `test`（正式 held-out 图像）。这两个离线诊断 **均不是训练过程的原始随机增强目标**；test 也不是训练 ELBO。
- 对固定 batch 缓存文本/视觉深度提示编码，复用原始 `sample_many()` 完成真实投影采样和 L2 归一化；在第一批次进行原生与缓存前向的随机数路径校验。
- 同一 checkpoint 上用 S=64,256,1024,4096 等嵌套采样规模；**不同重复各自独立采样**，输出均值、重复间标准差、MC 标准误及正态近似的 95% 区间半宽（这不是有限-S 偏差的置信保证）。
- **Total signed gap** 是 $\mathbb E_x[\ell_D(x)-\ell_O(x)]$；**Per-image Gap MAE** 是 $\mathbb E_x|\ell_D(x)-\ell_O(x)|$，不能用 $|\mathbb E_x[\ell_D-\ell_O]|$ 代替；另报逐样本绝对误差 P90 和低估比例。
- 每项 gap 在同一 checkpoint 内相减；不能用 DAVI 与 Branch-MC **不同 checkpoint** 的 NLL 差称作近似误差。
- 汇总以**样本数量**加权；训练随机种子差异（seed SD）与固定 checkpoint 的 MC 标准误分别计算；遇到缓存一致性、哈希或数学恒等式失败直接报错。

## 5. 第一阶段正式交付

Table A 主体：同一 checkpoint 的 Original NLL、Branch NLL、DAVI NLL、Mix Gap (=B−O)、Det Gap (=D−B)、Total Gap (=D−O)、逐图像 E|D−B|。详细六项分解保留在 raw CSV 和附录。主图：Det Gap / E|D−B| 与近似投影 logit 方差的观测关联；收敛图：固定 checkpoint 下随 S 的 Det Gap。所有图必须从真实 CSV 绘制。

## 6. 运行

```bash
cd MMRL
python -m unittest discover -s experiments/objective_fidelity/tests -p 'test_stage1_*.py' -v
GPU=0 DATASETS='caltech101' SEEDS='1' MODES='davi branch_mc' SAMPLES=128 REPEATS=1 GRID='32,128' bash experiments/objective_fidelity/run_phase1_native.sh
# 正式实验：S=4096，R=1 为默认；对 MC 稳定性存疑可设置 R=3，Aircraft / EuroSAT / Food101 三种子（计算开销明显较高）
```

**注意**：本环境 CPU 测试可以核验公式和脚本输出，但完整 CLIP/Dassl/GPU checkpoint 仍需要在实验机器上通过生产路径对照。不能从仅有的单次有限-S 样本推出近似全局紧致。
