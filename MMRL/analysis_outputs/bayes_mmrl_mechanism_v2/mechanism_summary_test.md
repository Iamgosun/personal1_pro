# BayesMMRL Mechanism Diagnosis 汇总分析报告

- Generated at: `2026-04-26 15:01:54`
- Split: `test`
- Save dir: `/root/autodl-tmp/MMRL/analysis_outputs/bayes_mmrl_mechanism_v2`

## 1. 运行与配置审计

| case_name                         | method    | method_from_path   | dataset    |   shots |   seed | protocol   | phase         | tag                                    | output_family   | cfg.TRAINER.NAME   | cfg.METHOD.NAME   | cfg.MODEL.BACKBONE.NAME   | BAYES_TARGET   |   N_MC_TEST | EVAL_MODE     | EVAL_AGGREGATION   | REP_SIGMA_MODE   | REP_PRIOR_MODE   |   REP_PRIOR_STD |   REP_KL_WEIGHT | REP_LAYERS               |
|:----------------------------------|:----------|:-------------------|:-----------|--------:|-------:|:-----------|:--------------|:---------------------------------------|:----------------|:-------------------|:------------------|:--------------------------|:---------------|------------:|:--------------|:-------------------|:-----------------|:-----------------|----------------:|----------------:|:-------------------------|
| MMRL_caltech101_16shot_seed1      | BayesMMRL | BayesMMRL          | caltech101 |      16 |      1 | FS         | fewshot_train | rep_zero_sig-diagonal_pstd-1.0_kl-5e-2 | output_sweeps   | RefactorRunner     | BayesMMRL         | ViT-B/16                  | rep_tokens     |          10 | mc_predictive | logit_mean         | diagonal         | zero             |            1    |          0.05   | [6, 7, 8, 9, 10, 11, 12] |
| BayesMMRL_caltech101_16shot_seed1 | MMRL      | MMRL               | caltech101 |      16 |      1 | FS         | fewshot_train | default                                | output_refactor | RefactorRunner     | MMRL              | ViT-B/16                  | rep_tokens     |           5 | mc_predictive | prob_mean          | global           | zero             |            0.05 |          0.0005 | [6, 7, 8, 9, 10, 11, 12] |

## 2. 总体指标

- **BayesMMRL_caltech101_16shot_seed1** (`MMRL`): acc=0.9700, nll=0.1035, brier=0.0497, ece=0.0144, conf=0.9768, entropy=0.0732, margin=0.9586, js_main_rep=0.0175, symkl_main_rep=0.1296
- **MMRL_caltech101_16shot_seed1** (`BayesMMRL`): acc=0.9615, mc_acc=0.9692, nll=0.0987, brier=0.0519, ece=0.0091, conf=0.9679, entropy=0.0954, margin=0.9418, js_main_rep=0.0267, symkl_main_rep=0.2245, mc_mutual_info=0.0089, mc_variation_ratio=0.0132, mc_sample_agreement=0.9868

| case_name                         | method    |    n |     acc |   acc_main |   acc_rep |       nll |     brier |     conf |   entropy |   margin |   branch_same_pred |   js_main_rep |   symkl_main_rep |        ece |     mc_acc |   mc_predictive_entropy |   mc_expected_entropy |   mc_mutual_info |   mc_sample_agreement |   mc_variation_ratio |   mc_vote_entropy |
|:----------------------------------|:----------|-----:|--------:|-----------:|----------:|----------:|----------:|---------:|----------:|---------:|-------------------:|--------------:|-----------------:|-----------:|-----------:|------------------------:|----------------------:|-----------------:|----------------------:|---------------------:|------------------:|
| BayesMMRL_caltech101_16shot_seed1 | MMRL      | 2465 | 0.96998 |   0.965517 |  0.959432 | 0.103473  | 0.0496891 | 0.976798 | 0.0731998 | 0.958583 |           0.96714  |     0.0175237 |         0.129584 | 0.0143544  | nan        |              nan        |            nan        |     nan          |            nan        |          nan         |       nan         |
| MMRL_caltech101_16shot_seed1      | BayesMMRL | 2465 | 0.96146 |   0.949696 |  0.952535 | 0.0987007 | 0.0519444 | 0.96791  | 0.095382  | 0.941803 |           0.948884 |     0.0267269 |         0.224486 | 0.00908018 |   0.969168 |                0.109345 |              0.100486 |       0.00885865 |              0.986775 |            0.0132252 |         0.0283026 |

### 2.1 Case 间指标差异

| metric         | A_case                            |   A_value | B_case                       |    B_value |   B_minus_A |
|:---------------|:----------------------------------|----------:|:-----------------------------|-----------:|------------:|
| acc            | BayesMMRL_caltech101_16shot_seed1 | 0.96998   | MMRL_caltech101_16shot_seed1 | 0.96146    | -0.00851927 |
| nll            | BayesMMRL_caltech101_16shot_seed1 | 0.103473  | MMRL_caltech101_16shot_seed1 | 0.0987007  | -0.00477226 |
| brier          | BayesMMRL_caltech101_16shot_seed1 | 0.0496891 | MMRL_caltech101_16shot_seed1 | 0.0519444  |  0.00225533 |
| ece            | BayesMMRL_caltech101_16shot_seed1 | 0.0143544 | MMRL_caltech101_16shot_seed1 | 0.00908018 | -0.00527419 |
| conf           | BayesMMRL_caltech101_16shot_seed1 | 0.976798  | MMRL_caltech101_16shot_seed1 | 0.96791    | -0.00888826 |
| entropy        | BayesMMRL_caltech101_16shot_seed1 | 0.0731998 | MMRL_caltech101_16shot_seed1 | 0.095382   |  0.0221823  |
| margin         | BayesMMRL_caltech101_16shot_seed1 | 0.958583  | MMRL_caltech101_16shot_seed1 | 0.941803   | -0.0167796  |
| js_main_rep    | BayesMMRL_caltech101_16shot_seed1 | 0.0175237 | MMRL_caltech101_16shot_seed1 | 0.0267269  |  0.00920324 |
| symkl_main_rep | BayesMMRL_caltech101_16shot_seed1 | 0.129584  | MMRL_caltech101_16shot_seed1 | 0.224486   |  0.0949016  |

## 3. Posterior / Bayesian block 诊断

- **MMRL_caltech101_16shot_seed1**: bayes_target=rep_tokens, block_class=BayesianTensorParameter, KL=24.5855, mean_shift_l2=6.8231, mean_shift_abs_mean=0.1156, sigma_mean=0.9787, sigma_std=0.0072, sigma_cv=0.0074, SNR_mean=0.1182, SNR_max=0.3523
- **BayesMMRL_caltech101_16shot_seed1**: No Bayesian posterior block found

| case_name                         | bayes_target   | block_class             |       KL | posterior_mean_shape   | prior_mean_shape   | posterior_sigma_shape   |   mean_shift_l2 |   mean_shift_abs_mean |   sigma_mean |    sigma_std |     sigma_cv |   SNR_mean |    SNR_max |
|:----------------------------------|:---------------|:------------------------|---------:|:-----------------------|:-------------------|:------------------------|----------------:|----------------------:|-------------:|-------------:|-------------:|-----------:|-----------:|
| MMRL_caltech101_16shot_seed1      | rep_tokens     | BayesianTensorParameter |  24.5855 | (5, 512)               | (5, 512)           | (5, 512)                |         6.82306 |              0.115584 |     0.978684 |   0.00723251 |   0.00739003 |   0.118213 |   0.352331 |
| BayesMMRL_caltech101_16shot_seed1 |                | nan                     | nan      | nan                    | nan                | nan                     |       nan       |            nan        |   nan        | nan          | nan          | nan        | nan        |

## 4. Paired outcome 分析

- Paired samples: **2465**
- B fixes A errors: **12** (0.49%)
- B breaks A correct predictions: **33** (1.34%)
- Both correct: **2358** (95.66%)
- Both wrong: **62** (2.52%)
- 初步 paired 结论：B 相对 A 的净破坏数更多，需要检查不确定性/分支差异是否导致退化。

### 4.1 Outcome counts

| outcome_type      |   count |
|:------------------|--------:|
| both_correct      |    2358 |
| both_wrong        |      62 |
| B_break_A_correct |      33 |
| B_fix_A_error     |      12 |

### 4.2 Top paired delta metrics

| delta_metric                   |        mean |       median |       std |       min |      max |
|:-------------------------------|------------:|-------------:|----------:|----------:|---------:|
| delta_symkl_main_rep_B_minus_A |  0.0949016  |  0.00819545  | 0.523184  | -3.96005  | 6.29778  |
| delta_entropy_B_minus_A        |  0.0221823  |  7.94194e-05 | 0.136453  | -1.53772  | 1.58847  |
| delta_margin_B_minus_A         | -0.0167796  | -2.09808e-05 | 0.112854  | -0.842883 | 0.80907  |
| delta_js_main_rep_B_minus_A    |  0.00920324 |  0.000513989 | 0.0636542 | -0.507538 | 0.645882 |
| delta_conf_B_minus_A           | -0.00888826 | -9.29832e-06 | 0.0624692 | -0.49544  | 0.494991 |
| delta_correct_B_minus_A        | -0.00851927 |  0           | 0.134872  | -1        | 1        |
| delta_nll_B_minus_A            | -0.00477226 |  6.5562e-06  | 0.305337  | -4.76741  | 4.10356  |
| delta_brier_B_minus_A          |  0.00225533 |  6.20141e-10 | 0.139927  | -1.4327   | 1.82019  |

## 5. 样本级输出概览

- Total sample rows: **4930**
- Cases: `['MMRL_caltech101_16shot_seed1', 'BayesMMRL_caltech101_16shot_seed1']`

### 5.1 Correctness by case

| case_name                         |   count |   accuracy |
|:----------------------------------|--------:|-----------:|
| BayesMMRL_caltech101_16shot_seed1 |    2465 |    0.96998 |
| MMRL_caltech101_16shot_seed1      |    2465 |    0.96146 |

## 6. 文件输出

- `audit_cfg.csv`: exists
- `audit_state_dict_bayes_keys.csv`: exists
- `posterior_diagnostics.csv`: exists
- `sample_outputs_test.csv`: exists
- `summary_metrics_test.csv`: exists
- `paired_outcomes_test.csv`: exists
- `mechanism_summary_test.md`: exists
- `mechanism_summary_test.html`: exists
