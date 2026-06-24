from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/data')
OUT_PDF = ROOT / 'figure3_shots_davi_vector.pdf'
OUT_SVG = ROOT / 'figure3_shots_davi_vector.svg'
OUT_PNG = ROOT / 'figure3_shots_davi_preview.png'

FILES = {
    'CrossModal': ROOT / 'CM_训练加上验证集aggregated_test_summary (3).csv',
    'TaskRes': ROOT / 'TR_验证集选参数aggregated_test_summary.csv',
    'LP++': ROOT / 'LP++训练集加上验证集aggregated_test_summary (3).csv',
    'CLAP': ROOT / 'CLAP训练加上验证集aggregated_test_summary (3).csv',
    'BayesAdapter': ROOT / 'BA_验证集选参数aggregated_test_summary.csv',
    'DAVI-Adapter': ROOT / 'Det_验证集选参数aggregated_test_summary.csv',
}
WIDE = ROOT / 'detba_baseline_metrics_wide.csv'
TARGET_DATASETS = {
    'caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101', 'imagenet',
    'oxford_flowers', 'oxford_pets', 'sun397', 'stanford_cars', 'ucf101'
}
SHOTS = [1, 2, 4, 8, 16, 32]


def parse_case_root(case_root: str):
    m = re.search(r'fewshot_train/([^/]+)/shots_(\d+)', str(case_root))
    return (m.group(1), int(m.group(2))) if m else (None, None)


rows = []
for method, path in FILES.items():
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        dataset, shot = parse_case_root(row.get('case_root', ''))
        if dataset not in TARGET_DATASETS or shot not in SHOTS:
            continue
        cov = row.get('confthr_99_valid_coverage_mean', row.get('confthr_99_coverage_mean', np.nan))
        rows.append({
            'method': method,
            'dataset': dataset,
            'shot': shot,
            'ACC': float(row['accuracy_mean']),
            'ECE': float(row['ece_mean']) / 100.0,
            'COV99': float(cov) / 100.0,
        })

fig3_df = pd.DataFrame(rows)

# Supplement TaskRes / SUN397 / 32-shot if missing.
if not ((fig3_df.method == 'TaskRes') & (fig3_df.dataset == 'sun397') & (fig3_df.shot == 32)).any() and WIDE.exists():
    wide = pd.read_csv(WIDE)
    wm = wide[(wide.method == 'TaskRes') & (wide.num_shots == 32)]
    rec = {
        'method': 'TaskRes',
        'dataset': 'sun397',
        'shot': 32,
        'ACC': float(wm[wm.metric == 'ACC'].iloc[0]['SUN397_mean']),
        'ECE': float(wm[wm.metric == 'ECE'].iloc[0]['SUN397_mean']) / 100.0,
        'COV99': float(wm[(wm.metric == 'Coverage') & (wm.confidence_threshold == 99)].iloc[0]['SUN397_mean']) / 100.0,
    }
    fig3_df = pd.concat([fig3_df, pd.DataFrame([rec])], ignore_index=True)

shot_avg = fig3_df.groupby(['method', 'shot'], as_index=False).agg({'ACC': 'mean', 'ECE': 'mean', 'COV99': 'mean'})

methods = ['CrossModal', 'TaskRes', 'LP++', 'CLAP', 'BayesAdapter', 'DAVI-Adapter']
style = {
    'CrossModal': dict(color='#d62728', marker='o'),
    'TaskRes': dict(color='#2ca02c', marker='s'),
    'LP++': dict(color='#ff7f0e', marker='^'),
    'CLAP': dict(color='#1f77b4', marker='o'),
    'BayesAdapter': dict(color='#000000', marker='o'),
    'DAVI-Adapter': dict(color='#9467bd', marker='*'),
}

fig, axs = plt.subplots(1, 3, figsize=(13.4, 3.9), facecolor='white')
for ax in axs:
    ax.set_facecolor('white')

metric_info = [
    ('ACC', '(a) ACC', 'Accuracy (%)'),
    ('ECE', '(b) ECE', 'Expected Calibration Error'),
    ('COV99', '(c) coverage at conf.99%', 'Coverage at 99% confidence'),
]

for ax, (metric, title, ylabel) in zip(axs, metric_info):
    for method in methods:
        g = shot_avg[shot_avg.method == method].sort_values('shot')
        ax.plot(g['shot'], g[metric], label=method, linewidth=1.7, markersize=4.2, **style[method])
    ax.set_xticks(SHOTS)
    ax.set_xlabel('Number of shots', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=9)
    ax.text(0.5, -0.26, title, ha='center', va='top', transform=ax.transAxes, fontsize=12)

axs[0].set_ylim(np.floor(shot_avg['ACC'].min() - 1.0), np.ceil(shot_avg['ACC'].max() + 1.0))
axs[1].set_ylim(max(0.0, shot_avg['ECE'].min() - 0.005), shot_avg['ECE'].max() + 0.01)
axs[2].set_ylim(max(0.0, shot_avg['COV99'].min() - 0.02), shot_avg['COV99'].max() + 0.03)
axs[0].legend(loc='lower right', fontsize=8, frameon=True)

fig.tight_layout(w_pad=1.2)
fig.savefig(OUT_PDF, facecolor='white', edgecolor='white', bbox_inches='tight')
fig.savefig(OUT_SVG, facecolor='white', edgecolor='white', bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=220, facecolor='white', edgecolor='white', bbox_inches='tight')
print(f'Saved: {OUT_PDF}')
print(f'Saved: {OUT_SVG}')
print(f'Saved: {OUT_PNG}')
