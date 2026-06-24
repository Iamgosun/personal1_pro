from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path('/mnt/data')
CSV_PATH = ROOT / 'fig2_calibration_bins_macro_updated_v10.csv'
OUT_PDF = ROOT / 'figure2_calibration_davi_vector.pdf'
OUT_SVG = ROOT / 'figure2_calibration_davi_vector.svg'
OUT_PNG = ROOT / 'figure2_calibration_davi_preview.png'

# Load updated calibration-bin data.
df = pd.read_csv(CSV_PATH)
df['method'] = df['method'].replace({
    'TR': 'TaskRes',
    'DetBayesRTMMRL': 'DAVI-Adapter'
})

methods = ['CrossModal', 'TaskRes', 'BayesAdapter', 'DAVI-Adapter']
method_labels = {
    'CrossModal': '(a) CrossModal',
    'TaskRes': '(b) TaskRes',
    'BayesAdapter': '(c) BayesAdapter',
    'DAVI-Adapter': '(d) DAVI-Adapter',
}

# White background and wide aspect ratio to match the requested paper-like layout.
fig = plt.figure(figsize=(15.0, 5.0), facecolor='white')
gs = fig.add_gridspec(2, 4, height_ratios=[0.78, 1.5], hspace=0.18, wspace=0.18)

for col, method in enumerate(methods):
    g = df[df['method'] == method].sort_values('bin_index')
    ax_top = fig.add_subplot(gs[0, col], facecolor='white')
    ax_bot = fig.add_subplot(gs[1, col], facecolor='white')

    # Histogram.
    lefts = g['range_left'].to_numpy() * 100
    rights = g['range_right'].to_numpy() * 100
    centers = (lefts + rights) / 2
    widths = rights - lefts
    sample_pct = g['sample_pct'].to_numpy()  # already stored in percentage units
    mean_conf = float(np.average(g['avg_confidence'], weights=sample_pct))
    mean_acc = float(np.average(g['avg_accuracy'], weights=sample_pct))

    ax_top.bar(centers, sample_pct, width=widths * 0.92, color='#4a4a4a', edgecolor='#222222', linewidth=0.45)
    ax_top.axvline(mean_conf, color='#2ca02c', linestyle='--', linewidth=1.0, label='Avg Conf.')
    ax_top.axvline(mean_acc, color='#1f77b4', linestyle='--', linewidth=1.0, label='Avg Acc.')
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, max(55, sample_pct.max() * 1.08))
    ax_top.set_xlabel('Confidence score', fontsize=5.8, labelpad=1.0)
    ax_top.set_ylabel('% of samples', fontsize=5.8, labelpad=1.0)
    ax_top.tick_params(axis='both', labelsize=5.2, pad=1)
    ax_top.grid(True, color='#d0d0d0', alpha=0.6, linewidth=0.4)
    ax_top.legend(loc='upper left', fontsize=4.3, frameon=False, handlelength=1.4, borderaxespad=0.1)
    for s in ax_top.spines.values():
        s.set_linewidth(0.6)
        s.set_color('black')

    # Reliability diagram.
    ax_bot.plot([0, 100], [0, 100], linestyle='--', color='black', linewidth=0.85)
    conf_color = '#2ca02c'
    acc_color = '#1f6fff'
    gap_color = '#e8c4cc'
    for _, row in g.iterrows():
        x0 = row['range_left'] * 100
        x1 = row['range_right'] * 100
        yc = row['avg_confidence']
        ya = row['avg_accuracy']
        ax_bot.hlines(yc, x0, x1, color=conf_color, linewidth=1.5)
        ax_bot.hlines(ya, x0, x1, color=acc_color, linewidth=1.5)
        ax_bot.fill_between([x0, x1], [yc, yc], [ya, ya], color=gap_color, alpha=0.75)

    handles = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=0.85, label='Perfect Calibration'),
        Line2D([0], [0], color=conf_color, linewidth=1.5, label='Confidence'),
        Line2D([0], [0], color=acc_color, linewidth=1.5, label='Accuracy'),
        Line2D([0], [0], color=gap_color, linewidth=4.5, alpha=0.75, label='Gap'),
    ]
    ax_bot.set_xlim(0, 100)
    ax_bot.set_ylim(0, 100)
    ax_bot.set_xlabel('Confidence score', fontsize=5.8, labelpad=1.0)
    ax_bot.set_ylabel('Accuracy and Confidence', fontsize=5.8, labelpad=1.0)
    ax_bot.tick_params(axis='both', labelsize=5.2, pad=1)
    ax_bot.grid(True, color='#d0d0d0', alpha=0.6, linewidth=0.4)
    ax_bot.legend(handles=handles, loc='upper left', fontsize=4.3, frameon=False, handlelength=1.4, borderaxespad=0.1)
    for s in ax_bot.spines.values():
        s.set_linewidth(0.6)
        s.set_color('black')

    ax_bot.text(0.5, -0.22, method_labels[method], transform=ax_bot.transAxes,
                ha='center', va='top', fontsize=8.4)

fig.subplots_adjust(left=0.04, right=0.995, top=0.98, bottom=0.15)
fig.savefig(OUT_PDF, facecolor='white', edgecolor='white', bbox_inches='tight')
fig.savefig(OUT_SVG, facecolor='white', edgecolor='white', bbox_inches='tight')
fig.savefig(OUT_PNG, dpi=220, facecolor='white', edgecolor='white', bbox_inches='tight')
print(f'Saved: {OUT_PDF}')
print(f'Saved: {OUT_SVG}')
print(f'Saved: {OUT_PNG}')
