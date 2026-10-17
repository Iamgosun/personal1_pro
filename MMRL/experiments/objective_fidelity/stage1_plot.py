"""Plot measured Stage-1 objective gaps (no changes to experimental metrics).

All labels are ASCII English to avoid missing CJK/math glyphs on headless servers.
PDF embeds TrueType fonts; PNG is provided for PDF viewer compatibility.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def run(raw_file, convergence_file, out_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = _read_csv(raw_file)
    curves = _read_csv(convergence_file)
    if not rows or not curves:
        raise ValueError('No actual Stage-1 observations')

    # DejaVu Sans is distributed with Matplotlib. Type 42 embeds real fonts in
    # PDFs, unlike the default Type 3 font that can render poorly in viewers.
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
        'font.size': 10,
        'savefig.dpi': 180,
    })
    out = Path(out_dir) if out_dir is not None else Path(raw_file).resolve().parent / 'figures'
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, stem):
        # Publish both formats; no non-ASCII glyphs anywhere in plot text.
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for method in sorted({r['method'] for r in rows}):
        subset = [r for r in rows if r['method'] == method]
        x = [float(r['propagated_logit_variance']) for r in subset]
        signed = [float(r['det_gap']) for r in subset]
        absolute = [float(r['mae_det_gap']) for r in subset]
        if not all(math.isfinite(v) for v in x + signed + absolute):
            raise ValueError('Cannot plot non-finite posterior variance / Det Gap')
        axes[0].scatter(x, signed, label=method, s=48, alpha=.9)
        axes[1].scatter(x, absolute, label=method, s=48, alpha=.9)
    axes[0].axhline(0, color='gray', linewidth=.8)
    axes[0].set_ylabel('Signed Det Gap: D - B (nats/image)')
    axes[1].set_ylabel('Per-image MAE: E|D - B| (nats/image)')
    for ax in axes:
        ax.set_xlabel('Approx. logit variance')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('DAVI approximation gap vs. posterior uncertainty')
    save(fig, 'figure_A_det_gap_vs_variance')

    by_checkpoint = defaultdict(list)
    for r in curves:
        by_checkpoint[(r['dataset'], r['method'], int(r['seed']))].append(r)
    for ds in sorted({key[0] for key in by_checkpoint}):
        selected = [(key, points) for key, points in by_checkpoint.items() if key[0] == ds]
        unique_s = {int(p['samples']) for _, points in selected for p in points}
        name = f'figure_A_mc_convergence_{ds}'
        if len(unique_s) < 2:
            # A single S is not a convergence curve. Remove prior stale chart.
            for ext in ('pdf', 'png'):
                stale = out / f'{name}.{ext}'
                if stale.exists():
                    stale.unlink()
            print(f'[SKIP] {ds}: only S={next(iter(unique_s))}; need >=2 sample counts for MC convergence')
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for (_, method, seed), points in sorted(selected):
            points = sorted(points, key=lambda r: int(r['samples']))
            x = [int(p['samples']) for p in points]
            y = [float(p['det_gap']) for p in points]
            err = [float(p['det_gap_mc_se']) if p['det_gap_mc_se'] not in ('', 'None', None)
                   else 0. for p in points]
            if any(e > 0 for e in err):
                ax.errorbar(x, y, yerr=err, marker='o', capsize=2,
                            label=f'{method} / seed {seed}')
            else:
                ax.plot(x, y, marker='o', label=f'{method} / seed {seed}')
        ax.set_xscale('log')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_xlabel('Posterior MC samples (S)')
        ax.set_ylabel('Det Gap: D - B (nats/image)')
        ax.set_title(f'MC convergence - {ds} (one line per checkpoint)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, frameon=False)
        save(fig, name)
    print(f'[OK] Stage-1 figures (PDF + PNG): {out}')


def main():
    parser = argparse.ArgumentParser(description='Stage-1 objective fidelity plots')
    parser.add_argument('--raw', required=True)
    parser.add_argument('--convergence', required=True)
    parser.add_argument('--out-dir', default=None,
                        help='Default: figures/ alongside the raw Table A CSV')
    args = parser.parse_args()
    run(args.raw, args.convergence, args.out_dir)


if __name__ == '__main__':
    main()
