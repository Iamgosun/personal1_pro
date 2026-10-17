"""Table A: same-checkpoint Original / Branch / DAVI classification NLL.

The six-term decomposition is retained in the raw CSV for optional diagnosis;
it is NOT the central Table A. Never subtract losses from two checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# Main paper: objective fidelity only.
TABLE_COLUMNS = (
    ('original_nll', 'Original NLL'),
    ('branch_nll', 'Branch NLL'),
    ('davi_nll', 'DAVI NLL'),
    ('mix_gap', 'Mix Gap (B-O)'),
    ('det_gap', 'Det Gap (D-B)'),
    ('total_gap', 'Total Gap (D-O)'),
    ('mae_det_gap', 'E|D-B|'),
)
# Optional decomposition, never confused with main comparison.
SIX = ('mix_gap', 'softmax_jensen_gap', 'gaussian_error',
       'propagation_error', 'scale_error', 'clamp_error')
EXTRA = ('mae_total_gap', 'p90_abs_det_gap', 'empirical_logit_variance',
         'propagated_logit_variance', 'moment_mu_mae', 'moment_var_mae',
         'mgf_ess_ratio_min', 'clip_fraction')
NEEDED = {k for k, _ in TABLE_COLUMNS} | set(SIX) | set(EXTRA)


def record_file(root, ds, mode, shot, seed, samples, repeats):
    return (root / 'native' / mode / ds / f'shot_{shot}' / f'seed{seed}'
            / 'phase1' / f'stage1_S{samples}_R{repeats}.json')


def load(root, ds, mode, shot, seed, samples, repeats, split):
    p = record_file(root, ds, mode, shot, seed, samples, repeats)
    if not p.is_file():
        raise FileNotFoundError(f'Run stage1_evaluate.py first: {p}')
    data = json.loads(p.read_text(encoding='utf-8'))
    expected = {'diagnostic_schema': 'stage1-native-3', 'mode': mode,
                'profile': 'native', 'dataset': ds, 'shot': shot,
                'seed': seed, 'mc_samples': samples, 'mc_repeats': repeats,
                'diagnostic_partial': False, 'cache_validated': True}
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f'{p}: {key} should be {value!r}, got {data.get(key)!r}')
    if not data.get('checkpoint_sha256'):
        raise ValueError(f'{p}: checkpoint hash is missing')
    if split not in data['splits']:
        raise ValueError(f'{p}: missing split {split!r}')
    detail = data['splits'][split]
    if not detail.get('cache_parity'):
        raise ValueError(f'{p}: production/cached logits parity was not checked')
    by_s = detail['by_sample_count']
    selected = by_s[str(samples)]
    if len(selected['repeats']) != repeats:
        raise ValueError(f'{p}: wrong number of MC repeats')
    vals = selected['summary']
    for key in NEEDED:
        if key not in vals or not math.isfinite(float(vals[key]['mean'])):
            raise ValueError(f'{p}: missing/nonfinite {key}')
    v = lambda key: float(vals[key]['mean'])
    if abs(v('branch_nll') - v('original_nll') - v('mix_gap')) > 1e-7:
        raise ValueError(f'{p}: mix gap mismatch')
    if abs(v('davi_nll') - v('branch_nll') - v('det_gap')) > 1e-7:
        raise ValueError(f'{p}: det gap mismatch')
    if abs(v('davi_nll') - v('original_nll') - v('total_gap')) > 1e-7:
        raise ValueError(f'{p}: total gap mismatch')
    if abs(sum(v(k) for k in SIX) - v('total_gap')) > 1e-7:
        raise ValueError(f'{p}: detailed decomposition mismatch')
    if v('mix_gap') < -1e-7 or v('softmax_jensen_gap') < -1e-7:
        raise ValueError(f'{p}: Jensen sign violation')
    if v('mae_det_gap') + 1e-8 < abs(v('det_gap')):
        raise ValueError(f'{p}: per-image |D-B| is smaller than |mean(D-B)|')

    record = {'dataset': ds, 'method': mode, 'shot': shot, 'seed': seed,
              'split': split, 'samples': samples, 'mc_repeats': repeats,
              'num_examples': detail['num_examples'],
              'input_sha256': detail['input_sha256'],
              'checkpoint_sha256': data['checkpoint_sha256'],
              'effective_config_sha256': data['effective_config_sha256'],
              'alpha': data['alpha'], 'eta': data['eta'], 'clamp': data['clamp'],
              'visual_pre_snr': detail['visual_pre_snr']}
    for key in sorted(NEEDED):
        record[key] = v(key)
        record[key + '_mc_se'] = vals[key]['mc_se']

    curve = []
    for s, stats in by_s.items():
        row = {'dataset': ds, 'method': mode, 'seed': seed,
               'split': split, 'samples': int(s)}
        for metric in ('det_gap', 'mix_gap', 'total_gap'):
            row[metric] = stats['summary'][metric]['mean']
            row[metric + '_mc_se'] = stats['summary'][metric]['mc_se']
        curve.append(row)
    return record, curve


def write_csv(path, rows):
    if not rows:
        raise ValueError('No measured results')
    with path.open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(args):
    # Resolve relative OUT with respect to the MMRL repository, not the caller's cwd.
    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    root = root.resolve()
    raw, curves = [], []
    for ds in args.datasets:
        for seed in args.seeds:
            comparison = []
            for mode in args.modes:
                row, curve = load(root, ds, mode, args.shot, seed,
                                  args.samples, args.mc_repeats, args.split)
                raw.append(row)
                curves.extend(curve)
                comparison.append(row)
            # Cross-check input equality only, never require matching posterior/initialization.
            for key in ('input_sha256', 'num_examples', 'alpha', 'eta', 'clamp'):
                if len({r[key] for r in comparison}) != 1:
                    raise ValueError(f'{ds} seed={seed}: inconsistent {key}')
    summary_dir = root / 'phase1_summary' / args.split
    # --out is a filename/prefix relative to the summary folder by default.
    # An explicit absolute --out remains supported for manual exports.
    dest = Path(args.out) if args.out else Path(f'table_a_native_{args.split}')
    if not dest.is_absolute():
        dest = summary_dir / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(dest.with_name(dest.name + '_raw.csv'), raw)
    write_csv(dest.with_name(dest.name + '_convergence.csv'), curves)
    table = []
    for ds in args.datasets:
        for mode in args.modes:
            records = [r for r in raw if r['dataset'] == ds and r['method'] == mode]
            record = {'Dataset': ds, 'Checkpoint': mode,
                      'Training seeds': len(records), 'Split': args.split,
                      'MC samples': args.samples}
            for key, title in TABLE_COLUMNS:
                mean = statistics.mean(r[key] for r in records)
                sd = statistics.stdev(r[key] for r in records) if len(records) > 1 else None
                record[title] = (f'{mean:.6f} ± {sd:.6f}' if sd is not None
                                 else f'{mean:.6f} (one seed)')
            table.append(record)
    write_csv(dest.with_suffix('.csv'), table)
    headers = list(table[0])
    md = [
        '# Table A — Objective Fidelity under Native DAVI/AVI Checkpoints', '',
        'For each row, all three NLL objectives use the SAME fixed checkpoint '
        'and SAME data. Units: nats per image.',
        f'Protocol: {args.split}; S={args.samples}, R={args.mc_repeats}. '
        'Mean ± training-seed SD where available. MC sampling errors are in the raw CSV.',
        '',
        '**Interpretation:** Mix = Branch−Original (rigorously nonnegative); '
        'Det = DAVI−Branch (signed); Total = DAVI−Original (signed). '
        'E|D−B| is the per-image absolute approximation error.',
        'Neither split is the original random-augmentation training ELBO; '
        'the held-out test NLL is not the official static_logit inference NLL.',
        '', '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    for row in table:
        md.append('| ' + ' | '.join(str(row[k]) for k in headers) + ' |')
    dest.with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(f'[OK] Table A: {dest.with_suffix(".csv")} and {dest.with_suffix(".md")}')
    print(f'[OK] Diagnostics: {dest.with_name(dest.name + "_raw.csv")}')
    print(f'[OK] Convergence: {dest.with_name(dest.name + "_convergence.csv")}')


def main():
    ap = argparse.ArgumentParser(description='Native Stage-1 objective fidelity Table A')
    ap.add_argument('--output-root', default='output_objective_fidelity_v3')
    ap.add_argument('--datasets', nargs='+', default=['caltech101'])
    ap.add_argument('--modes', nargs='+', choices=['original_mc', 'branch_mc', 'davi'],
                    default=['branch_mc', 'davi'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1])
    ap.add_argument('--shot', type=int, default=16)
    ap.add_argument('--samples', type=int, default=4096)
    ap.add_argument('--mc-repeats', type=int, default=1)
    ap.add_argument('--split', choices=['train_fixed', 'test'], default='train_fixed')
    ap.add_argument('--out', default=None, help='Output prefix, relative to OUT/phase1_summary/SPLIT unless absolute')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
