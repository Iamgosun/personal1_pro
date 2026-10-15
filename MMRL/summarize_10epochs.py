"""Summarize 10-epoch training cost by dataset, method, and matched configuration."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = (
    'train_time_min',
    'ms_per_update',
    'updates_per_s',
    'gflops_profiler_per_update',
    'peak_allocated_mib',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='output_efficiency_10ep')
    ap.add_argument('--output', default='output_efficiency_10ep/efficiency_summary.csv')
    a = ap.parse_args()
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in Path(a.input).rglob('efficiency.json')]
    if not rows:
        raise RuntimeError('No efficiency.json found')

    groups = defaultdict(list)
    for r in rows:
        key = (r['dataset'], r['method'], r['epochs_completed'],
               r['num_shots'], r['batch_size'], r.get('n_mc_train_effective'))
        groups[key].append(r)

    report = []
    for (dataset, method, epochs, shots, batch, mc), vals in sorted(
        groups.items(), key=lambda kv: str(kv[0])
    ):
        seeds = [r['seed'] for r in vals]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate seed for {(dataset, method, epochs, shots, batch, mc)}')
        d = {
            'dataset': dataset,
            'method': method,
            'epochs': epochs,
            'shots': shots,
            'batch_size': batch,
            'n_mc_train': mc if mc is not None else '-',
            'n_seeds': len(vals),
        }
        for field in FIELDS:
            xs = [v[field] for v in vals if v.get(field) is not None]
            if len(xs) > 1:
                d[field] = f'{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}'
            elif xs:
                d[field] = f'{xs[0]:.3f} (n=1)'
            else:
                d[field] = 'N/A'
        report.append(d)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'method', 'epochs', 'shots', 'batch_size',
                  'n_mc_train', 'n_seeds', *FIELDS]
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report)
    print(f'Saved {len(report)} dataset-method rows to {out}')


if __name__ == '__main__':
    main()
