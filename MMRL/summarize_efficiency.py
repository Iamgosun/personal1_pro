"""Summarize all efficiency.json under output directory, mean ± sample SD."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = ['ms_per_update', 'updates_per_s', 'peak_allocated_mib', 'peak_reserved_mib', 'gflops_profiler_per_update']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='output_efficiency')
    ap.add_argument('--output', default='efficiency_summary.csv')
    a = ap.parse_args()
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in Path(a.input).rglob('efficiency.json')]
    if not rows:
        raise RuntimeError('No efficiency.json found')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['method'])].append(r)
    report = []
    for (dataset, method), vals in sorted(groups.items()):
        d = {'dataset': dataset, 'method': method, 'n_seeds': len(vals)}
        seeds = [r['seed'] for r in vals]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate seed for {(dataset, method)}')
        for f in FIELDS:
            xs = [v[f] for v in vals if v.get(f) is not None]
            d[f] = (f'{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}' if len(xs)>1
                    else f'{xs[0]:.3f} (n=1)' if xs else 'N/A')
        report.append(d)
    with open(a.output, 'w', newline='', encoding='utf-8') as out:
        w = csv.DictWriter(out, fieldnames=['dataset','method','n_seeds']+FIELDS)
        w.writeheader(); w.writerows(report)
    print(f'Saved {len(report)} dataset-method rows to {a.output}')

if __name__ == '__main__':
    main()
