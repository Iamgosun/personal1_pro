"""Summarize MMRL efficiency JSONs, including encoder/head profiler splits.

Drop-in replacement for summarize_efficiency.py. Both old (no split) and new
JSON are supported. Identical (dataset, method, configuration, seed) records
at different paths are de-duplicated by newest file modification time;
a warning names all duplicates rather than crashing post-training.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

FIELDS = [
    'ms_per_update', 'updates_per_s',
    'peak_allocated_mib', 'peak_reserved_mib',
    'gflops_profiler_per_update',
    'encoding_forward_gflops_profiler',
    'probabilistic_forward_gflops_profiler',
    'other_including_backward_gflops_profiler',
    'encoding_forward_cuda_kernel_ms_sum',
    'probabilistic_forward_cuda_kernel_ms_sum',
    'other_including_backward_cuda_kernel_ms_sum',
]


def fmt(values):
    if not values:
        return 'N/A'
    if len(values) == 1:
        return f'{values[0]:.3f} (n=1)'
    return f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}'


def signature(r):
    # Configuration must be part of grouping, otherwise a run with MC=3 and
    # MC=10 can be wrongly averaged. Older JSON may not include n_mc_train.
    return (
        r.get('dataset'), r.get('method'), r.get('num_shots'),
        r.get('batch_size'), r.get('n_mc_train'),
        r.get('max_epoch'), r.get('pretrain_epochs_completed'),
        r.get('method_config_file'),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='output_efficiency')
    p.add_argument('--output', default='efficiency_summary.csv')
    a = p.parse_args()

    files = list(Path(a.input).rglob('efficiency.json'))
    if not files:
        raise RuntimeError(f'No efficiency.json found in {a.input}')

    latest = {}
    for path in files:
        r = json.loads(path.read_text(encoding='utf-8'))
        identity = signature(r) + (r['seed'],)
        old = latest.get(identity)
        if old is not None:
            oldpath, _ = old
            print(f'[warning] Duplicate same configuration/seed:\n'
                  f'  {oldpath}\n  {path}\n'
                  f'  Keeping most recently modified file for this seed.')
        if old is None or path.stat().st_mtime_ns > old[0].stat().st_mtime_ns:
            latest[identity] = (path, r)

    groups = defaultdict(list)
    for _, r in latest.values():
        groups[signature(r)].append(r)

    rows = []
    cols = [
        'dataset', 'method', 'shots', 'batch_size', 'n_mc_train',
        'max_epoch', 'pretrain_epochs', 'n_seeds',
    ] + FIELDS
    for key, runs in sorted(groups.items(), key=lambda entry: str(entry[0])):
        dataset, method, shots, batch, mc, epochs, pretrain, _ = key
        row = dict(dataset=dataset, method=method,
                   shots=shots, batch_size=batch, n_mc_train=mc,
                   max_epoch=epochs, pretrain_epochs=pretrain, n_seeds=len(runs))
        for field in FIELDS:
            vals = [float(r[field]) for r in runs if r.get(field) is not None]
            row[field] = fmt(vals)
        rows.append(row)

    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {dest}')


if __name__ == '__main__':
    main()
