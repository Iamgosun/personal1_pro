from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, pstdev


def summarize_seed_runs(directory: str, keyword: str = 'accuracy'):
    values = []
    pattern = re.compile(rf"\* {re.escape(keyword)}: ([0-9.]+)%")
    for log in Path(directory).rglob('log.txt'):
        text = log.read_text(encoding='utf-8', errors='ignore')
        matches = pattern.findall(text)
        if matches:
            values.append(float(matches[-1]))
    return values


def summarize_multi_runs(directory: str, keyword: str = 'accuracy', ci95: bool = False):
    values = summarize_seed_runs(directory, keyword=keyword)
    if not values:
        return {'n': 0, 'mean': None, 'std': None}
    out = {'n': len(values), 'mean': mean(values), 'std': pstdev(values) if len(values) > 1 else 0.0}
    if ci95 and len(values) > 1:
        out['ci95'] = 1.96 * out['std'] / (len(values) ** 0.5)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--keyword', default='accuracy')
    parser.add_argument('--ci95', action='store_true')
    args = parser.parse_args()
    print(summarize_multi_runs(args.directory, keyword=args.keyword, ci95=args.ci95))


if __name__ == '__main__':
    main()
