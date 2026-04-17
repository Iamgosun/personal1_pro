from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict


def save_summary(path: str, payload: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def save_classwise_metrics(path: str, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'metric'])
        writer.writerows(rows)


def save_confidence_coverage(path: str, payload: Dict):
    save_summary(path, payload)


def save_ece(path: str, payload: Dict):
    save_summary(path, payload)
