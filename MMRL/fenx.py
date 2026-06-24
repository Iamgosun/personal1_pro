from pathlib import Path
import json
import math

root = Path("output_refactor/ClipAdapters/TipA/FS")

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            yield prefix, obj

for p in root.rglob("*.json"):
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[READ_ERROR] {p}: {e}")
        continue

    bad_items = list(walk(data))
    if bad_items:
        print(f"\n[BAD] {p}")
        for key, value in bad_items:
            print(f"  {key} = {value}")