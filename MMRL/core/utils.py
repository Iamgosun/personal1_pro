from __future__ import annotations

import importlib
from typing import Iterable


def import_optional_modules(modules: Iterable[str]):
    imported = []
    for name in modules:
        try:
            imported.append(importlib.import_module(name))
        except Exception as exc:
            print(f'[WARN] optional import failed: {name}: {exc}')
            continue
    return imported
