from __future__ import annotations

import os
from pathlib import Path


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def build_output_dir(root: str, method: str, protocol: str, phase: str, dataset: str, shots: int, backbone: str, tag: str, seed: int) -> str:
    path = Path(root) / method / protocol / phase / dataset / f"shots_{shots}" / backbone.replace('/', '-') / tag / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
