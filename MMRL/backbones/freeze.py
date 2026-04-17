from __future__ import annotations

from typing import Iterable, Set


def freeze_all_but(model, allowed_substrings: Iterable[str]) -> Set[str]:
    allowed = set(allowed_substrings)
    enabled = set()
    for name, param in model.named_parameters():
        update = any(key in name for key in allowed)
        param.requires_grad_(update)
        if update:
            enabled.add(name)
    return enabled
