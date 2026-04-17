from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class EvalContext:
    protocol: str
    dataset_name: str
    split: str
    subsample_classes: Optional[str] = None
    phase: Optional[str] = None


@dataclass
class MethodOutputs:
    logits: torch.Tensor
    labels: Optional[torch.Tensor] = None
    aux_logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)
