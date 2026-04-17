from __future__ import annotations

import torch

from .adapters import (
    ClipAdapterResidual,
    CrossModalProbeAdapter,
    GaussianPerClassAdapter,
    RandomProbeAdapter,
    TaskResidualAdapter,
    TipAdapter,
    ZeroShotProbeAdapter,
)


def build_adapter(cfg, clip_model, base_text_features: torch.Tensor):
    init = cfg.TRAINER.ClipADAPTER.INIT
    if init == "RANDOM":
        return RandomProbeAdapter(cfg, clip_model, base_text_features)
    if "ZS" in init:
        return ZeroShotProbeAdapter(cfg, clip_model, base_text_features)
    if "TR" in init:
        return TaskResidualAdapter(cfg, clip_model, base_text_features)
    if "ClipA" in init:
        return ClipAdapterResidual(cfg, clip_model, base_text_features)
    if "TipA" in init:
        return TipAdapter(cfg, clip_model, base_text_features)
    if "CrossModal" in init:
        return CrossModalProbeAdapter(cfg, clip_model, base_text_features)
    if "GAUSSIAN_PER_CLASS" in init:
        return GaussianPerClassAdapter(cfg, clip_model, base_text_features)
    raise NotImplementedError(f"Initialization for Linear Probing not implemented: {init}")


AdapterMethod = build_adapter
