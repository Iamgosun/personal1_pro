from __future__ import annotations

import torch


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def lp_logits(logit_scale, features, prototypes):
    if features.ndim != 2:
        raise ValueError(f"lp_logits expects features [B, D], got {tuple(features.shape)}")
    if prototypes.ndim != 2:
        raise ValueError(f"lp_logits expects prototypes [C, D], got {tuple(prototypes.shape)}")

    features = _normalize(features)
    prototypes = _normalize(prototypes)
    return features @ prototypes.t() * logit_scale.exp()


def bayes_logits(logit_scale, features, prototypes):
    if features.ndim != 2:
        raise ValueError(f"bayes_logits expects features [B, D], got {tuple(features.shape)}")
    if prototypes.ndim != 3:
        raise ValueError(f"bayes_logits expects prototypes [S, C, D], got {tuple(prototypes.shape)}")

    features = _normalize(features)
    prototypes = _normalize(prototypes)
    all_logits = torch.einsum("bd,scd->bsc", features, prototypes) * logit_scale.exp()
    return all_logits.mean(dim=1)