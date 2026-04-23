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


def bayes_logits_all(logit_scale, features, prototypes):
    """
    Returns per-sample logits with shape [S, B, C].
    """
    if features.ndim != 2:
        raise ValueError(f"bayes_logits_all expects features [B, D], got {tuple(features.shape)}")
    if prototypes.ndim != 3:
        raise ValueError(f"bayes_logits_all expects prototypes [S, C, D], got {tuple(prototypes.shape)}")

    features = _normalize(features)
    prototypes = _normalize(prototypes)
    return torch.einsum("bd,scd->sbc", features, prototypes) * logit_scale.exp()


def bayes_logits(logit_scale, features, prototypes):
    """
    Backward-compatible predictive-mean logits [B, C].
    """
    return bayes_logits_all(logit_scale, features, prototypes).mean(dim=0)