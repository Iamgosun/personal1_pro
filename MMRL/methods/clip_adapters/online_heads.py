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


def capel_logits(logit_scale, features, prototypes, prompt_weights):
    """
    CAPEL logits-space prompt ensemble, following the paper equations.

    features:       [B, D]
    prototypes:     [C, K, D]
    prompt_weights: [C, K], raw learnable alpha

    returns:
      logits:     [B, C]
      sub_logits: [B, C, K], where Z_yk = tau * cos(x, w_yk)
    """
    if features.ndim != 2:
        raise ValueError(f"capel_logits expects features [B, D], got {tuple(features.shape)}")
    if prototypes.ndim != 3:
        raise ValueError(f"capel_logits expects prototypes [C, K, D], got {tuple(prototypes.shape)}")
    if prompt_weights.ndim != 2:
        raise ValueError(
            f"capel_logits expects prompt_weights [C, K], got {tuple(prompt_weights.shape)}"
        )
    if prototypes.shape[:2] != prompt_weights.shape:
        raise ValueError(
            "CAPEL prototype/prompt weight mismatch: "
            f"prototypes[:2]={tuple(prototypes.shape[:2])}, "
            f"prompt_weights={tuple(prompt_weights.shape)}"
        )

    features = _normalize(features.float())
    prototypes = _normalize(prototypes.float())

    # Paper Eq.4:
    #   Z_yk = tau * cos(x, w_yk)
    sub_logits = torch.einsum("bd,ckd->bck", features, prototypes) * logit_scale.exp()

    # Paper Eq.5 / Eq.12:
    #   class_logit_y = sum_k alpha_yk * Z_yk
    alpha = prompt_weights.float()
    logits = (sub_logits * alpha.unsqueeze(0)).sum(dim=-1)

    return logits, sub_logits


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
