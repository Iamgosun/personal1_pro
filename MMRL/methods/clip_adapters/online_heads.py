from __future__ import annotations

import torch


def lp_logits(logit_scale, features, prototypes):
    features = features / features.norm(dim=-1, keepdim=True)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    return features @ prototypes.t() * logit_scale.exp()


def bayes_logits(logit_scale, features, prototypes):
    features = features / features.norm(dim=-1, keepdim=True)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    all_logits = torch.einsum('bd,scd->bsc', features, prototypes) * logit_scale.exp()
    return all_logits.mean(dim=1)
