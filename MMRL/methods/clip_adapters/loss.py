from __future__ import annotations

import torch
import torch.nn.functional as F


class ClipAdaptersLoss:
    def __init__(self, cfg, adapter):
        self.cfg = cfg
        self.adapter = adapter
        self.base_kl_weight = float(cfg.CLIP_ADAPTERS.KL_WEIGHT)

    def __call__(self, outputs):
        loss = F.cross_entropy(outputs.logits, outputs.labels)
        extras = {'loss_ce': loss.detach()}

        if getattr(self.adapter, 'apply_constraint', 'none') != 'none':
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras['loss_constraint'] = constraint.detach()

        if hasattr(self.adapter, 'kl_divergence'):
            kl_weight = getattr(self.adapter, 'kl_weight', self.base_kl_weight)
            kl = self.adapter.kl_divergence() * kl_weight
            loss = loss + kl
            extras['loss_kl'] = kl.detach()

        outputs.losses.update(extras)
        return loss
