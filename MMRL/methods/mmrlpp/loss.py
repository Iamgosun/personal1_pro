from __future__ import annotations

from .modules import MMRLppLoss


class MMRLppLossAdapter:
    def __init__(self, reg_weight: float, alpha: float):
        self.loss_impl = MMRLppLoss(reg_weight=reg_weight, alpha=alpha)

    def __call__(self, outputs):
        return self.loss_impl(
            outputs.logits,
            outputs.aux_logits['rep'],
            outputs.features['img'],
            outputs.features['text'],
            outputs.features['img_ref'],
            outputs.features['text_ref'],
            outputs.labels,
        )
