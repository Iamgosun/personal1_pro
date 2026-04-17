from __future__ import annotations


class MMRLppLossAdapter:
    def __init__(self, legacy_loss):
        self.legacy_loss = legacy_loss

    def __call__(self, outputs):
        return self.legacy_loss(
            outputs.logits,
            outputs.aux_logits['rep'],
            outputs.features['img'],
            outputs.features['text'],
            outputs.features['img_ref'],
            outputs.features['text_ref'],
            outputs.labels,
        )
