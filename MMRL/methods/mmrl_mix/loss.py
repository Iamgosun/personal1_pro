from __future__ import annotations

import torch.nn as nn
from torch.nn import functional as F


class MMRLMixLoss(nn.Module):
    def __init__(self, reg_weight: float = 1.0, alpha: float = 0.7):
        super().__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def forward(
        self,
        logits,
        logits_rep,
        logits_fusion,
        image_features,
        text_features,
        image_features_clip,
        text_features_clip,
        label,
    ):
        xe_loss_main = F.cross_entropy(logits, label)
        xe_loss_rep = F.cross_entropy(logits_rep, label)
        xe_loss_fusion = F.cross_entropy(logits_fusion, label)

        cossim_reg_img = 1 - F.cosine_similarity(
            image_features, image_features_clip, dim=1
        ).mean()
        cossim_reg_text = 1 - F.cosine_similarity(
            text_features, text_features_clip, dim=1
        ).mean()

        return (
            xe_loss_fusion
            + self.alpha * xe_loss_main
            + (1.0 - self.alpha) * xe_loss_rep
            + self.reg_weight * cossim_reg_img
            + self.reg_weight * cossim_reg_text
        )


class MMRLMixLossAdapter:
    def __init__(self, reg_weight: float, alpha: float):
        self.loss_impl = MMRLMixLoss(reg_weight=reg_weight, alpha=alpha)

    def __call__(self, outputs):
        return self.loss_impl(
            outputs.logits,
            outputs.aux_logits["rep"],
            outputs.aux_logits["fusion"],
            outputs.features["img"],
            outputs.features["text"],
            outputs.features["img_ref"],
            outputs.features["text_ref"],
            outputs.labels,
        )