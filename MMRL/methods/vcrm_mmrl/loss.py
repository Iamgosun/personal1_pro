from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class VCRMMMRLLoss(nn.Module):
    """
    VCRM-MMRL loss.

    It keeps the original MMRL loss form:

        alpha * CE(f_c, w(x))
        + (1-alpha) * CE(f_r, w(x))
        + lambda * L_cos^v
        + lambda * L_cos^t

    but supports dynamic text prototypes:
        text_features: [B, C, d]

    and static text prototypes:
        text_features: [C, d]
    """

    def __init__(
        self,
        reg_weight: float = 1.0,
        alpha: float = 0.7,
        mod_weight: float = 0.0,
    ):
        super().__init__()
        self.reg_weight = float(reg_weight)
        self.alpha = float(alpha)
        self.mod_weight = float(mod_weight)

    @staticmethod
    def _text_cosine_regularization(
        text_features: torch.Tensor,
        text_features_clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_features:
                [C, d] or [B, C, d]
            text_features_clip:
                [C, d]
        """
        if text_features.dim() == 3:
            text_ref = text_features_clip.unsqueeze(0).expand_as(text_features)
            return 1.0 - F.cosine_similarity(
                text_features,
                text_ref,
                dim=-1,
            ).mean()

        return 1.0 - F.cosine_similarity(
            text_features,
            text_features_clip,
            dim=1,
        ).mean()

    def forward(
        self,
        logits,
        logits_rep,
        image_features,
        text_features,
        image_features_clip,
        text_features_clip,
        label,
        mod_loss=None,
    ):
        xe_loss_main = F.cross_entropy(logits, label)
        xe_loss_rep = F.cross_entropy(logits_rep, label)

        cossim_reg_img = 1.0 - F.cosine_similarity(
            image_features,
            image_features_clip,
            dim=1,
        ).mean()

        cossim_reg_text = self._text_cosine_regularization(
            text_features=text_features,
            text_features_clip=text_features_clip,
        )

        loss = (
            self.alpha * xe_loss_main
            + (1.0 - self.alpha) * xe_loss_rep
            + self.reg_weight * cossim_reg_img
            + self.reg_weight * cossim_reg_text
        )

        if mod_loss is not None and self.mod_weight > 0:
            loss = loss + self.mod_weight * mod_loss

        return loss


class VCRMMMRLLossAdapter:
    def __init__(
        self,
        reg_weight: float,
        alpha: float,
        mod_weight: float = 0.0,
    ):
        self.loss_impl = VCRMMMRLLoss(
            reg_weight=reg_weight,
            alpha=alpha,
            mod_weight=mod_weight,
        )

    def __call__(self, outputs):
        mod_loss = None
        if outputs.losses is not None:
            mod_loss = outputs.losses.get("mod_loss")

        return self.loss_impl(
            outputs.logits,
            outputs.aux_logits["rep"],
            outputs.features["img"],
            outputs.features["text"],
            outputs.features["img_ref"],
            outputs.features["text_ref"],
            outputs.labels,
            mod_loss=mod_loss,
        )