from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class MMRLMixLoss(nn.Module):
    """
    Branch Specialization for MMRL:
      1) main branch: keep the original stable classification target
      2) rep branch: focus more on samples that main branch is uncertain about
      3) complementarity regularization: discourage rep branch from collapsing
         to the same negative-class ranking as main branch

    Important:
      - No fusion CE here
      - Fusion is still used at inference exactly as before
      - No architecture change
    """

    def __init__(
        self,
        reg_weight: float = 0.5,
        alpha: float = 0.7,
        rep_focus_beta: float = 2.0,
        div_weight: float = 0.05,
        div_margin: float = 0.20,
    ):
        super().__init__()
        self.reg_weight = float(reg_weight)
        self.alpha = float(alpha)

        # specialization strength for rep branch
        self.rep_focus_beta = float(rep_focus_beta)

        # complementarity regularization
        self.div_weight = float(div_weight)
        self.div_margin = float(div_margin)

    @staticmethod
    def _build_rep_weights_from_main_confidence(
        logits: torch.Tensor,
        label: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """
        Rep branch should focus more on samples where main branch is uncertain.

        weight_i = 1 + beta * (1 - p_main(y_i))
        then normalize weights to mean 1 for stable optimization.
        """
        probs_main = F.softmax(logits, dim=1)
        p_true = probs_main.gather(1, label.unsqueeze(1)).squeeze(1).detach()

        weights = 1.0 + beta * (1.0 - p_true)
        weights = weights / weights.mean().clamp_min(1e-6)
        return weights

    @staticmethod
    def _negative_class_diversity_loss(
        logits_main: torch.Tensor,
        logits_rep: torch.Tensor,
        label: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """
        Keep rep branch complementary to main branch.

        We do NOT force full orthogonality.
        We only say:
          on negative classes, rep branch should not become too similar
          to the main branch.

        Implementation:
          1) remove GT class from both logits
          2) compare cosine similarity on negative logits
          3) only penalize if similarity > margin

        Main branch is detached here:
          - main branch should stay stable
          - rep branch is the one encouraged to diversify
        """
        batch_size, num_classes = logits_main.shape

        neg_mask = torch.ones_like(logits_main, dtype=torch.bool)
        neg_mask.scatter_(1, label.unsqueeze(1), False)

        main_neg = logits_main[neg_mask].view(batch_size, num_classes - 1).detach()
        rep_neg = logits_rep[neg_mask].view(batch_size, num_classes - 1)

        main_neg = F.normalize(main_neg, dim=1)
        rep_neg = F.normalize(rep_neg, dim=1)

        neg_cos = (main_neg * rep_neg).sum(dim=1)
        div_loss = F.relu(neg_cos - margin).mean()
        return div_loss

    def forward(
        self,
        logits: torch.Tensor,
        logits_rep: torch.Tensor,
        logits_fusion: torch.Tensor,  # kept only for interface compatibility
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_features_clip: torch.Tensor,
        text_features_clip: torch.Tensor,
        label: torch.Tensor,
        return_parts: bool = False,
    ):
        # 1) original main branch CE
        xe_loss_main = F.cross_entropy(logits, label)

        # 2) uncertainty-aware rep branch CE
        rep_weights = self._build_rep_weights_from_main_confidence(
            logits=logits,
            label=label,
            beta=self.rep_focus_beta,
        )
        xe_rep_per_sample = F.cross_entropy(logits_rep, label, reduction="none")
        xe_loss_rep = (xe_rep_per_sample * rep_weights).mean()

        # 3) original MMRL cosine regularization
        cossim_reg_img = 1 - F.cosine_similarity(
            image_features, image_features_clip, dim=1
        ).mean()
        cossim_reg_text = 1 - F.cosine_similarity(
            text_features, text_features_clip, dim=1
        ).mean()

        # 4) complementarity regularization
        div_loss = self._negative_class_diversity_loss(
            logits_main=logits,
            logits_rep=logits_rep,
            label=label,
            margin=self.div_margin,
        )

        total = (
            self.alpha * xe_loss_main
            + (1.0 - self.alpha) * xe_loss_rep
            + self.reg_weight * cossim_reg_img
            + self.reg_weight * cossim_reg_text
            + self.div_weight * div_loss
        )

        if not return_parts:
            return total

        parts = {
            "loss_total": total.detach(),
            "loss_main_ce": xe_loss_main.detach(),
            "loss_rep_ce_weighted": xe_loss_rep.detach(),
            "loss_div": div_loss.detach(),
            "loss_reg_img": cossim_reg_img.detach(),
            "loss_reg_text": cossim_reg_text.detach(),
            "rep_weight_mean": rep_weights.mean().detach(),
            "rep_weight_max": rep_weights.max().detach(),
            "rep_weight_min": rep_weights.min().detach(),
        }
        return total, parts


class MMRLMixLossAdapter:
    def __init__(
        self,
        reg_weight: float,
        alpha: float,
        rep_focus_beta: float = 2.0,
        div_weight: float = 0.05,
        div_margin: float = 0.20,
    ):
        self.loss_impl = MMRLMixLoss(
            reg_weight=reg_weight,
            alpha=alpha,
            rep_focus_beta=rep_focus_beta,
            div_weight=div_weight,
            div_margin=div_margin,
        )

    def __call__(self, outputs):
        total, parts = self.loss_impl(
            outputs.logits,
            outputs.aux_logits["rep"],
            outputs.aux_logits["fusion"],
            outputs.features["img"],
            outputs.features["text"],
            outputs.features["img_ref"],
            outputs.features["text_ref"],
            outputs.labels,
            return_parts=True,
        )
        outputs.losses.update(parts)
        outputs.losses["total"] = total.detach()
        return total