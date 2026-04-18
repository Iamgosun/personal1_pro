from __future__ import annotations

from core.registry import METHOD_REGISTRY
from methods.mmrl_family.base import BaseMMRLFamilyMethod
from .loss import MMRLMixLossAdapter


@METHOD_REGISTRY.register("MMRLMix")
class MMRLMixMethod(BaseMMRLFamilyMethod):
    method_name = "MMRLMix"
    cfg_section_name = "MMRL_MIX"
    clip_loader_name = "MMRL"
    loss_adapter_cls = MMRLMixLossAdapter

    def build_loss(self, method_cfg):
        """
        We override build_loss here because BaseMMRLFamilyMethod only passes
        (reg_weight, alpha). For branch specialization + complementarity regularization,
        we need a few extra hyperparameters, but we intentionally keep them
        code-local first so you can run this without touching config schema.

        If this version works well, you can later expose them into config.py/yaml.
        """
        return self.loss_adapter_cls(
            reg_weight=float(method_cfg.REG_WEIGHT),
            alpha=float(method_cfg.ALPHA),

            # === New hyperparameters ===
            # Rep branch focuses more on samples where main branch is uncertain.
            rep_focus_beta=2.0,

            # Keep rep branch from collapsing to main branch on negative classes.
            div_weight=0.05,
            div_margin=0.20,
        )