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