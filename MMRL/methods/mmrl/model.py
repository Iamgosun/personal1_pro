from __future__ import annotations

from core.registry import METHOD_REGISTRY
from methods.mmrl_family.base import BaseMMRLFamilyMethod
from .loss import MMRLLossAdapter


@METHOD_REGISTRY.register("MMRL")
class MMRLMethod(BaseMMRLFamilyMethod):
    method_name = "MMRL"
    cfg_section_name = "MMRL"
    clip_loader_name = "MMRL"
    loss_adapter_cls = MMRLLossAdapter