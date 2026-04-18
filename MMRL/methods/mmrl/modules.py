from __future__ import annotations

from methods.mmrl_family.modules import (
    CLIPTextEncoderPlain,
    MMRLTextEncoder,
    build_zero_shot_text_features,
    MMRLFamilyRepresentationLearner,
    MMRLFamilyModel,
)


class MultiModalRepresentationLearner(MMRLFamilyRepresentationLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, cfg.MMRL, classnames, clip_model)


class CustomMMRLModel(MMRLFamilyModel):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, cfg.MMRL, classnames, clip_model)


__all__ = [
    "CLIPTextEncoderPlain",
    "MMRLTextEncoder",
    "build_zero_shot_text_features",
    "MultiModalRepresentationLearner",
    "CustomMMRLModel",
]