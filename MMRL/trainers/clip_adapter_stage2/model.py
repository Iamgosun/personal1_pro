from __future__ import annotations

import torch
import torch.nn as nn

from .adapter_variants import build_adapter
from .text import TextEncoder, get_base_text_features


class CustomCLIP(nn.Module):
    """Compatibility wrapper that preserves the original trainer contract.

    forward(image, return_features=True) -> logits, image_features
    forward_features(features) -> logits
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        text_encoder = TextEncoder(clip_model)

        if cfg.TRAINER.ClipADAPTER.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features, text_embeddings_all = get_base_text_features(
                cfg, classnames, clip_model, text_encoder
            )
        else:
            print(">> Use enhanced base!")
            base_text_features, text_embeddings_all = get_base_text_features(
                cfg,
                classnames,
                clip_model,
                text_encoder,
                cfg.TRAINER.TaskRes.ENHANCED_BASE,
            )

        self.text_embeddings_all = text_embeddings_all
        self.adapter = build_adapter(cfg, clip_model, base_text_features)
        self.to(clip_model.visual.conv1.weight.device)

    def forward(self, image, return_features: bool = False):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except Exception:
            image_features = self.image_encoder(image.float())

        logits = self.forward_features(image_features)
        if return_features:
            return logits, image_features
        return logits

    def forward_features(self, features):
        init = self.adapter.initialization
        if "TR" in init:
            return self.forward_task_residual(features)
        if "ClipA" in init:
            return self.forward_clipadapter(features)
        if "TipA" in init:
            return self.forward_tipadapter(features)
        if "CrossModal" in init or init == "RANDOM" or "ZS" in init:
            return self.forward_lp(features)
        return self.forward_bayes(features)

    def forward_lp(self, features):
        prototypes = self.adapter()
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return image_features_norm @ prototypes_norm.t() * logit_scale

    def forward_bayes(self, features):
        prototypes = self.adapter(n_samples=3)
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = (prototypes / prototypes.norm(dim=-1, keepdim=True)).to(image_features_norm.device)
        logit_scale = self.logit_scale.exp().to(image_features_norm.device)
        all_logits = torch.einsum("bd,scd->bsc", image_features_norm, prototypes_norm) * logit_scale
        logits_mean = all_logits.mean(dim=1)
        return logits_mean

    def forward_task_residual(self, features):
        residual = self.adapter()
        prototypes = self.adapter.base_text_features + self.adapter.alpha * residual
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return image_features_norm @ prototypes_norm.t() * logit_scale

    def forward_clipadapter(self, features):
        prototypes = self.adapter()
        x = self.adapter.mlp(features)
        features = self.adapter.ratio * x + (1 - self.adapter.ratio) * features
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return image_features_norm @ prototypes_norm.t() * logit_scale

    def forward_tipadapter(self, features):
        prototypes = self.adapter()
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        if self.adapter.cache_keys is not None:
            cache_keys = self.adapter.cache_keys / self.adapter.cache_keys.norm(dim=-1, keepdim=True)
            affinity = features @ cache_keys.t().to(features.device).to(torch.float)
            cache_logits = torch.exp(((-1) * (self.adapter.beta - self.adapter.beta * affinity))) @ self.adapter.cache_values.to(features.device).to(torch.float)
            logits += self.adapter.alpha * cache_logits
        return logits
