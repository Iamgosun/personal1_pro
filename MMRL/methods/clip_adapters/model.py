from __future__ import annotations

import torch
import torch.nn as nn

from backbones.clip_loader import load_raw_clip_to_cpu
from backbones.freeze import freeze_all_but
from backbones.text_encoders import CLIPTextEncoder, build_base_text_features
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.base import BaseMethod
from .adapter_router import build_adapter
from .loss import ClipAdaptersLoss
from .online_heads import bayes_logits, lp_logits


class ClipAdaptersModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        text_encoder = CLIPTextEncoder(clip_model)
        if cfg.CLIP_ADAPTERS.ENHANCED_BASE == "none":
            base_text_features, text_embeddings_all = build_base_text_features(
                cfg, classnames, clip_model, text_encoder
            )
        else:
            base_text_features, text_embeddings_all = build_base_text_features(
                cfg,
                classnames,
                clip_model,
                text_encoder,
                cfg.CLIP_ADAPTERS.ENHANCED_BASE,
            )
        self.text_embeddings_all = text_embeddings_all
        self.adapter = build_adapter(cfg, clip_model, base_text_features)
        self.to(clip_model.visual.conv1.weight.device)

    def forward(self, image, return_features=False):
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
            residual = self.adapter()
            prototypes = self.adapter.base_text_features + self.adapter.alpha * residual
            return lp_logits(self.logit_scale, features, prototypes)
        if "ClipA" in init:
            prototypes = self.adapter()
            x = self.adapter.mlp(features)
            features = self.adapter.ratio * x + (1 - self.adapter.ratio) * features
            return lp_logits(self.logit_scale, features, prototypes)
        if "TipA" in init:
            prototypes = self.adapter()
            logits = lp_logits(self.logit_scale, features, prototypes)
            if self.adapter.cache_keys is not None:
                cache_keys = self.adapter.cache_keys / self.adapter.cache_keys.norm(
                    dim=-1, keepdim=True
                )
                affinity = features @ cache_keys.t().to(features.device).to(torch.float)
                cache_logits = torch.exp(
                    ((-1) * (self.adapter.beta - self.adapter.beta * affinity))
                ) @ self.adapter.cache_values.to(features.device).to(torch.float)
                logits = logits + self.adapter.alpha * cache_logits
            return logits
        if "CrossModal" in init or init == "RANDOM" or "ZS" in init:
            prototypes = self.adapter()
            return lp_logits(self.logit_scale, features, prototypes)
        prototypes = self.adapter(
            n_samples=self.adapter.cfg.CLIP_ADAPTERS.N_SAMPLES
            if hasattr(self.adapter, "cfg")
            else 3
        )
        return bayes_logits(self.logit_scale, features, prototypes)


@METHOD_REGISTRY.register("ClipAdapters")
@METHOD_REGISTRY.register("ClipADAPTER")
class ClipAdaptersMethod(BaseMethod):
    method_name = "ClipAdapters"

    def build(self):
        clip_model = load_raw_clip_to_cpu(self.cfg)
        if self.cfg.CLIP_ADAPTERS.PREC in {"fp32", "amp"}:
            clip_model.float()
        classnames = self.dm.dataset.classnames
        self.model = ClipAdaptersModel(self.cfg, classnames, clip_model).to(self.device)
        enabled = freeze_all_but(self.model, ["adapter"])
        print(f"[ClipAdaptersMethod] trainable params: {enabled}")
        self.loss = ClipAdaptersLoss(self.cfg, self.model.adapter)
        return self

    def get_precision(self) -> str:
        return self.cfg.CLIP_ADAPTERS.PREC

    def supports_cache(self) -> bool:
        return True

    def forward_train(self, batch):
        label = batch["label"].to(self.device)
        if "features" in batch:
            features = batch["features"].to(self.device)
            logits = self.model.forward_features(features)
            return MethodOutputs(
                logits=logits,
                labels=label,
                features={"img": features},
                extras={"mode": "cache"},
            )
        image = batch["img"].to(self.device)
        logits, image_features = self.model(image, return_features=True)
        return MethodOutputs(
            logits=logits,
            labels=label,
            features={"img": image_features},
            extras={"mode": "online"},
        )

    def forward_eval(self, batch, eval_ctx):
        return self.forward_train(batch)

    def on_cache_ready(self, trainer):
        adapter = self.model.adapter
        if hasattr(adapter, "init_tipadapter"):
            adapter.init_tipadapter(trainer.features_train, trainer.labels_train)