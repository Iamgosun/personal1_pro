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
        features_for_logits = self.adapter.adapt_features(features)

        if self.adapter.adapter_kind == "stochastic_prototype":
            n_samples = int(getattr(self.adapter.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3))
            prototypes = self.adapter.sample_prototypes(n_samples=n_samples)
            logits = bayes_logits(self.logit_scale, features_for_logits, prototypes)
        else:
            prototypes = self.adapter.get_prototypes()
            logits = lp_logits(self.logit_scale, features_for_logits, prototypes)

        cache_logits = self.adapter.cache_logits(features_for_logits)
        if cache_logits is not None:
            logits = logits + cache_logits

        return logits


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
        return bool(getattr(self.cfg.CLIP_ADAPTERS, "ALLOW_CACHE", True))

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
        if hasattr(adapter, "build_cache"):
            adapter.build_cache(trainer.features_train, trainer.labels_train)