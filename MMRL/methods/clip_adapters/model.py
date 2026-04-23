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
from .online_heads import bayes_logits, bayes_logits_all, lp_logits


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

    def _is_bayes_adapter(self) -> bool:
        return str(getattr(self.adapter, "initialization_name", "")).upper() == "BAYES_ADAPTER"

    def forward(self, image, return_features=False, n_samples=None):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except Exception:
            image_features = self.image_encoder(image.float())

        head = self.forward_features(image_features, n_samples=n_samples)

        if return_features:
            return head["logits"], image_features, head["logits_all"]
        return head["logits"]

    def forward_features(self, features, n_samples=None):
        features_for_logits = self.adapter.adapt_features(features)
        logits_all = None

        if self.adapter.adapter_kind == "stochastic_prototype":
            if n_samples is None:
                n_samples = int(getattr(self.adapter.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3))

            prototypes = self.adapter.sample_prototypes(n_samples=n_samples)

            if self._is_bayes_adapter():
                logits_all = bayes_logits_all(self.logit_scale, features_for_logits, prototypes)
                logits = logits_all.mean(dim=0)
            else:
                logits = bayes_logits(self.logit_scale, features_for_logits, prototypes)
        else:
            prototypes = self.adapter.get_prototypes()
            logits = lp_logits(self.logit_scale, features_for_logits, prototypes)

        cache_logits = self.adapter.cache_logits(features_for_logits)
        if cache_logits is not None:
            logits = logits + cache_logits
            if logits_all is not None:
                logits_all = logits_all + cache_logits.unsqueeze(0)

        return {
            "logits": logits,
            "logits_all": logits_all,
            "features_for_logits": features_for_logits,
        }


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

        self.current_epoch = 0
        self.total_epochs = 1

        self.loss = ClipAdaptersLoss(self.cfg, self.model.adapter)
        return self

    def set_epoch_context(self, epoch, total_epochs):
        self.current_epoch = int(epoch)
        self.total_epochs = max(1, int(total_epochs))

    def _is_bayes_adapter(self) -> bool:
        return str(getattr(self.model.adapter, "initialization_name", "")).upper() == "BAYES_ADAPTER"

    def _bayes_kl_weight(self) -> float:
        """
        Match the uploaded BayesAdapter schedule:
            kl_weight = (epoch / num_epochs) * 1 / (1000 * C * D)
        """
        if not self._is_bayes_adapter():
            return float(self.cfg.CLIP_ADAPTERS.KL_WEIGHT)

        base = float(self.model.adapter.bayes_kl_base_weight())
        scale = float(getattr(self.cfg.CLIP_ADAPTERS, "BAYES_KL_SCALE", 1.0))
        return (float(self.current_epoch) / float(self.total_epochs)) * base * scale

    def get_precision(self) -> str:
        return self.cfg.CLIP_ADAPTERS.PREC

    def supports_cache(self) -> bool:
        return bool(getattr(self.cfg.CLIP_ADAPTERS, "ALLOW_CACHE", True))

    def forward_train(self, batch):
        label = batch["label"].to(self.device)
        n_train_samples = int(getattr(self.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3))

        aux_logits = {}
        extras = {}

        if "features" in batch:
            features = batch["features"].to(self.device)
            head = self.model.forward_features(features, n_samples=n_train_samples)

            if self._is_bayes_adapter() and head["logits_all"] is not None:
                aux_logits["bayes_logits_all"] = head["logits_all"]
                extras["bayes_kl_weight"] = self._bayes_kl_weight()

            return MethodOutputs(
                logits=head["logits"],
                labels=label,
                aux_logits=aux_logits,
                features={"img": head["features_for_logits"]},
                extras={"mode": "cache", **extras},
            )

        image = batch["img"].to(self.device)
        logits, image_features, logits_all = self.model(
            image,
            return_features=True,
            n_samples=n_train_samples,
        )

        if self._is_bayes_adapter() and logits_all is not None:
            aux_logits["bayes_logits_all"] = logits_all
            extras["bayes_kl_weight"] = self._bayes_kl_weight()

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits=aux_logits,
            features={"img": image_features},
            extras={"mode": "online", **extras},
        )

    def forward_eval(self, batch, eval_ctx):
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        n_test_samples = int(
            getattr(
                self.cfg.CLIP_ADAPTERS,
                "N_TEST_SAMPLES",
                10 if self._is_bayes_adapter() else getattr(self.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3),
            )
        )

        if "features" in batch:
            features = batch["features"].to(self.device)
            head = self.model.forward_features(features, n_samples=n_test_samples)
            return MethodOutputs(
                logits=head["logits"],
                labels=label,
                features={"img": head["features_for_logits"]},
                extras={"mode": "cache_eval"},
            )

        image = batch["img"].to(self.device)
        logits, image_features, _ = self.model(
            image,
            return_features=True,
            n_samples=n_test_samples,
        )
        return MethodOutputs(
            logits=logits,
            labels=label,
            features={"img": image_features},
            extras={"mode": "online_eval"},
        )

    def on_cache_ready(self, trainer):
        adapter = self.model.adapter
        if hasattr(adapter, "build_cache"):
            adapter.build_cache(trainer.features_train, trainer.labels_train)