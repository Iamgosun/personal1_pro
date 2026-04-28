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
@METHOD_REGISTRY.register("CLAP")
class ClipAdaptersMethod(BaseMethod):
    method_name = "ClipAdapters"

    def build(self):
        if str(self.cfg.METHOD.NAME).upper() == "CLAP":
            self.method_name = "CLAP"

        clip_model = load_raw_clip_to_cpu(self.cfg)

        if self.cfg.CLIP_ADAPTERS.PREC in {"fp32", "amp"}:
            clip_model.float()

        classnames = self.dm.dataset.classnames
        self.model = ClipAdaptersModel(self.cfg, classnames, clip_model).to(self.device)

        enabled = freeze_all_but(self.model, ["adapter"])
        print(f"[{self.method_name}] trainable params before cache hooks: {enabled}")

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
        if not self._is_bayes_adapter():
            return float(self.cfg.CLIP_ADAPTERS.KL_WEIGHT)

        base = float(self.model.adapter.bayes_kl_base_weight())
        scale = float(getattr(self.cfg.CLIP_ADAPTERS, "BAYES_KL_SCALE", 1.0))
        return (float(self.current_epoch) / float(self.total_epochs)) * base * scale

    def get_precision(self) -> str:
        return self.cfg.CLIP_ADAPTERS.PREC

    def supports_cache(self) -> bool:
        return bool(getattr(self.cfg.CLIP_ADAPTERS, "ALLOW_CACHE", True))

    def _is_cross_modal(self) -> bool:
        return bool(getattr(self.model.adapter, "uses_cross_modal", False))

    def _prepare_cross_modal_cache_data(self, trainer):
        """
        CLAP CrossModal:
        after image feature extraction, add text prompt features as extra samples.
        """
        device = trainer.features_train.device
        text_embeddings_all = self.model.text_embeddings_all.detach().to(device=device, dtype=torch.float32)

        if text_embeddings_all.ndim != 3:
            raise ValueError(
                "CrossModal expects text_embeddings_all with shape [C, T, D], "
                f"got {tuple(text_embeddings_all.shape)}"
            )

        n_classes, n_templates, feat_dim = text_embeddings_all.shape
        text_features = text_embeddings_all.reshape(n_classes * n_templates, feat_dim)
        text_labels = (
            torch.arange(n_classes, device=device, dtype=torch.long)
            .repeat_interleave(n_templates)
        )

        resample = bool(getattr(self.cfg.CLIP_ADAPTERS, "CROSS_MODAL_RESAMPLE_TEXT", True))
        if resample:
            n_img = int(trainer.features_train.shape[0])
            idx = torch.randint(0, text_features.shape[0], (n_img,), device=device)
            text_features = text_features[idx]
            text_labels = text_labels[idx]

        trainer.features_train = torch.cat(
            [trainer.features_train.to(device=device, dtype=torch.float32), text_features],
            dim=0,
        )
        trainer.labels_train = torch.cat(
            [trainer.labels_train.to(device=device, dtype=torch.long), text_labels],
            dim=0,
        )

        # Keep logits aligned with the expanded feature pool.
        with torch.no_grad():
            head = self.model.forward_features(trainer.features_train)
            trainer.logits_train = head["logits"].detach()

        print(
            "[ClipAdaptersMethod] CrossModal cache data prepared: "
            f"features={tuple(trainer.features_train.shape)}, "
            f"labels={tuple(trainer.labels_train.shape)}"
        )

    def prepare_cache_data(self, trainer):
        if self._is_cross_modal():
            self._prepare_cross_modal_cache_data(trainer)

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

        # CLAP constraint initialization happens after feature extraction
        # and after CrossModal data expansion.
        if getattr(adapter, "apply_constraint", "none") != "none":
            print("[ClipAdaptersMethod] Initializing CLAP constraint multipliers")
            adapter.init_lagrangian_multipliers(
                trainer.labels_train.to(self.device),
                trainer.logits_train.to(self.device),
            )

        if hasattr(adapter, "build_cache"):
            adapter.build_cache(trainer.features_train, trainer.labels_train)

        # CLAP TipA plain path: one epoch only.
        if bool(getattr(adapter, "is_tip_adapter", False)):
            one_epoch = bool(getattr(self.cfg.CLIP_ADAPTERS, "CLAP_TIPA_ONE_EPOCH", True))
            if one_epoch and not bool(getattr(adapter, "finetune_cache", False)):
                trainer.max_epoch = 1
                print("[ClipAdaptersMethod] Plain TipA: set max_epoch=1 to match CLAP")