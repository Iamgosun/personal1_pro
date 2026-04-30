from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler

from backbones.clip_loader import load_raw_clip_to_cpu
from backbones.freeze import freeze_all_but
from backbones.text_encoders import CLIPTextEncoder, build_base_text_features
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from data.build import build_split_dataset
from methods.base import BaseMethod
from .adapter_router import build_adapter
from .loss import ClipAdaptersLoss
from .online_heads import bayes_logits, bayes_logits_all, capel_logits, lp_logits
import torch.nn.functional as F


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
        self.adapter = build_adapter(
            cfg,
            clip_model,
            base_text_features,
            classnames=classnames,
        )

        self.to(clip_model.visual.conv1.weight.device)

    def _is_bayes_adapter(self) -> bool:
        return str(getattr(self.adapter, "initialization_name", "")).upper() == "BAYES_ADAPTER"

    @staticmethod
    def _mc_predictive_log_probs(logits_all: torch.Tensor) -> torch.Tensor:
        """
        BayesAdapter posterior predictive in probability space:

            log E_W[softmax(logits(W))]

        This matches the DREAM theory better than softmax(E_W[logits]).
        Returns log-probabilities with shape [B, C].
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"_mc_predictive_log_probs expects logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )
        probs = torch.softmax(logits_all.float(), dim=-1).mean(dim=0).clamp_min(1e-12)
        return torch.log(probs).to(dtype=logits_all.dtype)

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
        sub_logits = None
        assignment_logits = None

        if self.adapter.adapter_kind in {"capel_prototype", "vnccapel_prototype"}:
            logits, sub_logits = capel_logits(
                self.logit_scale,
                features_for_logits,
                self.adapter.prototypes,
                self.adapter.get_prompt_weights(),
            )

            if self.adapter.adapter_kind == "vnccapel_prototype":
                # VNC uses unscaled cosine logits for prompt assignment.
                features_norm = F.normalize(features_for_logits.float(), dim=-1)
                prototypes_norm = F.normalize(self.adapter.prototypes.float(), dim=-1)
                assignment_logits = torch.einsum(
                    "bd,ckd->bck",
                    features_norm,
                    prototypes_norm,
                )

        elif self.adapter.adapter_kind == "stochastic_prototype":
            if n_samples is None:
                n_samples = int(getattr(self.adapter.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3))

            prototypes = self.adapter.sample_prototypes(n_samples=n_samples)

            if self._is_bayes_adapter():
                logits_all = bayes_logits_all(self.logit_scale, features_for_logits, prototypes)

                # Official BayesAdapter-style MC aggregation:
                # average logits over sampled classifiers.
                logits = logits_all.mean(dim=0)
            else:
                logits = bayes_logits(self.logit_scale, features_for_logits, prototypes)

        else:
            prototypes = self.adapter.get_prototypes()
            logits = lp_logits(self.logit_scale, features_for_logits, prototypes)

        # Optional additive evidence/cache logits.
        # For DREAM this implements:
        #   log p_cls(k|z) = log p_B(k|z) + lambda * standardized_density_ratio_k(z)
        cache_logits = self.adapter.cache_logits(features_for_logits)
        if cache_logits is not None:
            logits = logits + cache_logits

        # Optional probability-space postprocess.
        # For DREAM this implements:
        #   p_final = rho-gated mixture of p_cls and uniform.
        if hasattr(self.adapter, "postprocess_logits"):
            logits = self.adapter.postprocess_logits(
                logits,
                features_for_logits,
                training=bool(self.training),
            )

        return {
            "logits": logits,
            "logits_all": logits_all,
            "sub_logits": sub_logits,
            "assignment_logits": assignment_logits,
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

        # Online CrossModal state. Built in on_fit_start(), not saved to disk.
        self.online_text_features = None
        self.online_text_labels = None

        self.loss = ClipAdaptersLoss(self.cfg, self.model.adapter)
        return self

    def set_epoch_context(self, epoch, total_epochs):
        self.current_epoch = int(epoch)
        self.total_epochs = max(1, int(total_epochs))

    def _is_bayes_adapter(self) -> bool:
        return str(getattr(self.model.adapter, "initialization_name", "")).upper() == "BAYES_ADAPTER"

    def _is_tip_adapter(self) -> bool:
        return bool(getattr(self.model.adapter, "is_tip_adapter", False))

    def _is_cross_modal(self) -> bool:
        return bool(getattr(self.model.adapter, "uses_cross_modal", False))

    def _add_capel_aux(self, head, aux_logits):
        if head.get("sub_logits") is not None:
            aux_logits["capel_sub_logits"] = head["sub_logits"]

        if head.get("assignment_logits") is not None:
            aux_logits["vnc_assignment_logits"] = head["assignment_logits"]

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

    def _needs_online_support_features(self) -> bool:
        """
        Online mode should not use disk cold cache, but some adapters still need
        transient support features/statistics before training/eval.

        - CLAP constraint needs labels_train/logits_train.
        - TipA/TipA-f need support features to build cache_keys/cache_values.
        - DREAM-BayesAdapter needs support features to fit density evidence.
        """
        adapter = self.model.adapter

        if bool(getattr(adapter, "needs_support_features", False)):
            return True

        if getattr(adapter, "apply_constraint", "none") != "none":
            return True

        if self._is_tip_adapter():
            return True

        return False

    def _online_prefit_reps(self) -> int:
        if hasattr(self.cfg.CLIP_ADAPTERS, "ONLINE_PREFIT_REPS"):
            reps = int(getattr(self.cfg.CLIP_ADAPTERS, "ONLINE_PREFIT_REPS"))
        else:
            reps = int(getattr(self.cfg.CLIP_ADAPTERS, "CACHE_REPS", 1))

        return max(1, reps)

    def _online_prefit_train_aug(self) -> bool:
        if hasattr(self.cfg.CLIP_ADAPTERS, "ONLINE_PREFIT_TRAIN_AUG"):
            return bool(getattr(self.cfg.CLIP_ADAPTERS, "ONLINE_PREFIT_TRAIN_AUG"))
        return bool(getattr(self.cfg.CLIP_ADAPTERS, "CACHE_TRAIN_AUG", True))

    def _build_online_prefit_loader(self, trainer):
        """
        Build a loader for transient online support-feature prefit.

        If ONLINE_PREFIT_TRAIN_AUG=True, reuse trainer.train_loader_x so the same
        train-time augmentation pipeline is used.

        If ONLINE_PREFIT_TRAIN_AUG=False, build a deterministic train split loader
        with eval transform.
        """
        train_aug = self._online_prefit_train_aug()

        if train_aug:
            return trainer.train_loader_x

        dataset = build_split_dataset(
            trainer.cfg,
            trainer.dm.dataset.train_x,
            is_train=False,
        )

        return DataLoader(
            dataset,
            batch_size=trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            sampler=SequentialSampler(dataset),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    @torch.no_grad()
    def _collect_online_support_features(self, trainer):
        """
        Build a transient in-memory support feature pool for online mode.

        This does NOT write disk cache. It only collects realtime CLIP features
        and logits from support images before optimizer creation.

        Repeated passes are concatenated, not averaged:
            reps x N -> reps*N
        """
        reps = self._online_prefit_reps()

        model_was_training = bool(self.model.training)
        method_was_training = bool(self.training)

        self.model.eval()
        self.eval()

        labels_all = []
        logits_all = []
        features_all = []

        for rep_idx in range(reps):
            print(
                f"[ClipAdaptersMethod] online prefit support pass "
                f"{rep_idx + 1}/{reps}"
            )

            loader = self._build_online_prefit_loader(trainer)

            for batch in loader:
                image = batch["img"].to(self.device)
                label = batch["label"].to(self.device)

                try:
                    image_features = self.model.image_encoder(image.type(self.model.dtype))
                except Exception:
                    image_features = self.model.image_encoder(image.float())

                head = self.model.forward_features(image_features)

                labels_all.append(label.detach().cpu())
                logits_all.append(head["logits"].detach().cpu())
                features_all.append(image_features.detach().cpu())

        if model_was_training:
            self.model.train()
        if method_was_training:
            self.train()

        if not labels_all:
            raise RuntimeError(
                "Online support feature prefit found no batches in trainer.train_loader_x."
            )

        labels_train = torch.cat(labels_all, dim=0).to(self.device)
        logits_train = torch.cat(logits_all, dim=0).to(self.device)
        features_train = torch.cat(features_all, dim=0).to(self.device)

        print(
            "[ClipAdaptersMethod] online support features built: "
            f"features={tuple(features_train.shape)}, "
            f"labels={tuple(labels_train.shape)}, "
            f"logits={tuple(logits_train.shape)}"
        )

        return labels_train, logits_train, features_train

    def _build_online_cross_modal_text_pool(self):
        """
        Build CrossModal text prompt feature pool for online mode.

        In cache mode, text prompt features are concatenated into cached train
        features. In online mode, image features are produced per batch, so
        we keep prompt features in memory and sample them in forward_train().
        """
        text_embeddings_all = self.model.text_embeddings_all.detach().to(
            device=self.device,
            dtype=torch.float32,
        )

        if text_embeddings_all.ndim != 3:
            raise ValueError(
                "CrossModal expects text_embeddings_all with shape [C, T, D], "
                f"got {tuple(text_embeddings_all.shape)}"
            )

        n_classes, n_templates, feat_dim = text_embeddings_all.shape
        text_features = text_embeddings_all.reshape(n_classes * n_templates, feat_dim)
        text_labels = (
            torch.arange(n_classes, device=self.device, dtype=torch.long)
            .repeat_interleave(n_templates)
        )

        self.online_text_features = text_features
        self.online_text_labels = text_labels

        print(
            "[ClipAdaptersMethod] online CrossModal text pool built: "
            f"features={tuple(text_features.shape)}, "
            f"labels={tuple(text_labels.shape)}"
        )

    def _sample_online_cross_modal_text(self, n: int):
        if self.online_text_features is None or self.online_text_labels is None:
            return None, None

        n_text = int(self.online_text_features.shape[0])
        if n_text <= 0:
            return None, None

        idx = torch.randint(0, n_text, (int(n),), device=self.device)
        return self.online_text_features[idx], self.online_text_labels[idx]

    def _apply_tip_adapter_one_epoch(self, trainer):
        adapter = self.model.adapter

        if not self._is_tip_adapter():
            return

        one_epoch = bool(getattr(self.cfg.CLIP_ADAPTERS, "CLAP_TIPA_ONE_EPOCH", True))
        if one_epoch and not bool(getattr(adapter, "finetune_cache", False)):
            trainer.max_epoch = 1
            print("[ClipAdaptersMethod] Plain TipA: set max_epoch=1 to match CLAP")

    def _prepare_cross_modal_cache_data(self, trainer):
        """
        CLAP CrossModal cache mode:
        after image feature extraction, add text prompt features as extra samples.
        """
        device = trainer.features_train.device
        text_embeddings_all = self.model.text_embeddings_all.detach().to(
            device=device,
            dtype=torch.float32,
        )

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

    def on_fit_start(self, trainer):
        """
        Called after executor.on_build() and before optimizer creation.

        Online mode:
        - no disk cold cache
        - build transient support features only when needed
        - build transient text pool for CrossModal
        """
        exec_mode = str(getattr(self.cfg.METHOD, "EXEC_MODE", "online")).lower()
        if exec_mode != "online":
            return None

        adapter = self.model.adapter

        if self._is_cross_modal():
            self._build_online_cross_modal_text_pool()

        labels_train = None
        logits_train = None
        features_train = None

        if self._needs_online_support_features():
            labels_train, logits_train, features_train = self._collect_online_support_features(
                trainer
            )

            # Expose these for debugging and cache-style compatibility.
            trainer.labels_train = labels_train
            trainer.logits_train = logits_train
            trainer.features_train = features_train

        if getattr(adapter, "apply_constraint", "none") != "none":
            if labels_train is None or logits_train is None:
                raise RuntimeError(
                    "CLAP constraint in online mode requires support logits/labels, "
                    "but online prefit did not build them."
                )

            print("[ClipAdaptersMethod] Online: initializing CLAP constraint multipliers")
            adapter.init_lagrangian_multipliers(
                labels_train.to(self.device),
                logits_train.to(self.device),
            )

        if self._is_tip_adapter():
            if labels_train is None or features_train is None:
                raise RuntimeError(
                    "TipA online mode requires support features/labels, "
                    "but online prefit did not build them."
                )

            print("[ClipAdaptersMethod] Online: building TipA support feature bank")
            adapter.build_cache(
                features_train.to(self.device),
                labels_train.to(self.device),
            )

        if bool(getattr(adapter, "needs_support_features", False)):
            if labels_train is None or features_train is None:
                raise RuntimeError(
                    f"{adapter.__class__.__name__} requires support features, "
                    "but online prefit did not build them."
                )
            print(f"[ClipAdaptersMethod] Online: fitting {adapter.__class__.__name__} support head")
            adapter.build_cache(
                features_train.to(self.device),
                labels_train.to(self.device),
            )

        self._apply_tip_adapter_one_epoch(trainer)
        return None

    def forward_train(self, batch):
        label = batch["label"].to(self.device)
        n_train_samples = int(getattr(self.cfg.CLIP_ADAPTERS, "N_SAMPLES", 3))

        aux_logits = {}
        extras = {}

        if "features" in batch:
            features = batch["features"].to(self.device)
            head = self.model.forward_features(features, n_samples=n_train_samples)

            self._add_capel_aux(head, aux_logits)

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

        try:
            image_features = self.model.image_encoder(image.type(self.model.dtype))
        except Exception:
            image_features = self.model.image_encoder(image.float())

        # Online CrossModal: mix realtime image features with sampled prompt features.
        if self._is_cross_modal() and self.online_text_features is not None:
            text_features, text_labels = self._sample_online_cross_modal_text(
                n=int(image_features.shape[0])
            )

            if text_features is not None:
                mixed_features = torch.cat(
                    [image_features.to(torch.float32), text_features.to(torch.float32)],
                    dim=0,
                )
                mixed_labels = torch.cat([label, text_labels.to(label.device)], dim=0)

                head = self.model.forward_features(
                    mixed_features,
                    n_samples=n_train_samples,
                )

                self._add_capel_aux(head, aux_logits)

                if self._is_bayes_adapter() and head["logits_all"] is not None:
                    aux_logits["bayes_logits_all"] = head["logits_all"]
                    extras["bayes_kl_weight"] = self._bayes_kl_weight()

                return MethodOutputs(
                    logits=head["logits"],
                    labels=mixed_labels,
                    aux_logits=aux_logits,
                    features={"img": head["features_for_logits"]},
                    extras={"mode": "online_crossmodal", **extras},
                )

        head = self.model.forward_features(image_features, n_samples=n_train_samples)

        self._add_capel_aux(head, aux_logits)

        if self._is_bayes_adapter() and head["logits_all"] is not None:
            aux_logits["bayes_logits_all"] = head["logits_all"]
            extras["bayes_kl_weight"] = self._bayes_kl_weight()

        return MethodOutputs(
            logits=head["logits"],
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

        if getattr(adapter, "apply_constraint", "none") != "none":
            print("[ClipAdaptersMethod] Initializing CLAP constraint multipliers")
            adapter.init_lagrangian_multipliers(
                trainer.labels_train.to(self.device),
                trainer.logits_train.to(self.device),
            )

        if hasattr(adapter, "build_cache"):
            adapter.build_cache(trainer.features_train, trainer.labels_train)

        self._apply_tip_adapter_one_epoch(trainer)
