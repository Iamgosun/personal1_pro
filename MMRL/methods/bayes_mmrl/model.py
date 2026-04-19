from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from backbones.clip_loader import load_mmrl_clip_to_cpu
from backbones.freeze import freeze_all_but
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from data.build import build_split_dataset
from methods.base import BaseMethod
from methods.mmrl.loss import MMRLLoss

from .loss import BayesMMRLLossAdapter
from .modules import (
    CLIPTextEncoderPlain,
    BayesianCustomMMRLModel,
    build_zero_shot_text_features,
)


@METHOD_REGISTRY.register("BayesMMRL")
class BayesMMRLMethod(BaseMethod):
    method_name = "BayesMMRL"

    def _resolve_trainable_substrings(self):
        substrings = ["representation_learner"]
        if self.bayes_target == "proj_rep":
            substrings.append("image_encoder.bayes_proj_rep")
        else:
            substrings.append("image_encoder.proj_rep")
        return substrings

    def _build_support_loader(self):
        dataset = build_split_dataset(
            self.cfg,
            self.dm.dataset.train_x,
            is_train=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            sampler=SequentialSampler(dataset),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    @torch.no_grad()
    def _compute_support_image_class_prototypes(self):
        loader = self._build_support_loader()
        feat_dim = self.text_features_clip.shape[-1]

        feat_sums = torch.zeros(
            self.num_classes,
            feat_dim,
            device=self.device,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            self.num_classes,
            device=self.device,
            dtype=torch.float32,
        )

        for batch in loader:
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            img_features = self.image_encoder_clip(images.type(self.dtype))
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            feat_sums.index_add_(0, labels, img_features.float())
            counts.index_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=torch.float32),
            )

        counts = counts.clamp_min(1.0).unsqueeze(-1)
        protos = feat_sums / counts
        protos = protos / protos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return protos

    @torch.no_grad()
    def _build_rep_clip_prior(self):
        bayes_cfg = self.cfg.BAYES_MMRL
        rep_prior_mode = str(getattr(bayes_cfg, "REP_PRIOR_MODE", "zero"))
        rep_dim = int(bayes_cfg.REP_DIM)
        n_rep_tokens = int(bayes_cfg.N_REP_TOKENS)

        text_basis = self.text_features_clip[: self.num_classes].float()
        if text_basis.shape[-1] != rep_dim:
            raise ValueError(
                f"CLIP prior requires REP_DIM == CLIP embed dim, got {rep_dim} vs {text_basis.shape[-1]}"
            )

        if rep_prior_mode == "clip_text":
            centers = text_basis
        elif rep_prior_mode == "clip_joint":
            image_basis = self._compute_support_image_class_prototypes()
            blend = float(getattr(bayes_cfg, "CLIP_PRIOR_BLEND", 0.5))
            centers = (1.0 - blend) * text_basis + blend * image_basis
        else:
            raise ValueError(f"Unsupported REP_PRIOR_MODE: {rep_prior_mode}")

        centers = centers - centers.mean(dim=0, keepdim=True)

        if torch.allclose(centers, torch.zeros_like(centers)):
            centers = text_basis

        _, _, vh = torch.linalg.svd(centers, full_matrices=False)
        basis = vh[: min(n_rep_tokens, vh.shape[0])]

        if basis.shape[0] < n_rep_tokens:
            extra = torch.randn(
                n_rep_tokens - basis.shape[0],
                basis.shape[1],
                device=basis.device,
                dtype=basis.dtype,
            )
            basis = torch.cat([basis, extra], dim=0)

        basis = F.normalize(basis, dim=-1)
        scale = float(getattr(bayes_cfg, "CLIP_PRIOR_SCALE", 0.05))
        return (scale * basis).detach()

    def build(self):
        cfg = self.cfg
        bayes_cfg = cfg.BAYES_MMRL
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        self.bayes_target = str(getattr(bayes_cfg, "BAYES_TARGET", "rep_tokens"))

        clip_model = load_mmrl_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_mmrl_clip_to_cpu(cfg, "CLIP")

        if bayes_cfg.PREC in {"fp32", "amp"}:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype
        self.n_mc_train = max(1, int(bayes_cfg.N_MC_TRAIN))
        self.n_mc_test = max(1, int(bayes_cfg.N_MC_TEST))
        self.eval_use_posterior_mean = bool(bayes_cfg.EVAL_USE_POSTERIOR_MEAN)

        self.rep_kl_weight = float(
            getattr(
                bayes_cfg,
                "REP_KL_WEIGHT",
                getattr(bayes_cfg, "KL_WEIGHT", 1e-4),
            )
        )
        self.proj_rep_kl_weight = float(
            getattr(bayes_cfg, "PROJ_REP_KL_WEIGHT", 1e-6)
        )

        self.text_encoder_clip = CLIPTextEncoderPlain(clip_model_zero_shot).to(
            self.device
        )

        with torch.no_grad():
            text_features_clip = build_zero_shot_text_features(
                cfg,
                classnames,
                clip_model_zero_shot,
                self.text_encoder_clip,
            )
            self.text_features_clip = (
                text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
            ).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)

        self.model = BayesianCustomMMRLModel(cfg, classnames, clip_model).to(
            self.device
        )

        rep_prior_mode = str(getattr(bayes_cfg, "REP_PRIOR_MODE", "zero"))
        if self.bayes_target == "rep_tokens" and rep_prior_mode != "zero":
            prior_mean = self._build_rep_clip_prior()
            self.model.representation_learner.apply_rep_prior(
                prior_mean=prior_mean.to(self.device),
                init_mode=str(getattr(bayes_cfg, "REP_INIT_MODE", "prior_mean_noise")),
                init_std=float(getattr(bayes_cfg, "REP_INIT_STD", 0.01)),
                prior_std=float(
                    getattr(
                        bayes_cfg,
                        "REP_PRIOR_STD",
                        getattr(bayes_cfg, "PRIOR_STD", 0.05),
                    )
                ),
            )

        enabled = freeze_all_but(
            self.model,
            self._resolve_trainable_substrings(),
        )
        print(f"[BayesMMRLMethod] trainable params: {enabled}")

        self.sample_loss = MMRLLoss(
            reg_weight=bayes_cfg.REG_WEIGHT,
            alpha=bayes_cfg.ALPHA,
        )
        self.loss = BayesMMRLLossAdapter()
        return self

    def get_precision(self) -> str:
        return self.cfg.BAYES_MMRL.PREC

    def select_train_logits(self, outputs):
        return outputs.aux_logits.get("fusion", outputs.logits)

    def select_eval_logits(self, outputs, eval_ctx):
        logits = outputs.logits
        logits_fusion = outputs.aux_logits.get("fusion")
        if logits_fusion is None:
            return logits

        protocol = eval_ctx.protocol
        dataset = eval_ctx.dataset_name
        sub_cls = eval_ctx.subsample_classes or "all"

        if protocol == "B2N":
            if sub_cls == "base":
                return logits_fusion
            return logits

        if protocol == "FS":
            return logits_fusion

        if protocol == "CD":
            if dataset == "ImageNet":
                return logits_fusion
            return logits

        return logits

    def _build_outputs_from_samples(self, label, img_ref, sample_outputs):
        per_sample_losses = []
        logits_list = []
        logits_rep_list = []
        logits_fusion_list = []
        image_features_list = []
        text_features_list = []

        for logits, logits_rep, logits_fusion, image_features, text_features in sample_outputs:
            text_features = text_features[: self.num_classes]

            loss_s = self.sample_loss(
                logits,
                logits_rep,
                image_features,
                text_features,
                img_ref,
                self.text_features_clip,
                label,
            )
            per_sample_losses.append(loss_s)

            logits_list.append(logits)
            logits_rep_list.append(logits_rep)
            logits_fusion_list.append(logits_fusion)
            image_features_list.append(image_features)
            text_features_list.append(text_features)

        data_term = torch.stack(per_sample_losses, dim=0).mean(dim=0)

        raw_kl = self.model.kl_terms()
        kl_rep_term = self.rep_kl_weight * raw_kl["rep_tokens"]
        kl_proj_rep_term = self.proj_rep_kl_weight * raw_kl["proj_rep"]
        kl_term = kl_rep_term + kl_proj_rep_term
        total_loss = data_term + kl_term

        logits_mean = torch.stack(logits_list, dim=0).mean(dim=0)
        logits_rep_mean = torch.stack(logits_rep_list, dim=0).mean(dim=0)
        logits_fusion_mean = torch.stack(logits_fusion_list, dim=0).mean(dim=0)
        image_features_mean = torch.stack(image_features_list, dim=0).mean(dim=0)
        text_features_mean = torch.stack(text_features_list, dim=0).mean(dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(
            dim=-1,
            keepdim=True,
        )

        extras = dict(self.model.posterior_stats())

        return MethodOutputs(
            logits=logits_mean,
            labels=label,
            aux_logits={
                "rep": logits_rep_mean,
                "fusion": logits_fusion_mean,
            },
            features={
                "img": image_features_mean,
                "text": text_features_mean,
                "img_ref": img_ref,
                "text_ref": self.text_features_clip,
            },
            losses={
                "data_term": data_term,
                "kl_rep_term": kl_rep_term,
                "kl_proj_rep_term": kl_proj_rep_term,
                "kl_term": kl_term,
                "total": total_loss,
            },
            extras=extras,
        )

    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = img_ref / img_ref.norm(dim=-1, keepdim=True)

        sample_outputs = self.model.forward_train_samples(image, self.n_mc_train)
        return self._build_outputs_from_samples(label, img_ref, sample_outputs)

    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        logits, logits_rep, logits_fusion, image_features, text_features = (
            self.model.forward_eval(
                image,
                num_samples=self.n_mc_test,
                use_posterior_mean=self.eval_use_posterior_mean,
            )
        )
        text_features = text_features[: self.num_classes]

        extras = dict(self.model.posterior_stats())

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": image_features,
                "text": text_features,
            },
            extras=extras,
        )