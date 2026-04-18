from __future__ import annotations

import torch

from backbones.clip_loader import load_mmrl_clip_to_cpu
from backbones.freeze import freeze_all_but
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
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

    def build(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        clip_model = load_mmrl_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_mmrl_clip_to_cpu(cfg, "CLIP")

        if cfg.BAYES_MMRL.PREC in {"fp32", "amp"}:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype
        self.n_mc_train = max(1, int(cfg.BAYES_MMRL.N_MC_TRAIN))
        self.n_mc_test = max(1, int(cfg.BAYES_MMRL.N_MC_TEST))
        self.eval_use_posterior_mean = bool(cfg.BAYES_MMRL.EVAL_USE_POSTERIOR_MEAN)
        self.kl_weight = float(cfg.BAYES_MMRL.KL_WEIGHT)

        self.text_encoder_clip = CLIPTextEncoderPlain(clip_model_zero_shot).to(
            self.device
        )

        with torch.no_grad():
            text_features_clip = build_zero_shot_text_features(
                cfg, classnames, clip_model_zero_shot, self.text_encoder_clip
            )
            self.text_features_clip = (
                text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
            ).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)
        self.model = BayesianCustomMMRLModel(cfg, classnames, clip_model).to(
            self.device
        )

        enabled = freeze_all_but(
            self.model,
            ["representation_learner", "image_encoder.proj_rep"],
        )
        print(f"[BayesMMRLMethod] trainable params: {enabled}")

        self.sample_loss = MMRLLoss(
            reg_weight=cfg.BAYES_MMRL.REG_WEIGHT,
            alpha=cfg.BAYES_MMRL.ALPHA,
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
        kl_term = self.kl_weight * self.model.representation_learner.kl_divergence()
        total_loss = data_term + kl_term

        logits_mean = torch.stack(logits_list, dim=0).mean(dim=0)
        logits_rep_mean = torch.stack(logits_rep_list, dim=0).mean(dim=0)
        logits_fusion_mean = torch.stack(logits_fusion_list, dim=0).mean(dim=0)
        image_features_mean = torch.stack(image_features_list, dim=0).mean(dim=0)
        text_features_mean = torch.stack(text_features_list, dim=0).mean(dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(
            dim=-1, keepdim=True
        )

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
                "kl_term": kl_term,
                "total": total_loss,
            },
            extras={
                "posterior_sigma": self.model.representation_learner.posterior_sigma().detach(),
            },
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
            extras={
                "posterior_sigma": self.model.representation_learner.posterior_sigma().detach(),
            },
        )