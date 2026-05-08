from __future__ import annotations

import torch
import torch.nn.functional as F

from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.mmrl_family.base import BaseMMRLFamilyMethod

from .loss import BayesTextMMRLLossAdapter
from .modules import BayesTextMMRLModel


@METHOD_REGISTRY.register("BayesTextMMRL")
class BayesTextMMRLMethod(BaseMMRLFamilyMethod):
    method_name = "BayesTextMMRL"
    cfg_section_name = "BAYES_TEXT_MMRL"
    clip_loader_name = "MMRL"
    model_cls = BayesTextMMRLModel
    loss_adapter_cls = BayesTextMMRLLossAdapter

    trainable_substrings = (
        "representation_learner",
        "image_encoder.proj_rep",
        "text_posterior",
    )

    def build(self):
        method_cfg = getattr(self.cfg, self.cfg_section_name)

        self.kl_warmup_epochs = int(getattr(method_cfg, "KL_WARMUP_EPOCHS", 0))
        self.kl_beta = 1.0
        self.kl_normalizer = 1.0

        self.n_mc_train = max(1, int(method_cfg.N_MC_TRAIN))
        self.n_mc_test = max(1, int(method_cfg.N_MC_TEST))
        self.text_kl_weight = float(method_cfg.TEXT_KL_WEIGHT)
        self.eval_use_posterior_mean = bool(method_cfg.EVAL_USE_POSTERIOR_MEAN)

        return super().build()

    def build_loss(self, method_cfg):
        return BayesTextMMRLLossAdapter()


    def set_kl_normalizer(self, normalizer):
        try:
            normalizer = float(normalizer)
        except (TypeError, ValueError):
            normalizer = 1.0
        self.kl_normalizer = max(1.0, normalizer)

    def set_kl_beta(self, beta):
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            beta = 1.0
        self.kl_beta = min(1.0, max(0.0, beta))

    def _aggregation(self) -> str:
        aggregation = str(
            getattr(
                getattr(self.cfg, self.cfg_section_name),
                "EVAL_AGGREGATION",
                "prob_mean",
            )
        )
        if aggregation not in {"prob_mean", "logit_mean"}:
            raise ValueError(
                f"{self.cfg_section_name}.EVAL_AGGREGATION must be "
                f"'prob_mean' or 'logit_mean', got {aggregation}"
            )
        return aggregation

    @staticmethod
    def _expected_ce(logits_stack: torch.Tensor, label: torch.Tensor):
        return torch.stack(
            [
                F.cross_entropy(logits_stack[s], label)
                for s in range(logits_stack.shape[0])
            ],
            dim=0,
        ).mean()


    def _build_train_outputs(self, label, img_ref, out):
        method_cfg = getattr(self.cfg, self.cfg_section_name)

        text_mean = out["text_mean"][: self.num_classes]
        text_ref = self.text_features_clip[: self.num_classes].to(text_mean.device)

        logits_stack = out["logits_stack"][..., : self.num_classes]
        logits_rep_stack = out["logits_rep_stack"][..., : self.num_classes]

        loss_main = self._expected_ce(logits_stack, label)
        loss_rep = self._expected_ce(logits_rep_stack, label)

        # Same as MMRL image-side cosine regularization:
        # keep learned image feature close to frozen CLIP image feature.
        loss_cos_img = 1.0 - F.cosine_similarity(
            out["image_features"],
            img_ref,
            dim=1,
        ).mean()

        # Restored MMRL text-side cosine regularization:
        # keep learned BayesText/MMRL text mean close to frozen CLIP text feature.
        #
        # text_mean: [C, d], current MMRL text-side output used as posterior mean
        # text_ref:  [C, d], zero-shot CLIP text feature prior/reference
        loss_cos_text = 1.0 - F.cosine_similarity(
            text_mean,
            text_ref,
            dim=1,
        ).mean()

        raw_kl_text = self.model.text_posterior.kl_divergence(
            mean=text_mean,
            prior_mean=text_ref,
        )

        # Normalize by C*d, then by num batches if executor provides it.
        c, d = text_mean.shape
        cd_normalizer = float(max(1, c * d))
        batch_normalizer = float(getattr(self, "kl_normalizer", 1.0))
        kl_beta = float(getattr(self, "kl_beta", 1.0))

        kl_text_term = (
            kl_beta
            * self.text_kl_weight
            * raw_kl_text
            / cd_normalizer
            / batch_normalizer
        )

        alpha = float(method_cfg.ALPHA)
        reg_weight = float(method_cfg.REG_WEIGHT)

        # MMRL-style data term:
        # CE(main) + CE(rep) + image cosine reg + text cosine reg.
        #
        # KL is kept as the Bayesian posterior regularizer.
        data_term = (
            alpha * loss_main
            + (1.0 - alpha) * loss_rep
            + reg_weight * loss_cos_img
            + reg_weight * loss_cos_text
        )

        total = data_term + kl_text_term

        logits = out["logits"][:, : self.num_classes]
        logits_rep = out["logits_rep"][:, : self.num_classes]
        logits_fusion = out["logits_fusion"][:, : self.num_classes]

        losses = {
            "loss_main": loss_main,
            "loss_rep": loss_rep,
            "loss_cos_img": loss_cos_img,
            "loss_cos_text": loss_cos_text,
            "raw_kl_text": raw_kl_text,
            "kl_text_term": kl_text_term,
            "kl_beta": text_mean.detach().new_tensor(kl_beta),
            "kl_normalizer": text_mean.detach().new_tensor(batch_normalizer),
            "data_term": data_term,
            "total": total,
        }

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": out["image_features"],
                "text": text_mean,
                "img_ref": img_ref,
                "text_ref": text_ref,
            },
            losses=losses,
        )


    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = F.normalize(img_ref, dim=-1)

        out = self.model.forward_bayes_text(
            image,
            num_samples=self.n_mc_train,
            use_posterior_mean=False,
            aggregation=self._aggregation(),
        )

        return self._build_train_outputs(label, img_ref, out)

    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        out = self.model.forward_bayes_text(
            image,
            num_samples=self.n_mc_test,
            use_posterior_mean=self.eval_use_posterior_mean,
            aggregation=self._aggregation(),
        )

        text_mean = out["text_mean"][: self.num_classes]
        logits = out["logits"][:, : self.num_classes]
        logits_rep = out["logits_rep"][:, : self.num_classes]
        logits_fusion = out["logits_fusion"][:, : self.num_classes]

        sigma = self.model.text_posterior.posterior_sigma().detach()

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": out["image_features"],
                "text": text_mean,
            },
            extras={
                "text_sigma_mean": sigma.mean(),
                "text_sigma_max": sigma.max(),
                "text_sigma_min": sigma.min(),
            },
        )