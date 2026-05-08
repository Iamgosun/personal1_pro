from __future__ import annotations

import torch
import torch.nn.functional as F

from backbones.clip_loader import load_mmrl_clip_to_cpu
from backbones.freeze import freeze_all_but
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.base import BaseMethod
from methods.mmrl_family.modules import (
    CLIPTextEncoderPlain,
    build_zero_shot_text_features,
)

from .modules import BayesRTMMRLModel

class OutputTotalLoss(torch.nn.Module):
    def forward(self, outputs):
        return outputs.losses["total"]
    

@METHOD_REGISTRY.register("BayesRTMMRL")
class BayesRTMMRLMethod(BaseMethod):
    method_name = "BayesRTMMRL"
    cfg_section_name = "BAYESRT_MMRL"
    clip_loader_name = "MMRL"
    model_cls = BayesRTMMRLModel

    def build(self):
        cfg = self.cfg
        method_cfg = cfg.BAYESRT_MMRL
        classnames = self.dm.dataset.classnames

        self.method_cfg = method_cfg
        self.num_classes = len(classnames)

        self.n_mc_train = max(1, int(method_cfg.N_MC_TRAIN))
        self.n_mc_test = max(1, int(method_cfg.N_MC_TEST))
        self.eval_use_posterior_mean = bool(method_cfg.EVAL_USE_POSTERIOR_MEAN)

        self.r_kl_weight = float(method_cfg.R_KL_WEIGHT)
        self.t_kl_weight = float(method_cfg.T_KL_WEIGHT)

        self.kl_beta = 1.0
        self.kl_normalizer = 1.0
        self.kl_warmup_epochs = int(method_cfg.KL_WARMUP_EPOCHS)

        clip_model = load_mmrl_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_mmrl_clip_to_cpu(cfg, "CLIP")

        if method_cfg.PREC in {"fp32", "amp"}:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype

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
            self.text_features_clip = F.normalize(
                text_features_clip,
                dim=-1,
            ).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)

        self.model = self.model_cls(
            cfg,
            method_cfg,
            classnames,
            clip_model,
        ).to(self.device)

        trainable_substrings = [
            "representation_learner",
        ]

        if bool(method_cfg.BAYES_R_ENABLED):
            trainable_substrings.append("image_encoder.bayes_proj_rep")
        else:
            trainable_substrings.append("image_encoder.visual.proj_rep")

        if bool(method_cfg.BAYES_T_ENABLED):
            trainable_substrings.append("text_posterior")

        enabled = freeze_all_but(self.model, trainable_substrings)
        print(f"[BayesRTMMRL] trainable params: {enabled}")

        self.loss = OutputTotalLoss()

        return self
    


    def get_precision(self) -> str:
        return self.cfg.BAYESRT_MMRL.PREC

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

    @staticmethod
    def _probs_bma(logits_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_stack: [S, B, C]

        Returns:
            p_bar: [B, C]
        """
        return torch.softmax(logits_stack.float(), dim=-1).mean(dim=0)

    @staticmethod
    def _log_probs_bma(logits_stack: torch.Tensor) -> torch.Tensor:
        probs = BayesRTMMRLMethod._probs_bma(logits_stack)
        return torch.log(probs.clamp_min(1.0e-12)).to(logits_stack.dtype)

    @staticmethod
    def _logits_mean(logits_stack: torch.Tensor) -> torch.Tensor:
        return logits_stack.mean(dim=0)

    def _branch_logits_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ):
        """
        Default single-branch output for eval/logging/novel path.

        For a Bayesian branch, the most principled default is posterior
        predictive probability, i.e. log mean softmax.
        """
        logits_main = self._log_probs_bma(logits_main_stack)
        logits_rep = self._log_probs_bma(logits_rep_stack)
        return logits_main, logits_rep

    @staticmethod
    def _mutual_information_from_probs_stack(
        probs_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            probs_stack: [S, B, C]

        Returns:
            normalized MI per sample: [B]
        """
        probs_stack = probs_stack.float().clamp_min(1.0e-12)
        mean_probs = probs_stack.mean(dim=0).clamp_min(1.0e-12)

        h_mean = -(mean_probs * mean_probs.log()).sum(dim=-1)
        h_each = -(probs_stack * probs_stack.log()).sum(dim=-1).mean(dim=0)

        mi = h_mean - h_each

        num_classes = probs_stack.shape[-1]
        if num_classes > 1:
            mi = mi / torch.log(
                probs_stack.new_tensor(float(num_classes), dtype=torch.float32)
            )

        return mi

    def _fusion_static_prob_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mixture-of-experts style fusion:

            p = alpha * pbar_c + (1 - alpha) * pbar_r

        This matches the intended MMRL/BayesRT probability-space fusion.
        """
        alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)

        p_main = self._probs_bma(logits_main_stack)
        p_rep = self._probs_bma(logits_rep_stack)

        p = alpha * p_main + (1.0 - alpha) * p_rep
        return torch.log(p.clamp_min(1.0e-12)).to(logits_main_stack.dtype)

    def _fusion_static_logit_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logit-space ablation:

            z = alpha * mean(z_c) + (1 - alpha) * mean(z_r)
        """
        alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)

        z_main = self._logits_mean(logits_main_stack)
        z_rep = self._logits_mean(logits_rep_stack)

        return alpha * z_main + (1.0 - alpha) * z_rep

    def _fusion_poe_logprob_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Product-of-experts style ablation.

        This reproduces the old behavior when aggregation='prob_mean':

            log p = alpha * log pbar_c + (1 - alpha) * log pbar_r

        Keep this only as an explicit ablation, not as the default.
        """
        alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)

        log_p_main = self._log_probs_bma(logits_main_stack)
        log_p_rep = self._log_probs_bma(logits_rep_stack)

        return alpha * log_p_main + (1.0 - alpha) * log_p_rep

    def _fusion_uncertainty_prob_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ):
        """
        MI precision-weighted probability fusion:

            p = (pi_c * pbar_c + pi_r * pbar_r) / (pi_c + pi_r)

        where:
            pi_c = alpha / (MI_c + eps)
            pi_r = (1 - alpha) / (MI_r + eps)
        """
        eps = 1.0e-6
        alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)

        p_main_stack = torch.softmax(logits_main_stack.float(), dim=-1)
        p_rep_stack = torch.softmax(logits_rep_stack.float(), dim=-1)

        p_main = p_main_stack.mean(dim=0)
        p_rep = p_rep_stack.mean(dim=0)

        u_main = self._mutual_information_from_probs_stack(p_main_stack)
        u_rep = self._mutual_information_from_probs_stack(p_rep_stack)

        pi_main = alpha / (u_main + eps)
        pi_rep = (1.0 - alpha) / (u_rep + eps)

        p = (
            pi_main.unsqueeze(-1) * p_main
            + pi_rep.unsqueeze(-1) * p_rep
        ) / (pi_main + pi_rep).unsqueeze(-1)

        logits = torch.log(p.clamp_min(1.0e-12)).to(logits_main_stack.dtype)

        extras = {
            "u_main_mi": u_main.detach(),
            "u_rep_mi": u_rep.detach(),
            "precision_main": pi_main.detach(),
            "precision_rep": pi_rep.detach(),
        }

        return logits, extras

    def _build_fusion_from_stacks(
        self,
        logits_main_stack: torch.Tensor,
        logits_rep_stack: torch.Tensor,
    ):
        variant = str(
            getattr(
                self.cfg.BAYESRT_MMRL,
                "EVAL_FUSION_VARIANT",
                "static_prob",
            )
        ).lower()

        if variant in {"static", "static_prob", "prob"}:
            logits = self._fusion_static_prob_from_stacks(
                logits_main_stack,
                logits_rep_stack,
            )
            return logits, {"eval_fusion_variant": "static_prob"}

        if variant in {"static_logit", "logit"}:
            logits = self._fusion_static_logit_from_stacks(
                logits_main_stack,
                logits_rep_stack,
            )
            return logits, {"eval_fusion_variant": "static_logit"}

        if variant in {"uncertainty", "uncertainty_prob", "mi"}:
            logits, extras = self._fusion_uncertainty_prob_from_stacks(
                logits_main_stack,
                logits_rep_stack,
            )
            extras["eval_fusion_variant"] = "uncertainty_prob"
            return logits, extras

        if variant in {"poe", "poe_logprob", "old_static"}:
            logits = self._fusion_poe_logprob_from_stacks(
                logits_main_stack,
                logits_rep_stack,
            )
            return logits, {"eval_fusion_variant": "poe_logprob"}

        raise ValueError(
            "BAYESRT_MMRL.EVAL_FUSION_VARIANT must be one of "
            "{'static_prob', 'static_logit', 'uncertainty_prob', 'poe_logprob'}, "
            f"got {variant}"
        )



    @staticmethod
    def _expected_ce(
        logits_stack: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                F.cross_entropy(logits_stack[s], label)
                for s in range(logits_stack.shape[0])
            ],
            dim=0,
        ).mean()

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1).clamp_min(1.0e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)

        num_classes = logits.shape[-1]
        if num_classes > 1:
            entropy = entropy / torch.log(
                logits.new_tensor(float(num_classes), dtype=torch.float32)
            )

        return entropy

    @staticmethod
    def _mutual_information_from_logits_stack(
        logits_stack: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_stack: [S, B, C]

        Returns:
            MI estimate per sample: [B]
        """
        probs = torch.softmax(logits_stack.float(), dim=-1).clamp_min(1.0e-12)
        mean_probs = probs.mean(dim=0).clamp_min(1.0e-12)

        h_mean = -(mean_probs * mean_probs.log()).sum(dim=-1)
        h_each = -(probs * probs.log()).sum(dim=-1).mean(dim=0)

        mi = h_mean - h_each

        num_classes = logits_stack.shape[-1]
        if num_classes > 1:
            mi = mi / torch.log(
                logits_stack.new_tensor(float(num_classes), dtype=torch.float32)
            )

        return mi




    def _build_train_outputs(self, label, img_ref, out):
        cfg = self.cfg.BAYESRT_MMRL

        logits_main_stack = out["logits_main_stack"][..., : self.num_classes]
        logits_rep_stack = out["logits_rep_stack"][..., : self.num_classes]

        logits_main, logits_rep = self._branch_logits_from_stacks(
            logits_main_stack,
            logits_rep_stack,
        )
        logits_fusion, fusion_extras = self._build_fusion_from_stacks(
            logits_main_stack,
            logits_rep_stack,
        )
        text_features = out["text_features"][: self.num_classes]
        text_ref = self.text_features_clip[: self.num_classes].to(text_features.device)

        loss_main = self._expected_ce(logits_main_stack, label)
        loss_rep = self._expected_ce(logits_rep_stack, label)

        loss_cos_img = 1.0 - F.cosine_similarity(
            out["image_features_main"],
            img_ref,
            dim=1,
        ).mean()

        # Critical: text regularization uses mean path only.
        # Do not regularize sampled text features, otherwise T covariance collapses.
        loss_cos_text = 1.0 - F.cosine_similarity(
            text_features,
            text_ref,
            dim=1,
        ).mean()

        alpha = float(cfg.ALPHA)
        reg_weight = float(cfg.REG_WEIGHT)

        data_term = (
            alpha * loss_main
            + (1.0 - alpha) * loss_rep
            + reg_weight * loss_cos_img
            + reg_weight * loss_cos_text
        )

        kl_terms = self.model.kl_terms()
        raw_kl_r = kl_terms["r_proj"]
        raw_kl_t = kl_terms["t_proj"]

        kl_beta = float(getattr(self, "kl_beta", 1.0))
        kl_normalizer = float(getattr(self, "kl_normalizer", 1.0))
        kl_normalizer = max(1.0, kl_normalizer)

        kl_r_term = (
            kl_beta
            * self.r_kl_weight
            * raw_kl_r
            / kl_normalizer
        )

        t_normalizer = 1.0
        if getattr(self.model, "text_posterior", None) is not None:
            t_normalizer = float(max(1, self.model.text_posterior.prior_mean.numel()))

        kl_t_term = (
            kl_beta
            * self.t_kl_weight
            * raw_kl_t
            / t_normalizer
            / kl_normalizer
        )

        total = data_term + kl_r_term + kl_t_term

        losses = {
            "loss_main": loss_main,
            "loss_rep": loss_rep,
            "loss_cos_img": loss_cos_img,
            "loss_cos_text": loss_cos_text,
            "data_term": data_term,
            "raw_kl_r": raw_kl_r,
            "raw_kl_t": raw_kl_t,
            "kl_r_term": kl_r_term,
            "kl_t_term": kl_t_term,
            "kl_beta": data_term.detach().new_tensor(kl_beta),
            "kl_normalizer": data_term.detach().new_tensor(kl_normalizer),
            "total": total,
        }

  
        extras = dict(fusion_extras)

        if getattr(self.model.image_encoder, "bayes_proj_rep", None) is not None:
            sigma_r = self.model.image_encoder.bayes_proj_rep.posterior_sigma().detach()
            extras.update(
                {
                    "r_sigma_mean": sigma_r.mean(),
                    "r_sigma_min": sigma_r.min(),
                    "r_sigma_max": sigma_r.max(),
                }
            )

        if getattr(self.model, "text_posterior", None) is not None:
            sigma_t = self.model.text_posterior.posterior_sigma().detach()
            extras.update(
                {
                    "t_sigma_mean": sigma_t.mean(),
                    "t_sigma_min": sigma_t.min(),
                    "t_sigma_max": sigma_t.max(),
                }
            )

        return MethodOutputs(
            logits=logits_main,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": out["image_features_main"],
                "text": text_features,
                "img_ref": img_ref,
                "text_ref": text_ref,
            },
            losses=losses,
            extras=extras,
        )

    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = F.normalize(img_ref, dim=-1)

        out = self.model.forward_joint(
            image=image,
            num_samples=self.n_mc_train,
            use_posterior_mean=False,
        )

        return self._build_train_outputs(label, img_ref, out)

    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        use_posterior_mean = bool(self.eval_use_posterior_mean)

        # Conservative novel/new-task path:
        # B2N novel uses main branch in select_eval_logits.
        # This additionally avoids text sampling if requested.
        protocol = str(eval_ctx.protocol).upper()
        dataset = str(eval_ctx.dataset_name)
        sub_cls = str(eval_ctx.subsample_classes or "all")

        if bool(self.cfg.BAYESRT_MMRL.NOVEL_TEXT_MEAN_ONLY):
            is_b2n_novel = protocol == "B2N" and sub_cls != "base"
            is_cd_target = protocol == "CD" and dataset != "ImageNet"

            if is_b2n_novel or is_cd_target:
                use_posterior_mean = True

        out = self.model.forward_joint(
            image=image,
            num_samples=self.n_mc_test,
            use_posterior_mean=use_posterior_mean,
        )

        logits_main_stack = out["logits_main_stack"][..., : self.num_classes]
        logits_rep_stack = out["logits_rep_stack"][..., : self.num_classes]

        logits_main, logits_rep = self._branch_logits_from_stacks(
            logits_main_stack,
            logits_rep_stack,
        )

        logits_fusion, fusion_extras = self._build_fusion_from_stacks(
            logits_main_stack,
            logits_rep_stack,
        )

        logits_fusion_static = self._fusion_static_prob_from_stacks(
            logits_main_stack,
            logits_rep_stack,
        )

        extras = dict(fusion_extras)

        if getattr(self.model.image_encoder, "bayes_proj_rep", None) is not None:
            sigma_r = self.model.image_encoder.bayes_proj_rep.posterior_sigma().detach()
            extras.update(
                {
                    "r_sigma_mean": sigma_r.mean(),
                    "r_sigma_min": sigma_r.min(),
                    "r_sigma_max": sigma_r.max(),
                }
            )

        if getattr(self.model, "text_posterior", None) is not None:
            sigma_t = self.model.text_posterior.posterior_sigma().detach()
            extras.update(
                {
                    "t_sigma_mean": sigma_t.mean(),
                    "t_sigma_min": sigma_t.min(),
                    "t_sigma_max": sigma_t.max(),
                }
            )

        return MethodOutputs(
            logits=logits_main,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
                "fusion_static": logits_fusion_static,
                "fusion_static_prob": logits_fusion_static,
            },
            features={
                "img": out["image_features_main"],
                "text": out["text_features"][: self.num_classes],
            },
            extras=extras,
        )

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

        # Same decoupled inference policy as MMRL:
        # base / source domain uses fusion; novel / target domain uses main branch.
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