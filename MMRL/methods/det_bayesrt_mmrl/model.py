from __future__ import annotations

import torch
import torch.nn.functional as F

from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.bayesrt_mmrl.model import BayesRTMMRLMethod

from .modules import DetBayesRTMMRLModel
from .losses import det_jensen_ce


@METHOD_REGISTRY.register("DetBayesRTMMRL")
class DetBayesRTMMRLMethod(BayesRTMMRLMethod):
    """
    New method: deterministic-Jensen training variant of BayesRTMMRL.

    It does not modify the original BayesRTMMRL class.
    It reuses BayesRTMMRL's build/eval/fusion/KL logic, but replaces train-time
    MC expected CE with a sampling-free deterministic Jensen objective.

    Config section is intentionally kept as BAYESRT_MMRL so existing BayesRT
    method YAML keys remain compatible.
    """

    method_name = "DetBayesRTMMRL"
    cfg_section_name = "BAYESRT_MMRL"
    clip_loader_name = "MMRL"
    model_cls = DetBayesRTMMRLModel

    def build(self):
        super().build()
        method_cfg = self.cfg.BAYESRT_MMRL

        # These are optional. They are read with defaults so core/config.py does
        # not need to be changed to run this new method.
        self.det_var_scale = float(getattr(method_cfg, "DET_VAR_SCALE", 1.0))
        self.det_var_clamp = float(getattr(method_cfg, "DET_VAR_CLAMP", 0.0))

        return self


    def _build_train_outputs_det(self, label, img_ref, out):
        cfg = self.cfg.BAYESRT_MMRL

        mu_main = out["mu_main"][..., : self.num_classes]
        var_main = out["var_main"][..., : self.num_classes]
        mu_rep = out["mu_rep"][..., : self.num_classes]
        var_rep = out["var_rep"][..., : self.num_classes]

        loss_main = det_jensen_ce(
            mu_main,
            var_main,
            label,
            var_scale=self.det_var_scale,
            var_clamp=self.det_var_clamp,
        )

        loss_rep = det_jensen_ce(
            mu_rep,
            var_rep,
            label,
            var_scale=self.det_var_scale,
            var_clamp=self.det_var_clamp,
        )

        text_features = out["text_features"][: self.num_classes]
        text_ref = self.text_features_clip[: self.num_classes].to(text_features.device)

        loss_cos_img = 1.0 - F.cosine_similarity(
            out["image_features_main"],
            img_ref,
            dim=1,
        ).mean()

        # Same policy as original BayesRTMMRL: regularize mean text path only.
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

        # Train-time reporting logits use posterior mean logits.
        logits_main = mu_main
        logits_rep = mu_rep
        logits_fusion = alpha * logits_main + (1.0 - alpha) * logits_rep

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
            "det_var_main_mean": var_main.detach().mean(),
            "det_var_main_max": var_main.detach().max(),
            "det_var_rep_mean": var_rep.detach().mean(),
            "det_var_rep_max": var_rep.detach().max(),
        }

        extras = {
            "det_jensen_train": data_term.detach().new_tensor(1.0),
            "det_var_main_mean": var_main.detach().mean(),
            "det_var_rep_mean": var_rep.detach().mean(),
        }

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

        out = self.model.forward_joint_moments(image=image)
        return self._build_train_outputs_det(label, img_ref, out)
