from __future__ import annotations

import torch
import torch.nn.functional as F

from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.det_bayesrt_mmrl.losses import det_jensen_ce
from methods.det_bayesrt_mmrl.model import DetBayesRTMMRLMethod

from .modules import FusedDetBayesRTMMRLModel


@METHOD_REGISTRY.register("FusedDetBayesRTMMRL")
class FusedDetBayesRTMMRLMethod(DetBayesRTMMRLMethod):
    """
    Fused-logit deterministic Jensen variant of DetBayesRTMMRL.

    Difference from DetBayesRTMMRL:
      - DetBayesRTMMRL trains branch-wise:
            alpha * DJ(main) + (1 - alpha) * DJ(rep)
      - This method trains the fused classifier:
            DJ(alpha * main + (1 - alpha) * rep)

    The text posterior q(P_t) is shared by main and rep, so fused variance is
    recomputed with the fused visual query instead of being assembled from
    branch variances.
    """

    method_name = "FusedDetBayesRTMMRL"
    cfg_section_name = "BAYESRT_MMRL"
    clip_loader_name = "MMRL"
    model_cls = FusedDetBayesRTMMRLModel

    def _build_train_outputs_fused(self, label, img_ref, out):
        cfg = self.cfg.BAYESRT_MMRL

        mu_main = out["mu_main"][..., : self.num_classes]
        var_main = out["var_main"][..., : self.num_classes]
        mu_rep = out["mu_rep"][..., : self.num_classes]
        var_rep = out["var_rep"][..., : self.num_classes]
        mu_fusion = out["mu_fusion"][..., : self.num_classes]
        var_fusion = out["var_fusion_shared"][..., : self.num_classes]
        var_shared_r = out["var_shared_r"][..., : self.num_classes]
        var_shared_t = out["var_shared_t"][..., : self.num_classes]

        # Main/rep losses are diagnostics only. The optimized likelihood
        # surrogate is loss_fusion.
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

        loss_fusion = det_jensen_ce(
            mu_fusion,
            var_fusion,
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

        # Same policy as BayesRT/DetBayesRT: regularize mean text path only.
        loss_cos_text = 1.0 - F.cosine_similarity(
            text_features,
            text_ref,
            dim=1,
        ).mean()

        reg_weight = float(cfg.REG_WEIGHT)
        data_term = (
            loss_fusion
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
            "loss_fusion": loss_fusion,
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
            "det_var_fusion_mean": var_fusion.detach().mean(),
            "det_var_fusion_max": var_fusion.detach().max(),
            "det_var_shared_r_mean": var_shared_r.detach().mean(),
            "det_var_shared_t_mean": var_shared_t.detach().mean(),
        }

        extras = {
            "fused_det_jensen_train": data_term.detach().new_tensor(1.0),
            "det_var_fusion_mean": var_fusion.detach().mean(),
            "det_var_shared_r_mean": var_shared_r.detach().mean(),
            "det_var_shared_t_mean": var_shared_t.detach().mean(),
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
            logits=mu_main,
            labels=label,
            aux_logits={
                "rep": mu_rep,
                "fusion": mu_fusion,
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

        out = self.model.forward_joint_fused_moments(
            image=image,
            alpha_main=float(self.cfg.BAYESRT_MMRL.ALPHA),
        )
        return self._build_train_outputs_fused(label, img_ref, out)
