from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.det_bayesrt_mmrl.modules import (
    DetBayesRTMMRLModel,
    _factorized_project_moments,
    _normalize_moments,
    _norm_directional_var,
    _rep_logit_var,
)


class FusedDetBayesRTMMRLModel(DetBayesRTMMRLModel):
    """
    DetBayesRTMMRL model with an additional fused-logit deterministic moment
    forward for the training objective:

        z_f = tau * [alpha * image_main + (1 - alpha) * image_rep]^T text

    Text posterior q(P_t) is shared by main and rep branches, so the fused
    text-side variance must be computed with the fused visual query rather than
    by linearly combining branch variances.
    """

    def _text_moments(self, compound_rep_tokens_text):
        eot_hidden = self.text_encoder.forward_hidden(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )

        if self.text_posterior is None:
            text_pre_mean = (
                eot_hidden.float()
                @ self.text_projection.to(eot_hidden.device).float()
            )
            text_pre_var = torch.zeros_like(text_pre_mean)
        else:
            text_pre_mean, text_pre_var = _factorized_project_moments(
                self.text_posterior,
                eot_hidden.float(),
            )

        text_mean, text_rho, text_pre_var = _normalize_moments(
            text_pre_mean,
            text_pre_var,
            eps=self.det_moment_eps,
        )

        return text_mean, text_rho, text_pre_var

    @staticmethod
    def _rep_visual_directional_var(
        rep_mean: torch.Tensor,
        rep_pre_var: torch.Tensor,
        rep_rho: torch.Tensor,
        text_mean: torch.Tensor,
        alpha_rep: float,
        eps: float = 1.0e-6,
    ) -> torch.Tensor:
        """
        Variance contribution from the random representation-side visual
        projection in the fused logit.

        The fused visual direction is alpha_main * main + alpha_rep * rep.
        Only rep is random on the visual side, so the derivative wrt the rep
        normalized feature is scaled by alpha_rep.

        Returns:
            [B, C] variance of cosine similarity, before multiplying by tau^2.
        """
        r = rep_mean.float()
        t = text_mean.float()
        rv = rep_pre_var.float().clamp_min(0.0)
        rrho = rep_rho.float().clamp_min(float(eps))

        dot_rt = r @ t.t()  # [B, C]
        direction_r = (
            t[None, :, :]
            - dot_rt[:, :, None] * r[:, None, :]
        ) / rrho[:, None, None]

        direction_r = float(alpha_rep) * direction_r
        return (direction_r.pow(2) * rv[:, None, :]).sum(dim=-1).to(rep_mean.dtype)

    def _fused_text_directional_var(
        self,
        fused_visual_query: torch.Tensor,
        text_mean: torch.Tensor,
        text_pre_var: torch.Tensor,
        text_rho: torch.Tensor,
    ) -> torch.Tensor:
        """
        Variance contribution from the shared random text projection q(P_t).

        Since main and rep share the same random text feature, the correct text
        query is the fused visual direction:

            b = alpha_main * image_main + alpha_rep * rep_mean

        Returns:
            [B, C] variance of cosine similarity, before multiplying by tau^2.
        """
        return _norm_directional_var(
            query=fused_visual_query.float(),
            y_mean=text_mean.float(),
            pre_var=text_pre_var.float(),
            rho=text_rho.float(),
            eps=self.det_moment_eps,
        )

    def forward_joint_fused_moments(
        self,
        image: torch.Tensor,
        alpha_main: float,
    ):
        """
        Sampling-free deterministic moment forward for a fused-logit objective.

        Args:
            image: input images.
            alpha_main: weight of the main branch. The rep weight is
                1 - alpha_main, matching BAYESRT_MMRL.ALPHA convention in the
                existing BayesRT/DetBayesRT code.

        Returns:
            A dict containing branch moments plus fused moments:
                mu_fusion, var_fusion_shared.
        """
        alpha_main = float(alpha_main)
        alpha_rep = 1.0 - alpha_main

        compound_rep_tokens_text, compound_rep_tokens_visual = (
            self.representation_learner()
        )

        # Text posterior moments are shared by main and rep branches.
        text_mean, text_rho, text_pre_var = self._text_moments(
            compound_rep_tokens_text
        )

        # Visual hidden states.
        cls_hidden, rep_hidden = self.image_encoder.encode_hidden(
            image.type(self.dtype),
            compound_rep_tokens_visual,
        )

        # Main branch image feature is a point-estimate path on the visual side.
        image_main = cls_hidden @ self.image_encoder.visual.proj
        image_main = F.normalize(image_main, dim=-1)

        # Rep branch visual projection moments: rep_hidden @ P_r.
        if self.image_encoder.bayes_proj_rep is None:
            rep_pre_mean = (
                rep_hidden.float()
                @ self.image_encoder.visual.proj_rep.to(rep_hidden.device).float()
            )
            rep_pre_var = torch.zeros_like(rep_pre_mean)
        else:
            rep_pre_mean, rep_pre_var = _factorized_project_moments(
                self.image_encoder.bayes_proj_rep,
                rep_hidden.float(),
            )

        rep_mean, rep_rho, rep_pre_var = _normalize_moments(
            rep_pre_mean,
            rep_pre_var,
            eps=self.det_moment_eps,
        )

        tau = float(self.logit_scale)

        image_main_f = image_main.float()
        rep_mean_f = rep_mean.float()
        text_mean_f = text_mean.float()

        # Branch posterior-mean logits for diagnostics and comparability.
        mu_main = tau * (image_main_f @ text_mean_f.t())
        mu_rep = tau * (rep_mean_f @ text_mean_f.t())

        # Branch variances for diagnostics. These match the existing
        # DetBayesRT branch-wise moment definitions.
        var_main = tau * tau * _norm_directional_var(
            query=image_main_f,
            y_mean=text_mean_f,
            pre_var=text_pre_var.float(),
            rho=text_rho.float(),
            eps=self.det_moment_eps,
        )

        var_rep = _rep_logit_var(
            r_mean=rep_mean_f,
            r_pre_var=rep_pre_var.float(),
            r_rho=rep_rho.float(),
            t_mean=text_mean_f,
            t_pre_var=text_pre_var.float(),
            t_rho=text_rho.float(),
            tau=tau,
            eps=self.det_moment_eps,
        )

        # Fused posterior-mean visual query. This is intentionally not
        # renormalized, because it corresponds to alpha * logit_main +
        # (1-alpha) * logit_rep.
        fused_visual = alpha_main * image_main_f + alpha_rep * rep_mean_f
        mu_fusion = tau * (fused_visual @ text_mean_f.t())

        # Fused variance under shared q(P_t).
        var_r_cos = self._rep_visual_directional_var(
            rep_mean=rep_mean_f,
            rep_pre_var=rep_pre_var.float(),
            rep_rho=rep_rho.float(),
            text_mean=text_mean_f,
            alpha_rep=alpha_rep,
            eps=self.det_moment_eps,
        )

        var_t_cos = self._fused_text_directional_var(
            fused_visual_query=fused_visual,
            text_mean=text_mean_f,
            text_pre_var=text_pre_var.float(),
            text_rho=text_rho.float(),
        )

        var_shared_r = tau * tau * var_r_cos
        var_shared_t = tau * tau * var_t_cos
        var_fusion_shared = var_shared_r + var_shared_t

        return {
            "mu_main": mu_main.to(image_main.dtype),
            "var_main": var_main.to(image_main.dtype),
            "mu_rep": mu_rep.to(image_main.dtype),
            "var_rep": var_rep.to(image_main.dtype),
            "mu_fusion": mu_fusion.to(image_main.dtype),
            "var_fusion_shared": var_fusion_shared.to(image_main.dtype),
            "var_shared_r": var_shared_r.to(image_main.dtype),
            "var_shared_t": var_shared_t.to(image_main.dtype),
            "image_features_main": image_main,
            "image_features_rep": rep_mean.to(image_main.dtype),
            "text_features": text_mean.to(image_main.dtype),
            "text_pre_var": text_pre_var.detach(),
            "rep_pre_var": rep_pre_var.detach(),
        }
