from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from methods.bayesrt_mmrl.modules import BayesRTMMRLModel, FactorizedGaussianMatrix


def _factorized_project_moments(posterior, x: torch.Tensor):
    """
    Deterministic moment propagation through an existing FactorizedGaussianMatrix.

    Args:
        posterior: methods.bayesrt_mmrl.modules.FactorizedGaussianMatrix
        x: [..., input_dim]

    Returns:
        mean: [..., output_dim]
        var:  [..., output_dim], diagonal variance of x @ W
    """
    x_f = x.float()
    mean_w = posterior.posterior_mean.float().to(x.device)
    sigma2_w = posterior.expanded_posterior_sigma().pow(2).float().to(x.device)

    mean = x_f @ mean_w
    var = x_f.pow(2) @ sigma2_w

    return mean.to(x.dtype), var.to(x.dtype)


def _normalize_moments(
    pre_mean: torch.Tensor,
    pre_var: torch.Tensor,
    eps: float = 1.0e-6,
):
    """
    First-order delta approximation for y = normalize(a).

    Args:
        pre_mean: [N, D]
        pre_var:  [N, D], diagonal variance of a

    Returns:
        y_mean: [N, D], normalize(pre_mean)
        rho:    [N], norm(pre_mean)
        pre_var: unchanged, used for directional variance
    """
    pre_mean_f = pre_mean.float()
    pre_var_f = pre_var.float().clamp_min(0.0)

    rho = pre_mean_f.norm(dim=-1).clamp_min(float(eps))
    y_mean = pre_mean_f / rho.unsqueeze(-1)

    return y_mean.to(pre_mean.dtype), rho.to(pre_mean.dtype), pre_var_f.to(pre_mean.dtype)


def _norm_directional_var(
    query: torch.Tensor,
    y_mean: torch.Tensor,
    pre_var: torch.Tensor,
    rho: torch.Tensor,
    eps: float = 1.0e-6,
):
    """
    Compute q^T Cov(normalize(a)) q without materializing Cov.

    Args:
        query:   [B, D]
        y_mean:  [C, D]
        pre_var: [C, D]
        rho:     [C]

    Returns:
        directional variance: [B, C]
    """
    q = query.float()
    y = y_mean.float()
    v = pre_var.float().clamp_min(0.0)
    r = rho.float().clamp_min(float(eps))

    dot = q @ y.t()  # [B, C]

    direction = (
        q[:, None, :]
        - dot[:, :, None] * y[None, :, :]
    ) / r[None, :, None]

    out = (direction.pow(2) * v[None, :, :]).sum(dim=-1)
    return out.to(query.dtype)


def _rep_logit_var(
    r_mean: torch.Tensor,
    r_pre_var: torch.Tensor,
    r_rho: torch.Tensor,
    t_mean: torch.Tensor,
    t_pre_var: torch.Tensor,
    t_rho: torch.Tensor,
    tau: float = 100.0,
    eps: float = 1.0e-6,
):
    """
    First-order logit variance for:
        z = tau * normalize(g P_r)^T normalize(h P_t)

    Args:
        r_mean:    [B, D]
        r_pre_var: [B, D]
        r_rho:     [B]
        t_mean:    [C, D]
        t_pre_var: [C, D]
        t_rho:     [C]

    Returns:
        var: [B, C]
    """
    r = r_mean.float()
    t = t_mean.float()
    rv = r_pre_var.float().clamp_min(0.0)
    tv = t_pre_var.float().clamp_min(0.0)

    rrho = r_rho.float().clamp_min(float(eps))
    trho = t_rho.float().clamp_min(float(eps))

    dot = r @ t.t()  # [B, C]

    # Random r side, query is t_c.
    dir_r = (
        t[None, :, :]
        - dot[:, :, None] * r[:, None, :]
    ) / rrho[:, None, None]
    term_r = (dir_r.pow(2) * rv[:, None, :]).sum(dim=-1)

    # Random t side, query is r_n.
    dir_t = (
        r[:, None, :]
        - dot[:, :, None] * t[None, :, :]
    ) / trho[None, :, None]
    term_t = (dir_t.pow(2) * tv[None, :, :]).sum(dim=-1)

    var = float(tau) * float(tau) * (term_r + term_t)
    return var.to(r_mean.dtype)



def _clone_method_cfg_with_r_prior_mode(method_cfg, r_prior_mode: str):
    """
    BayesRTMMRLModel only accepts {'zero', 'self_proj_rep'} in the base code.
    DetBayesRTMMRL adds random R initialization without modifying the original
    BayesRTMMRL files. For random modes, we call the base constructor with a
    safe temporary mode, then replace image_encoder.bayes_proj_rep afterward.
    """
    cfg_for_super = method_cfg.clone() if hasattr(method_cfg, "clone") else copy.deepcopy(method_cfg)

    was_frozen = False
    if hasattr(cfg_for_super, "is_frozen"):
        try:
            was_frozen = bool(cfg_for_super.is_frozen())
        except TypeError:
            was_frozen = False

    if hasattr(cfg_for_super, "defrost"):
        cfg_for_super.defrost()

    cfg_for_super.R_PRIOR_MODE = str(r_prior_mode)

    if was_frozen and hasattr(cfg_for_super, "freeze"):
        cfg_for_super.freeze()

    return cfg_for_super


def _make_mmrl_random_proj_rep_mean(visual: torch.nn.Module) -> torch.Tensor:
    """
    Match MMRL ViT proj_rep initialization:
        scale = width ** -0.5
        proj_rep = scale * randn(width, output_dim)
    """
    shape_ref = visual.proj_rep.detach().float()
    width = int(shape_ref.shape[0])
    std = float(width ** -0.5)
    return torch.empty_like(shape_ref).normal_(mean=0.0, std=std)



class DetBayesRTMMRLModel(BayesRTMMRLModel):
    """
    BayesRTMMRL model with an additional sampling-free deterministic moment forward.

    This class preserves the original BayesRTMMRL structure:
        q(P_t), q(P_r), normalized cosine logits, main/rep branches.

    Only the training likelihood estimator is changed by the method class:
        MC expected CE -> deterministic Jensen CE based on logit mean/variance.

    Eval can still call the inherited forward_joint() and use MC posterior predictive.
    """

    def __init__(self, cfg, method_cfg, classnames, clip_model):
        requested_r_prior_mode = str(getattr(method_cfg, "R_PRIOR_MODE", "zero"))

        random_r_modes = {
            "random",
            "random_mmrl",
            "mmrl_random",
            "random_init",
            "mmrl_random_init",
        }

        if requested_r_prior_mode in random_r_modes:
            # Avoid base BayesRTMMRLModel raising on the new mode.
            cfg_for_super = _clone_method_cfg_with_r_prior_mode(method_cfg, "zero")
        else:
            cfg_for_super = method_cfg

        super().__init__(cfg, cfg_for_super, classnames, clip_model)

        # Randomly initialize the variational mean q(P_r), while keeping
        # the prior mean of p(P_r) fixed at zero.
        if bool(getattr(method_cfg, "BAYES_R_ENABLED", False)) and requested_r_prior_mode in random_r_modes:
            random_mean = _make_mmrl_random_proj_rep_mean(clip_model.visual)

            self.image_encoder.bayes_proj_rep = FactorizedGaussianMatrix(
                mean=random_mean,
                sigma_mode=str(method_cfg.R_SIGMA_MODE),
                prior_std=float(method_cfg.R_PRIOR_STD),
                train_mean=True,
                min_sigma=1.0e-6,
            )

            self.image_encoder.bayes_proj_rep.prior_mean.zero_()

        self.det_r_prior_mode = requested_r_prior_mode
        self.det_moment_eps = float(getattr(method_cfg, "DET_MOMENT_EPS", 1.0e-6))
        self.logit_scale = float(getattr(method_cfg, "LOGIT_SCALE", 100.0))

    def forward_joint_moments(self, image: torch.Tensor):
        """
        Sampling-free deterministic moment forward.

        Returns:
            mu_main, var_main: [B, C]
            mu_rep,  var_rep:  [B, C]
        """
        compound_rep_tokens_text, compound_rep_tokens_visual = (
            self.representation_learner()
        )

        # Text hidden states: [C, text_width]
        eot_hidden = self.text_encoder.forward_hidden(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )

        # Text projection moments: a_c = h_c P_t
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

        # Visual hidden states.
        cls_hidden, rep_hidden = self.image_encoder.encode_hidden(
            image.type(self.dtype),
            compound_rep_tokens_visual,
        )

        # Main branch image feature is deterministic.
        image_main = cls_hidden @ self.image_encoder.visual.proj
        image_main = F.normalize(image_main, dim=-1)

        # Representation projection moments: b_n = g_n P_r
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
        text_mean_f = text_mean.float()
        rep_mean_f = rep_mean.float()

        # Main branch logit moments:
        # z_m = tau * image_main^T normalize(h P_t)
        mu_main = tau * (image_main_f @ text_mean_f.t())

        var_main = tau * tau * _norm_directional_var(
            query=image_main_f,
            y_mean=text_mean_f,
            pre_var=text_pre_var.float(),
            rho=text_rho.float(),
            eps=self.det_moment_eps,
        )

        # Rep branch logit moments:
        # z_r = tau * normalize(g P_r)^T normalize(h P_t)
        mu_rep = tau * (rep_mean_f @ text_mean_f.t())

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

        return {
            "mu_main": mu_main.to(image_main.dtype),
            "var_main": var_main.to(image_main.dtype),
            "mu_rep": mu_rep.to(image_main.dtype),
            "var_rep": var_rep.to(image_main.dtype),
            "image_features_main": image_main,
            "image_features_rep": rep_mean.to(image_main.dtype),
            "text_features": text_mean.to(image_main.dtype),
            "text_pre_var": text_pre_var.detach(),
            "rep_pre_var": rep_pre_var.detach(),
        }
