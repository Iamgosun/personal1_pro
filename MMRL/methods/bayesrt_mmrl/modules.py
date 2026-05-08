from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.mmrl_family.modules import MMRLFamilyRepresentationLearner


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x))

def _resolve_sigma_shape(shape: tuple[int, int], mode: str):
    rows, cols = shape
    mode = str(mode)

    if mode == "global":
        return ()

    # W: [input_dim, output_dim]
    # hidden @ W -> feature
    if mode == "input":
        return (rows, 1)

    if mode == "output":
        return (1, cols)

    if mode == "diagonal":
        return shape

    raise ValueError(
        f"Unsupported sigma mode: {mode}. "
        "Expected one of {'global', 'input', 'output', 'diagonal'}."
    )


class FactorizedGaussianMatrix(nn.Module):
    """
    Factorized Gaussian matrix posterior.

    For R projection:
        q(W_r) = N(mu_r, diag(sigma_r^2))
        p(W_r) = N(mu_0, diag(sigma_0^2))
        train_mean=True

    For T projection:
        q(P_t) = N(P_t^0, diag(sigma_t^2))
        p(P_t) = N(P_t^0, diag(sigma_0^2))
        train_mean=False

    Shape convention follows CLIP:
        W: [input_dim, output_dim]
        output feature = hidden @ W
    """

    def __init__(
        self,
        mean: torch.Tensor,
        sigma_mode: str,
        prior_std: float,
        train_mean: bool,
        min_sigma: float = 1.0e-6,
    ):
        super().__init__()

        mean = mean.detach().float()
        if mean.dim() != 2:
            raise ValueError(f"Expected a 2D matrix, got shape={tuple(mean.shape)}")

        self.shape = tuple(mean.shape)
        self.sigma_mode = str(sigma_mode)
        self.min_sigma = float(min_sigma)
        self.train_mean = bool(train_mean)

        if float(prior_std) <= self.min_sigma:
            raise ValueError(
                f"prior_std must be greater than min_sigma, "
                f"got prior_std={prior_std}, min_sigma={self.min_sigma}"
            )

        if self.train_mean:
            self.posterior_mean = nn.Parameter(mean.clone())
            self.register_buffer("prior_mean", mean.clone())
        else:
            self.register_buffer("posterior_mean", mean.clone())
            self.register_buffer("prior_mean", mean.clone())

        sigma_shape = _resolve_sigma_shape(self.shape, self.sigma_mode)

        self.register_buffer(
            "prior_std",
            torch.full(sigma_shape, float(prior_std), dtype=torch.float32),
        )

        rho_init = _softplus_inverse(
            (self.prior_std.float() - self.min_sigma).clamp_min(1.0e-12)
        )
        self.posterior_rho = nn.Parameter(rho_init.clone())

    def posterior_sigma(self) -> torch.Tensor:
        return F.softplus(self.posterior_rho.float()) + self.min_sigma

    def expanded_posterior_sigma(self) -> torch.Tensor:
        return self.posterior_sigma().expand_as(self.prior_mean)

    def expanded_prior_sigma(self) -> torch.Tensor:
        return self.prior_std.float().expand_as(self.prior_mean)

    def sample_many(
        self,
        num_samples: int,
        use_mean: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            [S, input_dim, output_dim]
        """
        num_samples = max(1, int(num_samples))
        mean = self.posterior_mean.float()

        if use_mean:
            return mean.unsqueeze(0).expand(num_samples, *mean.shape)

        eps = torch.randn(
            num_samples,
            *mean.shape,
            device=mean.device,
            dtype=mean.dtype,
        )
        sigma = self.expanded_posterior_sigma().to(mean.dtype).unsqueeze(0)
        return mean.unsqueeze(0) + sigma * eps

    def kl_divergence(self) -> torch.Tensor:
        mu_q = self.posterior_mean.float()
        mu_p = self.prior_mean.float()

        sigma_q2 = self.expanded_posterior_sigma().pow(2)
        sigma_p2 = self.expanded_prior_sigma().pow(2)

        kl = 0.5 * (
            sigma_q2 / sigma_p2
            + (mu_q - mu_p).pow(2) / sigma_p2
            - 1.0
            - torch.log(sigma_q2 / sigma_p2)
        )
        return kl.sum()


class BayesRTTextEncoderHidden(nn.Module):
    """
    MMRL text encoder that returns the EOT hidden state before text projection.

    The original MMRL text encoder directly returns:
        eot_hidden @ text_projection

    BayesRT needs eot_hidden so that it can sample:
        P_t^(s) ~ q(P_t)
        w_k^(s) = normalize(eot_hidden_k @ P_t^(s))
    """

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

    def forward_hidden(
        self,
        prompts: torch.Tensor,
        tokenized_prompts: torch.Tensor,
        compound_rep_tokens_text,
    ) -> torch.Tensor:
        prompts = prompts.to(self.positional_embedding.device)
        tokenized_prompts = tokenized_prompts.to(prompts.device)

        n_rep_tokens = compound_rep_tokens_text[0].shape[0]

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        eot_index = tokenized_prompts.argmax(dim=-1)

        outputs = self.transformer([x, compound_rep_tokens_text, 0, eot_index])
        x = outputs[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        eot_hidden = x[
            torch.arange(x.shape[0], device=x.device),
            eot_index + n_rep_tokens,
        ]
        return eot_hidden


class BayesRTVisualWrapper(nn.Module):
    """
    Wrapper around MMRL ViT visual encoder.

    It copies the visual forward path up to the representation-token hidden state,
    then applies either:
        - deterministic visual.proj_rep, or
        - Bayesian q(W_r) over the R projection head.

    This avoids modifying clip/model_mmrl.py.
    """

    def __init__(self, visual: nn.Module, method_cfg):
        super().__init__()

        self.visual = visual
        self.bayes_r_enabled = bool(method_cfg.BAYES_R_ENABLED)

        if not hasattr(visual, "proj_rep"):
            raise AttributeError(
                "BayesRTMMRL expects the MMRL visual encoder to expose `proj_rep`."
            )

        proj_rep = visual.proj_rep.detach().float()

        if self.bayes_r_enabled:
            r_prior_mode = str(method_cfg.R_PRIOR_MODE)

            if r_prior_mode == "zero":
                prior_mean = torch.zeros_like(proj_rep)
            elif r_prior_mode == "self_proj_rep":
                prior_mean = proj_rep
            else:
                raise ValueError(
                    f"Unsupported BAYESRT_MMRL.R_PRIOR_MODE={r_prior_mode}. "
                    "Expected one of {'zero', 'self_proj_rep'}."
                )

            self.bayes_proj_rep = FactorizedGaussianMatrix(
                mean=prior_mean,
                sigma_mode=str(method_cfg.R_SIGMA_MODE),
                prior_std=float(method_cfg.R_PRIOR_STD),
                train_mean=True,
                min_sigma=1.0e-6,
            )
        else:
            self.bayes_proj_rep = None

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_hidden(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual,
    ):
        v = self.visual

        if not hasattr(v, "conv1") or not hasattr(v, "transformer"):
            raise TypeError(
                "BayesRTVisualWrapper currently supports the ViT-style MMRL visual "
                "encoder used by CLIP ViT-B/16."
            )

        x = image.type(self.dtype)

        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        cls = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([cls, x], dim=1)

        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)

        x = x.permute(1, 0, 2)
        outputs = v.transformer([x, compound_rep_tokens_visual, 0])
        x = outputs[0]
        x = x.permute(1, 0, 2)

        n_tokens = compound_rep_tokens_visual[0].shape[0]

        cls_hidden = v.ln_post(x[:, 0, :])
        rep_hidden = v.ln_post(x[:, 1 : 1 + n_tokens, :]).mean(dim=1)

        return cls_hidden, rep_hidden

    def forward_mean(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual,
    ):
        cls_hidden, rep_hidden = self.encode_hidden(
            image,
            compound_rep_tokens_visual,
        )

        image_main = cls_hidden @ self.visual.proj

        if self.bayes_proj_rep is None:
            image_rep = rep_hidden @ self.visual.proj_rep
        else:
            w_mean = self.bayes_proj_rep.posterior_mean.to(rep_hidden.dtype)
            image_rep = rep_hidden @ w_mean

        return image_main, image_rep, cls_hidden, rep_hidden

    def rep_samples(
        self,
        rep_hidden: torch.Tensor,
        num_samples: int,
        use_mean: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            [S, B, d]
        """
        num_samples = max(1, int(num_samples))

        if self.bayes_proj_rep is None:
            rep = rep_hidden @ self.visual.proj_rep
            return rep.unsqueeze(0).expand(num_samples, *rep.shape)

        w_samples = self.bayes_proj_rep.sample_many(
            num_samples,
            use_mean=use_mean,
        ).to(rep_hidden.dtype)

        return torch.einsum("bd,sdk->sbk", rep_hidden, w_samples)

    def kl_divergence(self) -> torch.Tensor:
        if self.bayes_proj_rep is None:
            return self.visual.proj_rep.new_zeros(())
        return self.bayes_proj_rep.kl_divergence()


class BayesRTMMRLModel(nn.Module):
    """
    Independent BayesRT-MMRL model.

    R side:
        Optional Bayesian posterior over image representation branch projection P_v^r.

    T side:
        Optional fixed-mean Bayesian posterior over CLIP text projection P_t.

    Switches:
        BAYES_R_ENABLED: True/False
        BAYES_T_ENABLED: True/False
    """

    def __init__(self, cfg, method_cfg, classnames, clip_model):
        super().__init__()

        self.alpha = float(method_cfg.ALPHA)
        self.dtype = clip_model.dtype
        self.bayes_r_enabled = bool(method_cfg.BAYES_R_ENABLED)
        self.bayes_t_enabled = bool(method_cfg.BAYES_T_ENABLED)

        self.representation_learner = MMRLFamilyRepresentationLearner(
            cfg,
            method_cfg,
            classnames,
            clip_model,
        ).type(clip_model.dtype)

        self.register_buffer(
            "tokenized_prompts",
            self.representation_learner.tokenized_prompts.clone(),
        )
        self.register_buffer(
            "prompt_embeddings",
            self.representation_learner.prompt_embeddings.clone(),
        )

        self.image_encoder = BayesRTVisualWrapper(clip_model.visual, method_cfg)
        self.text_encoder = BayesRTTextEncoderHidden(clip_model)

        text_projection = clip_model.text_projection.detach().float()

        if self.bayes_t_enabled:
            self.text_posterior = FactorizedGaussianMatrix(
                mean=text_projection,
                sigma_mode=str(method_cfg.T_SIGMA_MODE),
                prior_std=float(method_cfg.T_PRIOR_STD),
                train_mean=False,
                min_sigma=float(method_cfg.T_MIN_SIGMA),
            )
        else:
            self.text_posterior = None
            self.register_buffer("text_projection", text_projection)


    @staticmethod
    def probs_bma(logits_stack: torch.Tensor) -> torch.Tensor:
        """
        Bayesian model averaging in probability space.

        Args:
            logits_stack: [S, B, C]

        Returns:
            p_bar: [B, C]
        """
        return torch.softmax(logits_stack.float(), dim=-1).mean(dim=0)

    @staticmethod
    def log_probs_bma(logits_stack: torch.Tensor) -> torch.Tensor:
        """
        Returns log posterior predictive probabilities: log p_bar.
        """
        probs = BayesRTMMRLModel.probs_bma(logits_stack)
        return torch.log(probs.clamp_min(1.0e-12)).to(logits_stack.dtype)

    @staticmethod
    def logits_mean(logits_stack: torch.Tensor) -> torch.Tensor:
        """
        MC average in logit space.
        """
        return logits_stack.mean(dim=0)

    @staticmethod
    def aggregate_logits(
        logits_stack: torch.Tensor,
        aggregation: str = "prob_mean",
    ) -> torch.Tensor:
        """
        Backward-compatible helper.

        New BayesRT fusion code should prefer consuming logits_stack directly.
        """
        if aggregation == "logit_mean":
            return BayesRTMMRLModel.logits_mean(logits_stack)

        if aggregation == "prob_mean":
            return BayesRTMMRLModel.log_probs_bma(logits_stack)

        raise ValueError(
            f"Unsupported aggregation={aggregation}. "
            "Expected one of {'prob_mean', 'logit_mean'}."
        )


    def text_sample_features(
        self,
        eot_hidden: torch.Tensor,
        num_samples: int,
        use_mean: bool = False,
    ):
        """
        Returns:
            text_samples: [S, C, d]
            text_mean:    [C, d]
        """
        num_samples = max(1, int(num_samples))

        if self.text_posterior is None:
            text_mean = eot_hidden.float() @ self.text_projection.to(eot_hidden.device)
            text_mean = F.normalize(text_mean, dim=-1)
            text_samples = text_mean.unsqueeze(0).expand(num_samples, *text_mean.shape)
            return text_samples, text_mean

        p_samples = self.text_posterior.sample_many(
            num_samples,
            use_mean=use_mean,
        ).to(eot_hidden.device)

        text_samples = torch.einsum(
            "cd,sdk->sck",
            eot_hidden.float(),
            p_samples,
        )
        text_samples = F.normalize(text_samples, dim=-1)

        text_mean = eot_hidden.float() @ self.text_posterior.posterior_mean.to(
            eot_hidden.device
        )
        text_mean = F.normalize(text_mean, dim=-1)

        return text_samples, text_mean

    def forward_joint(
        self,
        image: torch.Tensor,
        num_samples: int,
        use_posterior_mean: bool = False,
        aggregation: str | None = None,  # deprecated, ignored
    ):
        num_samples = max(1, int(num_samples))

        compound_rep_tokens_text, compound_rep_tokens_visual = (
            self.representation_learner()
        )

        eot_hidden = self.text_encoder.forward_hidden(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )

        text_samples, text_mean = self.text_sample_features(
            eot_hidden=eot_hidden,
            num_samples=num_samples,
            use_mean=use_posterior_mean,
        )

        image_main, image_rep_mean, _, rep_hidden = self.image_encoder.forward_mean(
            image.type(self.dtype),
            compound_rep_tokens_visual,
        )

        image_rep_stack = self.image_encoder.rep_samples(
            rep_hidden=rep_hidden,
            num_samples=num_samples,
            use_mean=use_posterior_mean,
        )

        image_main = F.normalize(image_main, dim=-1)
        image_rep_mean = F.normalize(image_rep_mean, dim=-1)
        image_rep_stack = F.normalize(image_rep_stack, dim=-1)

        text_samples = text_samples.type(image_main.dtype)
        text_mean = text_mean.type(image_main.dtype)

        logits_main_stack = 100.0 * torch.einsum(
            "bd,scd->sbc",
            image_main,
            text_samples,
        )

        logits_rep_stack = 100.0 * torch.einsum(
            "sbd,scd->sbc",
            image_rep_stack,
            text_samples,
        )

        # Keep branch posterior predictive outputs for logging / fallback only.
        # Fusion should be built from logits_*_stack in BayesRTMMRLMethod.
        p_main = self.probs_bma(logits_main_stack)
        p_rep = self.probs_bma(logits_rep_stack)

        logits_main = torch.log(p_main.clamp_min(1.0e-12)).to(logits_main_stack.dtype)
        logits_rep = torch.log(p_rep.clamp_min(1.0e-12)).to(logits_rep_stack.dtype)

        # Backward-compatible static probability fusion.
        p_fusion = self.alpha * p_main + (1.0 - self.alpha) * p_rep
        logits_fusion = torch.log(p_fusion.clamp_min(1.0e-12)).to(
            logits_main_stack.dtype
        )

        return {
            "logits_main": logits_main,
            "logits_rep": logits_rep,
            "logits_fusion": logits_fusion,
            "logits_main_stack": logits_main_stack,
            "logits_rep_stack": logits_rep_stack,
            "image_features_main": image_main,
            "image_features_rep": image_rep_mean,
            "text_features": text_mean,
            "text_samples": text_samples,
        }












    def kl_terms(self):
        zero = next(self.parameters()).new_zeros(())

        kl_r = self.image_encoder.kl_divergence()

        kl_t = zero
        if self.text_posterior is not None:
            kl_t = self.text_posterior.kl_divergence()

        return {
            "r_proj": kl_r,
            "t_proj": kl_t,
        }