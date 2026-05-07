from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from methods.mmrl_family.modules import MMRLFamilyModel


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x))


class BayesianTextFeatureHead(nn.Module):
    """
    Posterior over MMRL text features:

        q(w_c) = N(mu_c, sigma_c^2 I)

    mu_c is not a free parameter here. It is the MMRL text-side output.
    sigma_c is one learned scalar per class.
    """

    def __init__(
        self,
        num_classes: int,
        prior_std: float = 0.05,
        init_std: float | None = None,
        min_sigma: float = 1e-6,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.min_sigma = float(min_sigma)

        prior_std = float(prior_std)

        if prior_std <= self.min_sigma:
            raise ValueError(
                f"prior_std must be greater than min_sigma, "
                f"got prior_std={prior_std}, min_sigma={self.min_sigma}"
            )

        self.register_buffer(
            "prior_std",
            torch.full((self.num_classes, 1), prior_std, dtype=torch.float32),
        )

        # Initialize posterior std exactly as prior std:
        # softplus(rho) + min_sigma = prior_std
        rho_init = _softplus_inverse(
            torch.full(
                (self.num_classes, 1),
                prior_std - self.min_sigma,
                dtype=torch.float32,
            )
        )
        self.posterior_rho = nn.Parameter(rho_init)


    def posterior_sigma(self) -> torch.Tensor:
        return F.softplus(self.posterior_rho.float()) + self.min_sigma

    def sample(
        self,
        mean: torch.Tensor,
        num_samples: int,
        use_posterior_mean: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            mean: [C, d], normalized MMRL text features
        Returns:
            samples: [S, C, d], normalized sampled text features
        """
        num_samples = max(1, int(num_samples))
        mean = mean.float()

        if use_posterior_mean:
            samples = mean.unsqueeze(0).expand(num_samples, *mean.shape)
            return F.normalize(samples, dim=-1)

        eps = torch.randn(
            num_samples,
            mean.shape[0],
            mean.shape[1],
            device=mean.device,
            dtype=mean.dtype,
        )

        sigma = self.posterior_sigma().to(mean.device, mean.dtype).unsqueeze(0)
        samples = mean.unsqueeze(0) + sigma * eps
        return F.normalize(samples, dim=-1)

    def kl_divergence(
        self,
        mean: torch.Tensor,
        prior_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL[q(w)||p(w)] with

            q(w_c) = N(mean_c, sigma_c^2 I)
            p(w_c) = N(prior_mean_c, prior_std_c^2 I)

        Args:
            mean: [C, d]
            prior_mean: [C, d]
        """
        mean = mean.float()
        prior_mean = prior_mean.float().to(mean.device)

        if mean.shape != prior_mean.shape:
            raise ValueError(
                f"text prior shape mismatch: mean={tuple(mean.shape)}, "
                f"prior={tuple(prior_mean.shape)}"
            )

        sigma_q = self.posterior_sigma().to(mean.device)
        sigma_p = self.prior_std.to(mean.device)

        sigma_q2 = sigma_q.pow(2)
        sigma_p2 = sigma_p.pow(2)

        d = mean.shape[-1]
        mean_delta2 = (mean - prior_mean).pow(2).sum(dim=-1, keepdim=True)

        kl_per_class = 0.5 * (
            d * torch.log(sigma_p2 / sigma_q2)
            + (d * sigma_q2 + mean_delta2) / sigma_p2
            - d
        )

        return kl_per_class.sum()


class BayesTextMMRLModel(MMRLFamilyModel):
    """
    MMRL + Bayesian posterior over text output features.

    The image encoder and representation learner are inherited from MMRLFamilyModel.
    The new stochastic part is only text_features -> sampled text_features.
    """

    def __init__(self, cfg, method_cfg, classnames, clip_model):
        super().__init__(cfg, method_cfg, classnames, clip_model)

        self.text_posterior = BayesianTextFeatureHead(
            num_classes=len(classnames),
            prior_std=float(method_cfg.TEXT_PRIOR_STD),
            min_sigma=float(getattr(method_cfg, "TEXT_MIN_SIGMA", 1e-6)),
        )

    @staticmethod
    def _logits_from_text_samples(
        image_features: torch.Tensor,
        text_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, d]
            text_samples: [S, C, d]

        Returns:
            logits: [S, B, C]
        """
        return 100.0 * torch.einsum("bd,scd->sbc", image_features, text_samples)

    @staticmethod
    def _aggregate_logits(
        logits_stack: torch.Tensor,
        aggregation: str = "prob_mean",
    ) -> torch.Tensor:
        """
        Args:
            logits_stack: [S, B, C]

        Returns:
            [B, C]
        """
        if aggregation == "logit_mean":
            return logits_stack.mean(dim=0)

        if aggregation == "prob_mean":
            probs = torch.softmax(logits_stack.float(), dim=-1).mean(dim=0)
            return torch.log(probs.clamp_min(1e-12)).to(logits_stack.dtype)

        raise ValueError(
            f"Unsupported aggregation={aggregation}. "
            "Expected one of {'prob_mean', 'logit_mean'}."
        )

    def forward_bayes_text(
        self,
        image: torch.Tensor,
        num_samples: int,
        use_posterior_mean: bool = False,
        aggregation: str = "prob_mean",
    ):
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual = (
                self.representation_learner()
            )
            text_mean = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                compound_rep_tokens_text,
            )
        else:
            if self.text_features_for_inference is None:
                rep_text, rep_visual = self.representation_learner()
                self.compound_rep_tokens_text_for_inference = rep_text
                self.compound_rep_tokens_visual_for_inference = rep_visual
                self.text_features_for_inference = self.text_encoder(
                    self.prompt_embeddings,
                    self.tokenized_prompts,
                    self.compound_rep_tokens_text_for_inference,
                )

            compound_rep_tokens_visual = self.compound_rep_tokens_visual_for_inference
            text_mean = self.text_features_for_inference

        image_features, image_features_rep = self.image_encoder(
            [image.type(self.dtype), compound_rep_tokens_visual]
        )

        image_features = F.normalize(image_features, dim=-1)
        image_features_rep = F.normalize(image_features_rep, dim=-1)
        text_mean = F.normalize(text_mean, dim=-1)

        text_samples = self.text_posterior.sample(
            text_mean,
            num_samples=num_samples,
            use_posterior_mean=use_posterior_mean,
        ).type(image_features.dtype)

        logits_stack = self._logits_from_text_samples(image_features, text_samples)
        logits_rep_stack = self._logits_from_text_samples(
            image_features_rep,
            text_samples,
        )

        logits = self._aggregate_logits(logits_stack, aggregation=aggregation)
        logits_rep = self._aggregate_logits(logits_rep_stack, aggregation=aggregation)
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return {
            "logits": logits,
            "logits_rep": logits_rep,
            "logits_fusion": logits_fusion,
            "logits_stack": logits_stack,
            "logits_rep_stack": logits_rep_stack,
            "image_features": image_features,
            "image_features_rep": image_features_rep,
            "text_mean": text_mean,
            "text_samples": text_samples,
        }