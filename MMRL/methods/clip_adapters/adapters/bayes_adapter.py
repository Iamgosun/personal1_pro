from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseAdapter


class BayesAdapter(BaseAdapter):
    """
    BayesAdapter with the same core parameterization as the uploaded code:
    - posterior mean: [C, D]
    - posterior logstd: [C]   (one scalar std per class, shared across feature dims)
    - sampling is done in the unnormalized text-feature space
    """

    initialization_name = "BAYES_ADAPTER"
    adapter_kind = "stochastic_prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        n_classes, feat_dim = base_text_features.shape
        self.num_classes = int(n_classes)
        self.clip_latent_dim = int(feat_dim)

        prior_std = float(getattr(cfg.CLIP_ADAPTERS, "BAYES_PRIOR_STD", 0.01))
        if prior_std <= 0:
            raise ValueError(f"BAYES_PRIOR_STD must be > 0, got {prior_std}")

        prior_logstd = math.log(prior_std)

        # q(W): mean [C, D]
        self.text_features_unnorm_mean = nn.Parameter(base_text_features.clone())

        # q(W): shared scalar std per class -> logstd [C]
        self.text_features_unnorm_logstd = nn.Parameter(
            torch.full(
                (self.num_classes,),
                prior_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        # p(W): initialized from zero-shot/base text features
        self.register_buffer("prior_mean", base_text_features.clone())
        self.register_buffer(
            "prior_logstd",
            torch.full(
                (self.num_classes,),
                prior_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

    def get_prototypes(self) -> torch.Tensor:
        """
        Deterministic posterior mean prototypes [C, D].
        """
        return self.text_features_unnorm_mean

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample prototypes in unnormalized text-feature space.
        Returns [S, C, D].
        """
        eps = torch.randn(
            n_samples,
            self.num_classes,
            self.clip_latent_dim,
            device=self.text_features_unnorm_mean.device,
            dtype=self.text_features_unnorm_mean.dtype,
        )
        std = torch.exp(self.text_features_unnorm_logstd).unsqueeze(-1)  # [C, 1]
        return self.text_features_unnorm_mean.unsqueeze(0) + eps * std.unsqueeze(0)

    def kl_divergence(self) -> torch.Tensor:
        """
        Keep the same KL form used by the uploaded BayesAdapter code,
        instead of switching to the repo's GaussianPerClassAdapter KL.

        Uploaded code uses:
            kl_trace + kl_diff_sq + kl_logdet
        (without the usual 0.5 and constant -k terms)
        """
        posterior_std = torch.exp(self.text_features_unnorm_logstd)  # [C]
        prior_std = torch.exp(self.prior_logstd)  # [C]

        kl_trace = self.clip_latent_dim * (
            posterior_std.pow(2) / prior_std.pow(2)
        ).sum()

        kl_diff_sq = (
            (self.text_features_unnorm_mean - self.prior_mean).pow(2)
            / prior_std.pow(2).unsqueeze(-1)
        ).sum()

        kl_logdet = self.clip_latent_dim * (
            prior_std.pow(2).log() - posterior_std.pow(2).log()
        ).sum()

        return kl_trace + kl_diff_sq + kl_logdet

    def bayes_kl_base_weight(self) -> float:
        """
        Matches the uploaded code's scale:
            1 / (1000 * num_classes * clip_latent_dim)
        The epoch-dependent linear ramp is handled in the method layer.
        """
        return 1.0 / (1000.0 * self.num_classes * self.clip_latent_dim)