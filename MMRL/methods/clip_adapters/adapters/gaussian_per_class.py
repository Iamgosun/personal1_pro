import math

import torch
import torch.nn as nn

from .base import BaseAdapter


class GaussianPerClassAdapter(BaseAdapter):
    initialization_name = "GAUSSIAN_PER_CLASS"
    adapter_kind = "stochastic_prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using per-class Gaussian initialization")

        n_classes, feat_dim = base_text_features.shape
        self.variational_mu = nn.Parameter(base_text_features.clone())
        self.variational_log_sigma = nn.Parameter(
            torch.full((n_classes, feat_dim), math.log(0.01), device=base_text_features.device)
        )

        self.register_buffer("prior_mu", base_text_features.clone())
        self.register_buffer(
            "prior_sigma",
            torch.full((n_classes, feat_dim), 0.01, device=base_text_features.device),
        )

    def get_prototypes(self) -> torch.Tensor:
        return self.variational_mu

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        epsilon = torch.randn(
            (n_samples, *self.variational_log_sigma.shape),
            device=self.variational_mu.device,
            dtype=self.variational_mu.dtype,
        )
        sigma = torch.exp(self.variational_log_sigma)
        return self.variational_mu.unsqueeze(0) + epsilon * sigma.unsqueeze(0)

    def kl_divergence(self):
        posterior_sigma = torch.exp(self.variational_log_sigma) + 1e-8
        prior_sigma = self.prior_sigma.to(self.variational_mu.device) + 1e-8
        prior_mu = self.prior_mu.to(self.variational_mu.device)

        kl_per_dim = (
            (posterior_sigma ** 2 + (self.variational_mu - prior_mu) ** 2) / (2 * prior_sigma ** 2)
            - 0.5
            - torch.log(posterior_sigma)
            + torch.log(prior_sigma)
        )
        kl = kl_per_dim.sum()
        return torch.clamp(kl, max=1e4)

    def extra_loss(self):
        return self.kl_divergence()