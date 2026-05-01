from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


def _cad_get(cfg, name: str, default):
    return getattr(cfg.CLIP_ADAPTERS, name, default)


def _softplus_inverse(x: float) -> float:
    x = float(x)
    if x <= 0:
        raise ValueError("softplus inverse input must be positive")
    return math.log(math.expm1(x))


class SBEAAdapter(BaseAdapter):
    """
    Sparse Bayesian Energy Adapter.

    This is a CLIP_ADAPTERS adapter, not a standalone METHOD.

    Routing:
        METHOD.NAME = ClipAdapters
        CLIP_ADAPTERS.INIT = SBEA

    Core idea:
        prototype_c^(s) = base_text_feature_c + delta_c^(s)

        delta_cj ~ q(delta_cj)
        q(delta_cj) = N(mu_cj, sigma_cj^2)

        ARD prior:
        p(delta_cj) = N(0, prior_sigma_j^2)
    """
    initialization_name = "BAYES_ADAPTER"
    sbea_initialization_name = "SBEA"
    adapter_kind = "stochastic_prototype"


    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        base_text_features = base_text_features.detach().clone()

        n_classes, feat_dim = base_text_features.shape
        self.num_classes = int(n_classes)
        self.clip_latent_dim = int(feat_dim)

        self.register_buffer("prior_mean", base_text_features.clone())

        # ----------------------------
        # Bayesian residual q(delta)
        # ----------------------------
        init_std = float(_cad_get(cfg, "SBEA_POSTERIOR_INIT_STD", 0.005))
        min_std = float(_cad_get(cfg, "SBEA_MIN_STD", 1.0e-6))

        if init_std <= 0:
            raise ValueError(f"SBEA_POSTERIOR_INIT_STD must be > 0, got {init_std}")
        if min_std <= 0:
            raise ValueError(f"SBEA_MIN_STD must be > 0, got {min_std}")

        self.min_std = min_std

        self.delta_mu = nn.Parameter(
            torch.zeros_like(base_text_features)
        )

        self.delta_rho = nn.Parameter(
            torch.full_like(
                base_text_features,
                _softplus_inverse(init_std),
            )
        )

        # ----------------------------
        # ARD prior over residual dims
        # ----------------------------
        prior_std = float(_cad_get(cfg, "SBEA_PRIOR_STD", 0.05))
        if prior_std <= 0:
            raise ValueError(f"SBEA_PRIOR_STD must be > 0, got {prior_std}")

        self.prior_mode = str(_cad_get(cfg, "SBEA_PRIOR_MODE", "ard")).lower()
        self.ard_learn_prior = bool(_cad_get(cfg, "SBEA_ARD_LEARN_PRIOR", True))

        prior_logstd_init = math.log(prior_std)

        if self.prior_mode == "ard":
            if self.ard_learn_prior:
                self.prior_logstd = nn.Parameter(
                    torch.full(
                        (self.clip_latent_dim,),
                        prior_logstd_init,
                        dtype=base_text_features.dtype,
                        device=base_text_features.device,
                    )
                )
            else:
                self.register_buffer(
                    "prior_logstd",
                    torch.full(
                        (self.clip_latent_dim,),
                        prior_logstd_init,
                        dtype=base_text_features.dtype,
                        device=base_text_features.device,
                    ),
                )
        elif self.prior_mode in {"fixed", "isotropic"}:
            self.register_buffer(
                "prior_logstd",
                torch.full(
                    (self.clip_latent_dim,),
                    prior_logstd_init,
                    dtype=base_text_features.dtype,
                    device=base_text_features.device,
                ),
            )
        else:
            raise ValueError(
                f"Unsupported SBEA_PRIOR_MODE={self.prior_mode!r}. "
                "Use 'ard' or 'fixed'."
            )

        self.register_buffer(
            "prior_logstd_anchor",
            torch.full(
                (self.clip_latent_dim,),
                prior_logstd_init,
                dtype=base_text_features.dtype,
                device=base_text_features.device,
            ),
        )

        # ----------------------------
        # Optional feature adapter
        # ----------------------------
        self.use_feature_adapter = bool(_cad_get(cfg, "SBEA_USE_FEATURE_ADAPTER", True))

        bottleneck_dim = int(_cad_get(cfg, "SBEA_ADAPTER_DIM", 0))
        bottleneck_ratio = float(_cad_get(cfg, "SBEA_ADAPTER_BOTTLENECK_RATIO", 0.25))

        if bottleneck_dim <= 0:
            bottleneck_dim = max(1, int(round(self.clip_latent_dim * bottleneck_ratio)))

        self.feature_scale = nn.Parameter(
            torch.tensor(
                float(_cad_get(cfg, "SBEA_FEATURE_SCALE_INIT", 0.1)),
                dtype=base_text_features.dtype,
                device=base_text_features.device,
            )
        )

        if self.use_feature_adapter:
            self.feature_adapter = nn.Sequential(
                nn.LayerNorm(self.clip_latent_dim),
                nn.Linear(self.clip_latent_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, self.clip_latent_dim),
            )
        else:
            self.feature_adapter = nn.Identity()

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()

        if not self.use_feature_adapter:
            return features

        residual = self.feature_adapter(features)
        return features + self.feature_scale.float() * residual

    def posterior_std(self) -> torch.Tensor:
        return F.softplus(self.delta_rho.float()) + float(self.min_std)

    def prior_std(self) -> torch.Tensor:
        return torch.exp(self.prior_logstd.float()).clamp_min(float(self.min_std))

    def get_prototypes(self) -> torch.Tensor:
        """
        Deterministic posterior mean prototypes [C, D].
        """
        return self.prior_mean + self.delta_mu

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample residual text prototypes.

        Returns:
            prototypes: [S, C, D]
        """
        eps = torch.randn(
            n_samples,
            self.num_classes,
            self.clip_latent_dim,
            device=self.delta_mu.device,
            dtype=self.delta_mu.dtype,
        )

        std = self.posterior_std().to(dtype=self.delta_mu.dtype)

        delta = self.delta_mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return self.prior_mean.unsqueeze(0) + delta

    def kl_divergence(self) -> torch.Tensor:
        """
        KL[q(delta) || p(delta)].

        q(delta_cj) = N(mu_cj, sigma_cj^2)
        p(delta_cj) = N(0, prior_sigma_j^2)
        """
        mu = self.delta_mu.float()
        q_std = self.posterior_std()
        p_std = self.prior_std().view(1, self.clip_latent_dim)

        q_var = q_std.pow(2)
        p_var = p_std.pow(2)

        kl = (
            torch.log(p_std / q_std)
            + 0.5 * (q_var + mu.pow(2)) / p_var
            - 0.5
        )

        return kl.sum()

    def bayes_kl_base_weight(self) -> float:
        """
        Keep compatible with ClipAdaptersMethod._bayes_kl_weight().
        """
        if hasattr(self.cfg.CLIP_ADAPTERS, "SBEA_KL_WEIGHT"):
            return float(self.cfg.CLIP_ADAPTERS.SBEA_KL_WEIGHT)

        return float(getattr(self.cfg.CLIP_ADAPTERS, "KL_WEIGHT", 1.0e-4))

    def ard_hyper_regularization(self) -> torch.Tensor:
        """
        Stabilizer for learnable ARD prior std.
        """
        if not (self.prior_mode == "ard" and self.ard_learn_prior):
            return self.delta_mu.new_tensor(0.0)

        scale = float(_cad_get(self.cfg, "SBEA_ARD_HYPER_SCALE", 1.0))
        scale = max(scale, 1.0e-6)

        diff = (self.prior_logstd.float() - self.prior_logstd_anchor.float()) / scale
        return 0.5 * diff.pow(2).sum()

    def prototype_anchor_regularization(self) -> torch.Tensor:
        return self.delta_mu.float().pow(2).mean()

    @staticmethod
    def mc_predictive_log_probs(logits_all: torch.Tensor) -> torch.Tensor:
        """
        log E_s[softmax(logits_s)].

        Args:
            logits_all: [S, B, C]
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"mc_predictive_log_probs expects [S, B, C], got {tuple(logits_all.shape)}"
            )

        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)

        n_samples = torch.tensor(
            float(x.shape[0]),
            dtype=x.dtype,
            device=x.device,
        )

        return torch.logsumexp(log_probs, dim=0) - torch.log(n_samples)

    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Used by ClipAdaptersModel.forward_features() when adapter exposes this hook.

        During training, optionally keep mean logits for compatibility.
        During eval, default to posterior predictive log probabilities.
        """
        train_return = str(
            _cad_get(self.cfg, "SBEA_TRAIN_RETURN", "mean_logits")
        ).lower()

        eval_return = str(
            _cad_get(self.cfg, "SBEA_EVAL_RETURN", "mc_predictive")
        ).lower()

        mode = train_return if training else eval_return

        if mode in {"mean_logits", "logit_mean"}:
            return logits_all.mean(dim=0)

        if mode in {"mc_predictive", "posterior_predictive", "prob_mean"}:
            return self.mc_predictive_log_probs(logits_all).to(dtype=logits_all.dtype)

        raise ValueError(
            f"Unsupported SBEA return mode {mode!r}. "
            "Use mean_logits or mc_predictive."
        )

    def sbea_uncertainty(self, logits_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optional MC energy / epistemic uncertainty statistics.

        This does not affect other adapters because it is only called by SBEA loss
        or future SBEA-specific OOD evaluation code.
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"sbea_uncertainty expects [S, B, C], got {tuple(logits_all.shape)}"
            )

        eps = float(_cad_get(self.cfg, "SBEA_EPS", 1.0e-12))
        temp = float(_cad_get(self.cfg, "SBEA_ENERGY_TEMPERATURE", 1.0))
        temp = max(temp, eps)

        x = logits_all.float()
        probs_all = torch.softmax(x, dim=-1)
        mean_probs = probs_all.mean(dim=0)

        pred_entropy = -(
            mean_probs.clamp_min(eps) * mean_probs.clamp_min(eps).log()
        ).sum(dim=-1)

        expected_entropy = -(
            probs_all.clamp_min(eps) * probs_all.clamp_min(eps).log()
        ).sum(dim=-1).mean(dim=0)

        mutual_information = pred_entropy - expected_entropy

        energies = -temp * torch.logsumexp(x / temp, dim=-1)

        return {
            "predictive_entropy": pred_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
            "energy_mean": energies.mean(dim=0),
            "energy_variance": energies.var(dim=0, unbiased=False),
            "confidence": mean_probs.max(dim=-1).values,
            "prob_variance": probs_all.var(dim=0, unbiased=False).mean(dim=-1),
        }