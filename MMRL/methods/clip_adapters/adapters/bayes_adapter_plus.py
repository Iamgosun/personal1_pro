from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


class BayesAdapterPlus(BaseAdapter):
    """
    Stable BayesAdapter++.

    Design goals:
    1. Keep accuracy close to the original BayesAdapter by default.
    2. Use empirical-Bayes support statistics conservatively.
    3. Avoid large prototype sampling noise in high-dimensional CLIP space.
    4. Avoid treating CACHE_REPS augmentations as real shots.

    Compatibility:
    - initialization_name stays "BAYES_ADAPTER" so existing model.py and loss.py
      Bayes paths are reused.
    - bayes_plus_initialization_name is used only for identification/debugging.
    """

    initialization_name = "BAYES_ADAPTER"
    bayes_plus_initialization_name = "BAYES_ADAPTER_PLUS"
    adapter_kind = "stochastic_prototype"
    needs_support_features = True
    uses_cache = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        n_classes, feat_dim = base_text_features.shape
        self.num_classes = int(n_classes)
        self.clip_latent_dim = int(feat_dim)

        ca = cfg.CLIP_ADAPTERS

        prior_std = float(getattr(ca, "BAYES_PRIOR_STD", 0.01))
        if prior_std <= 0:
            raise ValueError(f"BAYES_PRIOR_STD must be > 0, got {prior_std}")

        # Keep posterior std small by default. This is critical for accuracy.
        posterior_init_std = float(
            getattr(ca, "BAYES_PLUS_POSTERIOR_INIT_STD", prior_std)
        )
        if posterior_init_std <= 0:
            raise ValueError(
                f"BAYES_PLUS_POSTERIOR_INIT_STD must be > 0, got {posterior_init_std}"
            )

        self.rank = int(getattr(ca, "BAYES_PLUS_RANK", 0))
        if self.rank < 0:
            raise ValueError(f"BAYES_PLUS_RANK must be >= 0, got {self.rank}")

        # EB mean is conservative by default.
        self.eb_tau = float(getattr(ca, "BAYES_PLUS_EB_TAU", 16.0))
        self.eb_alpha_max = float(getattr(ca, "BAYES_PLUS_EB_ALPHA_MAX", 0.35))

        # EB variance must be per-dimension and tightly clipped.
        self.eb_var_a = float(getattr(ca, "BAYES_PLUS_EB_VAR_A", prior_std * prior_std))
        self.eb_var_b = float(getattr(ca, "BAYES_PLUS_EB_VAR_B", 0.05))
        self.prior_min_std = float(getattr(ca, "BAYES_PLUS_PRIOR_MIN_STD", 0.003))
        self.prior_max_std = float(getattr(ca, "BAYES_PLUS_PRIOR_MAX_STD", 0.03))

        # By default, do not move posterior mean/std to EB prior.
        # This preserves original BayesAdapter-like accuracy.
        self.shift_posterior_mean_to_eb = bool(
            getattr(ca, "BAYES_PLUS_SHIFT_POSTERIOR_MEAN_TO_EB", False)
        )
        self.shift_posterior_std_to_eb = bool(
            getattr(ca, "BAYES_PLUS_SHIFT_POSTERIOR_STD_TO_EB", False)
        )

        # Avoid counting repeated cache augmentations as real shots.
        self.effective_reps = int(
            getattr(
                ca,
                "BAYES_PLUS_EFFECTIVE_REPS",
                getattr(ca, "CACHE_REPS", 1),
            )
        )
        self.effective_reps = max(1, self.effective_reps)

        self.maha_shrinkage = float(getattr(ca, "BAYES_PLUS_MAHA_SHRINKAGE", 0.50))
        self.maha_eps = float(getattr(ca, "BAYES_PLUS_MAHA_EPS", 1e-4))

        self.low_rank_kl_scale = float(
            getattr(ca, "BAYES_PLUS_LOW_RANK_KL_SCALE", 1e-4)
        )

        prior_logstd = math.log(prior_std)
        posterior_logstd = math.log(posterior_init_std)

        # q(W): posterior mean [C, D], initialized exactly from zero-shot text features.
        self.text_features_unnorm_mean = nn.Parameter(base_text_features.clone())

        # q(W): diagonal posterior logstd [C, D].
        self.text_features_unnorm_logstd = nn.Parameter(
            torch.full(
                (self.num_classes, self.clip_latent_dim),
                posterior_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        # p(W): EB prior, initially zero-shot text features.
        self.register_buffer("prior_mean", base_text_features.detach().clone())
        self.register_buffer(
            "prior_logstd",
            torch.full(
                (self.num_classes, self.clip_latent_dim),
                prior_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

        # Optional low-rank posterior perturbation.
        # Default rank=0 because accuracy restoration comes first.
        if self.rank > 0:
            init_std = float(getattr(ca, "BAYES_PLUS_LOW_RANK_INIT_STD", 1e-4))
            init_scale = float(getattr(ca, "BAYES_PLUS_LOW_RANK_SCALE", 0.003))

            self.low_rank_class_factor = nn.Parameter(
                torch.randn(
                    self.num_classes,
                    self.rank,
                    device=base_text_features.device,
                    dtype=base_text_features.dtype,
                )
                * init_std
            )
            self.low_rank_feature_factor = nn.Parameter(
                torch.randn(
                    self.rank,
                    self.clip_latent_dim,
                    device=base_text_features.device,
                    dtype=base_text_features.dtype,
                )
                * init_std
            )
            self.low_rank_log_scale = nn.Parameter(
                torch.tensor(
                    math.log(max(init_scale, 1e-12)),
                    device=base_text_features.device,
                    dtype=base_text_features.dtype,
                )
            )
        else:
            self.low_rank_class_factor = None
            self.low_rank_feature_factor = None
            self.low_rank_log_scale = None

        base_norm = _normalize(base_text_features)
        self.register_buffer("support_class_means", base_norm.detach().clone())
        self.register_buffer(
            "support_counts_raw",
            torch.zeros(
                self.num_classes,
                device=base_text_features.device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "support_counts_effective",
            torch.zeros(
                self.num_classes,
                device=base_text_features.device,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "maha_inv_var",
            torch.ones(
                self.clip_latent_dim,
                device=base_text_features.device,
                dtype=torch.float32,
            ),
        )

    def get_prototypes(self) -> torch.Tensor:
        return self.text_features_unnorm_mean

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0, got {n_samples}")

        mean = self.text_features_unnorm_mean
        logstd = self.text_features_unnorm_logstd.clamp(min=-12.0, max=0.0)

        eps_diag = torch.randn(
            n_samples,
            self.num_classes,
            self.clip_latent_dim,
            device=mean.device,
            dtype=mean.dtype,
        )
        samples = mean.unsqueeze(0) + eps_diag * torch.exp(logstd).unsqueeze(0)

        if self.rank > 0:
            eps_lr = torch.randn(
                n_samples,
                self.rank,
                device=mean.device,
                dtype=mean.dtype,
            )
            low_rank_noise = torch.einsum(
                "sr,cr,rd->scd",
                eps_lr,
                self.low_rank_class_factor,
                self.low_rank_feature_factor,
            )
            samples = samples + torch.exp(self.low_rank_log_scale) * low_rank_noise

        return samples

    def _estimate_effective_counts(self, raw_counts: torch.Tensor) -> torch.Tensor:
        """
        Cache mode repeats each support image CACHE_REPS times.
        Effective shot count should be approximately raw_count / CACHE_REPS.
        """
        return (raw_counts / float(self.effective_reps)).clamp_min(0.0)

    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        """
        Build conservative empirical-Bayes prior and Mahalanobis statistics.

        Important fixes:
        - use effective shot count, not augmented cache count;
        - estimate variance per feature dimension, not one scalar expanded to D dims;
        - keep posterior initialized from zero-shot unless explicitly enabled.
        """
        if features_train is None or labels_train is None:
            return None

        device = self.text_features_unnorm_mean.device

        features = features_train.detach().to(device=device, dtype=torch.float32)
        labels = labels_train.detach().to(device=device, dtype=torch.long)

        if features.ndim != 2:
            raise ValueError(
                f"BayesAdapterPlus.build_cache expects features [N, D], got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.clip_latent_dim:
            raise ValueError(
                "BayesAdapterPlus feature dimension mismatch: "
                f"expected {self.clip_latent_dim}, got {features.shape[-1]}"
            )

        features = _normalize(features)
        base = _normalize(self.base_text_features.to(device=device))

        class_means = base.clone()
        class_var_dim = torch.zeros(
            self.num_classes,
            self.clip_latent_dim,
            device=device,
            dtype=torch.float32,
        )
        counts_raw = torch.zeros(
            self.num_classes,
            device=device,
            dtype=torch.float32,
        )

        for c in range(self.num_classes):
            mask = labels == c
            n_raw = int(mask.sum().item())
            counts_raw[c] = float(n_raw)

            if n_raw <= 0:
                continue

            feats_c = features[mask]
            mean_c = _normalize(feats_c.mean(dim=0, keepdim=True)).squeeze(0)
            class_means[c] = mean_c

            if n_raw > 1:
                # Per-dimension variance. This is much safer than a scalar
                # squared-distance variance expanded over all dimensions.
                class_var_dim[c] = (feats_c - mean_c.unsqueeze(0)).pow(2).mean(dim=0)
            else:
                class_var_dim[c].zero_()

        counts_eff = self._estimate_effective_counts(counts_raw)

        alpha = counts_eff / (counts_eff + max(self.eb_tau, 1e-12))
        alpha = alpha.clamp(min=0.0, max=max(0.0, min(self.eb_alpha_max, 1.0)))

        prior_mean = _normalize(
            (1.0 - alpha).unsqueeze(-1) * base
            + alpha.unsqueeze(-1) * class_means
        )

        prior_var = (
            self.eb_var_a / (counts_eff + 1.0).unsqueeze(-1)
            + self.eb_var_b * class_var_dim
        )
        prior_std = prior_var.sqrt().clamp(
            min=max(self.prior_min_std, 1e-12),
            max=max(self.prior_max_std, self.prior_min_std),
        )
        prior_logstd = prior_std.log()

        # Diagonal shared covariance for Mahalanobis score.
        assigned_means = class_means[labels.clamp(min=0, max=self.num_classes - 1)]
        residuals = features - assigned_means

        if residuals.numel() > 0:
            raw_var = residuals.pow(2).mean(dim=0)
        else:
            raw_var = torch.ones(
                self.clip_latent_dim,
                device=device,
                dtype=torch.float32,
            )

        global_var = raw_var.mean().clamp_min(self.maha_eps)
        shrink = min(max(self.maha_shrinkage, 0.0), 1.0)
        shared_var = (1.0 - shrink) * raw_var + shrink * global_var
        maha_inv_var = 1.0 / shared_var.clamp_min(self.maha_eps)

        with torch.no_grad():
            self.support_class_means.copy_(
                class_means.to(
                    device=self.support_class_means.device,
                    dtype=self.support_class_means.dtype,
                )
            )
            self.support_counts_raw.copy_(
                counts_raw.to(
                    device=self.support_counts_raw.device,
                    dtype=self.support_counts_raw.dtype,
                )
            )
            self.support_counts_effective.copy_(
                counts_eff.to(
                    device=self.support_counts_effective.device,
                    dtype=self.support_counts_effective.dtype,
                )
            )
            self.maha_inv_var.copy_(
                maha_inv_var.to(
                    device=self.maha_inv_var.device,
                    dtype=self.maha_inv_var.dtype,
                )
            )

            self.prior_mean.copy_(
                prior_mean.to(
                    device=self.prior_mean.device,
                    dtype=self.prior_mean.dtype,
                )
            )
            self.prior_logstd.copy_(
                prior_logstd.to(
                    device=self.prior_logstd.device,
                    dtype=self.prior_logstd.dtype,
                )
            )

            if self.shift_posterior_mean_to_eb:
                self.text_features_unnorm_mean.copy_(
                    self.prior_mean.to(dtype=self.text_features_unnorm_mean.dtype)
                )

            if self.shift_posterior_std_to_eb:
                self.text_features_unnorm_logstd.copy_(
                    self.prior_logstd.to(dtype=self.text_features_unnorm_logstd.dtype)
                )

        print(
            "[BayesAdapterPlus] conservative EB cache built: "
            f"features={tuple(features.shape)}, "
            f"classes={self.num_classes}, "
            f"rank={self.rank}, "
            f"raw_count_min={int(counts_raw.min().item())}, "
            f"raw_count_max={int(counts_raw.max().item())}, "
            f"eff_count_min={counts_eff.min().item():.2f}, "
            f"eff_count_max={counts_eff.max().item():.2f}, "
            f"alpha_max={alpha.max().item():.4f}, "
            f"prior_std_mean={prior_std.mean().item():.6f}, "
            f"prior_std_max={prior_std.max().item():.6f}"
        )

        return None

    def kl_divergence(self) -> torch.Tensor:
        q_logstd = self.text_features_unnorm_logstd.float().clamp(min=-12.0, max=0.0)
        p_logstd = self.prior_logstd.float().clamp(min=-12.0, max=0.0)

        q_var = torch.exp(2.0 * q_logstd)
        p_var = torch.exp(2.0 * p_logstd).clamp_min(1e-12)

        mean_diff_sq = (
            self.text_features_unnorm_mean.float() - self.prior_mean.float()
        ).pow(2)

        kl_diag = 0.5 * (
            (q_var + mean_diff_sq) / p_var
            - 1.0
            + 2.0 * (p_logstd - q_logstd)
        ).sum()

        if self.rank <= 0:
            return kl_diag

        low_rank_penalty = 0.5 * self.low_rank_kl_scale * (
            self.low_rank_class_factor.float().pow(2).sum()
            + self.low_rank_feature_factor.float().pow(2).sum()
            + torch.exp(self.low_rank_log_scale.float()).pow(2)
        )

        return kl_diag + low_rank_penalty

    def bayes_kl_base_weight(self) -> float:
        override = getattr(self.cfg.CLIP_ADAPTERS, "BAYES_PLUS_KL_BASE_WEIGHT", None)
        if override is not None:
            return float(override)

        return 1.0 / (1000.0 * self.num_classes * self.clip_latent_dim)

    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        if logits_all.ndim != 3:
            raise ValueError(
                "bayes_base_logits_from_mc expects logits_all [S, B, C], "
                f"got {tuple(logits_all.shape)}"
            )

        use_predictive_train = bool(
            getattr(self.cfg.CLIP_ADAPTERS, "BAYES_PLUS_TRAIN_PREDICTIVE_LOGPROB", False)
        )

        if training and not use_predictive_train:
            return logits_all.mean(dim=0)

        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        log_pred = torch.logsumexp(log_probs, dim=0) - math.log(float(logits_all.shape[0]))
        return log_pred.to(dtype=logits_all.dtype)

    @torch.no_grad()
    def mahalanobis_score(self, features: torch.Tensor) -> torch.Tensor:
        features = _normalize(features.to(device=self.support_class_means.device))
        means = _normalize(self.support_class_means)

        diff = features.unsqueeze(1) - means.unsqueeze(0)
        dist = (diff.pow(2) * self.maha_inv_var.float().view(1, 1, -1)).sum(dim=-1)
        return dist.min(dim=1).values

    @torch.no_grad()
    def uncertainty_scores(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        logits_all: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if logits_all is not None:
            x = logits_all.float()
            probs = torch.softmax(x, dim=-1).mean(dim=0)
            logits_mean = x.mean(dim=0)
        else:
            x = logits.float()
            # Eval logits may already be log posterior predictive probabilities.
            logsum = torch.logsumexp(x, dim=-1)
            if torch.allclose(logsum, torch.zeros_like(logsum), atol=1e-3, rtol=1e-3):
                probs = x.exp()
            else:
                probs = torch.softmax(x, dim=-1)
            logits_mean = x

        max_prob = probs.max(dim=-1).values
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_norm = entropy / math.log(float(self.num_classes))

        energy = -torch.logsumexp(logits_mean, dim=-1)

        maha = self.mahalanobis_score(features)
        maha_norm = torch.log1p(maha) / math.log1p(float(self.clip_latent_dim))

        ca = self.cfg.CLIP_ADAPTERS
        w_maxprob = float(getattr(ca, "BAYES_PLUS_OOD_W_MAXPROB", 0.35))
        w_entropy = float(getattr(ca, "BAYES_PLUS_OOD_W_ENTROPY", 0.20))
        w_maha = float(getattr(ca, "BAYES_PLUS_OOD_W_MAHA", 0.30))
        w_energy = float(getattr(ca, "BAYES_PLUS_OOD_W_ENERGY", 0.15))

        joint = (
            w_maxprob * (1.0 - max_prob)
            + w_entropy * entropy_norm
            + w_maha * maha_norm
            + w_energy * energy
        )

        return {
            "max_prob": max_prob,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "energy": energy,
            "mahalanobis": maha,
            "mahalanobis_norm": maha_norm,
            "ood_score": joint,
        }

    def reset_hparams(self, params: Dict) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)