from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


class BayesianTaskResidualAdapter(BaseAdapter):
    """
    Bayesian Task Residual Adapter.

    Core idea:
        TaskRes learns a deterministic full residual:
            W = T + alpha * R

        This adapter Bayesianizes the task residual:
            q(R) = N(mu_R, sigma_c^2 I)

        Therefore:
            W = T + alpha * g_c * R

        where:
            T       : frozen CLIP text prototypes, [C, D]
            R       : full task residual, [C, D]
            alpha   : TaskRes residual scale
            g_c     : optional evidence gate per class
            sigma_c : class-wise residual uncertainty

    This keeps the full-rank adaptation capacity of TaskRes and reuses
    BayesAdapter-style stochastic prototype inference.

    Recommended init:
        CLIP_ADAPTERS.INIT: BTR
    """

    initialization_name = "BAYES_ADAPTER"
    btr_initialization_name = "BTR"
    adapter_kind = "stochastic_prototype"

    # This is overwritten in __init__ depending on support init / evidence gate.
    needs_support_features = False

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = cfg.CLIP_ADAPTERS
        n_classes, feat_dim = base_text_features.shape

        self.num_classes = int(n_classes)
        self.clip_latent_dim = int(feat_dim)

        # ------------------------------------------------------------------
        # TaskRes residual scale
        # ------------------------------------------------------------------
        self.alpha = float(getattr(cad, "BTR_ALPHA", 0.5))
        if self.alpha <= 0:
            raise ValueError(f"BTR_ALPHA must be > 0, got {self.alpha}")

        # ------------------------------------------------------------------
        # Bayesian residual posterior q(R)
        # ------------------------------------------------------------------
        prior_std = float(getattr(cad, "BTR_PRIOR_STD", 0.01))
        if prior_std <= 0:
            raise ValueError(f"BTR_PRIOR_STD must be > 0, got {prior_std}")

        init_std = float(getattr(cad, "BTR_INIT_STD", prior_std))
        if init_std <= 0:
            raise ValueError(f"BTR_INIT_STD must be > 0, got {init_std}")

        self.prior_logstd_value = math.log(prior_std)
        init_logstd = math.log(init_std)

        # q(R): residual mean [C, D]
        # Init at zero so initial classifier is exactly zero-shot CLIP:
        #     W = T + alpha * 0
        self.residual_mean = nn.Parameter(torch.zeros_like(base_text_features))

        # q(R): class-wise scalar std [C]
        self.residual_logstd = nn.Parameter(
            torch.full(
                (self.num_classes,),
                init_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        self.register_buffer(
            "residual_prior_logstd",
            torch.full(
                (self.num_classes,),
                self.prior_logstd_value,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

        # ------------------------------------------------------------------
        # KL weight
        # ------------------------------------------------------------------
        default_beta = 1.0 / float(
            max(1, 1000 * self.num_classes * self.clip_latent_dim)
        )
        self.btr_beta = float(getattr(cad, "BTR_BETA", default_beta))

        # ------------------------------------------------------------------
        # Data term
        # ------------------------------------------------------------------
        # Options:
        #   "pp_nll": posterior predictive NLL, better aligned with calibration
        #   "mc_ce" : sample-wise MC CE, closer to existing BayesAdapter code
        self.data_term = str(getattr(cad, "BTR_DATA_TERM", "pp_nll")).lower()

        # Optional Brier regularization.
        self.brier_weight = float(getattr(cad, "BTR_BRIER_WEIGHT", 0.0))

        # ------------------------------------------------------------------
        # Temperature
        # ------------------------------------------------------------------
        # Applied before posterior predictive aggregation:
        #     E_q softmax(logits(W) / T)
        self.temperature = float(getattr(cad, "BTR_TEMPERATURE", 1.0))
        self.temperature_on_train = bool(
            getattr(cad, "BTR_TEMPERATURE_ON_TRAIN", False)
        )

        if self.temperature <= 0:
            raise ValueError(f"BTR_TEMPERATURE must be > 0, got {self.temperature}")

        # ------------------------------------------------------------------
        # Posterior predictive behavior
        # ------------------------------------------------------------------
        self.posterior_predictive_on_train = bool(
            getattr(cad, "BTR_POSTERIOR_PREDICTIVE_ON_TRAIN", False)
        )

        # ------------------------------------------------------------------
        # Evidence gate
        # ------------------------------------------------------------------
        self.use_evidence_gate = bool(getattr(cad, "BTR_USE_EVIDENCE_GATE", True))
        self.gate_lambda = float(getattr(cad, "BTR_GATE_LAMBDA", 2.0))
        self.gate_min = float(getattr(cad, "BTR_GATE_MIN", 0.0))
        self.gate_max = float(getattr(cad, "BTR_GATE_MAX", 1.0))

        if self.gate_lambda < 0:
            raise ValueError(f"BTR_GATE_LAMBDA must be >= 0, got {self.gate_lambda}")

        self.register_buffer(
            "residual_gate",
            torch.ones(
                self.num_classes,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

        self.register_buffer(
            "support_counts",
            torch.zeros(
                self.num_classes,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

        # ------------------------------------------------------------------
        # Support initialization
        # ------------------------------------------------------------------
        self.support_init = bool(getattr(cad, "BTR_SUPPORT_INIT", True))
        self.init_strength = float(getattr(cad, "BTR_INIT_STRENGTH", 1.0))
        self.init_max_norm = float(getattr(cad, "BTR_INIT_MAX_NORM", 0.0))

        self.needs_support_features = bool(self.use_evidence_gate or self.support_init)

        self.register_buffer(
            "btr_support_initialized",
            torch.tensor(False, device=base_text_features.device),
        )

        # ------------------------------------------------------------------
        # Optional visual adapter
        # ------------------------------------------------------------------
        # Strict TaskRes/BayesAdapter residual adaptation should keep this off.
        self.use_visual_adapter = bool(
            getattr(cad, "BTR_USE_VISUAL_ADAPTER", False)
        )
        self.visual_rank = int(getattr(cad, "BTR_VISUAL_RANK", 32))
        self.visual_rank = max(1, min(self.visual_rank, self.clip_latent_dim))

        self.visual_scale = float(getattr(cad, "BTR_VISUAL_SCALE", 0.1))
        self.visual_dropout_p = float(getattr(cad, "BTR_VISUAL_DROPOUT", 0.1))
        self.visual_use_ln = bool(getattr(cad, "BTR_VISUAL_USE_LN", True))
        self.visual_activation = str(
            getattr(cad, "BTR_VISUAL_ACT", "gelu")
        ).lower()

        if self.use_visual_adapter:
            self.visual_ln = (
                nn.LayerNorm(self.clip_latent_dim)
                if self.visual_use_ln
                else nn.Identity()
            )
            self.visual_down = nn.Linear(
                self.clip_latent_dim,
                self.visual_rank,
                bias=False,
            )
            self.visual_up = nn.Linear(
                self.visual_rank,
                self.clip_latent_dim,
                bias=False,
            )
            self.visual_dropout = nn.Dropout(p=self.visual_dropout_p)

            nn.init.normal_(self.visual_down.weight, std=0.02)
            nn.init.zeros_(self.visual_up.weight)
        else:
            self.visual_ln = nn.Identity()
            self.visual_down = None
            self.visual_up = None
            self.visual_dropout = nn.Identity()

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.visual_activation == "relu":
            return F.relu(x)
        if self.visual_activation == "silu":
            return F.silu(x)
        return F.gelu(x)

    # ----------------------------------------------------------------------
    # Optional visual adapter
    # ----------------------------------------------------------------------
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.use_visual_adapter:
            return features

        dtype_out = features.dtype
        v = self._normalize(features.float())

        h = self.visual_ln(v)
        h = self.visual_down(h)
        h = self._activation(h)
        h = self.visual_dropout(h)
        delta = self.visual_up(h)

        # Tangent projection at image feature v.
        delta = delta - (delta * v).sum(dim=-1, keepdim=True) * v

        v_adapt = self._normalize(v + float(self.visual_scale) * delta)
        return v_adapt.to(dtype=dtype_out)

    # ----------------------------------------------------------------------
    # Evidence gate and support initialization
    # ----------------------------------------------------------------------
    def _counts_to_gate(self, counts: torch.Tensor) -> torch.Tensor:
        if not self.use_evidence_gate:
            return torch.ones_like(counts)

        if self.gate_lambda <= 0:
            gate = torch.ones_like(counts)
        else:
            gate = counts / (counts + float(self.gate_lambda))

        gate = gate.clamp(float(self.gate_min), float(self.gate_max))
        return gate

    def _class_means_from_support(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.base_text_features.device
        dtype = torch.float32

        features = features_train.detach().to(device=device, dtype=dtype)
        labels = labels_train.detach().to(device=device, dtype=torch.long)

        features = self._normalize(features)

        sums = torch.zeros(
            self.num_classes,
            self.clip_latent_dim,
            device=device,
            dtype=dtype,
        )
        counts = torch.zeros(
            self.num_classes,
            1,
            device=device,
            dtype=dtype,
        )

        valid = (labels >= 0) & (labels < self.num_classes)
        if valid.any():
            sums.index_add_(0, labels[valid], features[valid])
            ones = torch.ones(
                int(valid.sum().item()),
                1,
                device=device,
                dtype=dtype,
            )
            counts.index_add_(0, labels[valid], ones)

        text = self._normalize(
            self.base_text_features.detach().to(device=device, dtype=dtype)
        )

        means = sums / counts.clamp_min(1.0)
        missing = counts.squeeze(-1) <= 0
        if missing.any():
            means[missing] = text[missing]

        means = self._normalize(means)
        return means, counts.squeeze(-1)

    @torch.no_grad()
    def build_cache(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
    ) -> None:
        """
        Build support-dependent residual initialization and evidence gates.

        residual target:
            class_mean - text

        Since:
            W = text + alpha * gate * R

        Conservative initialization:
            R_init = (class_mean - text) / alpha

        The gate still controls how much residual is released.
        """
        if features_train is None or labels_train is None:
            return None

        class_means, counts = self._class_means_from_support(
            features_train=features_train,
            labels_train=labels_train,
        )

        device = self.base_text_features.device
        dtype = self.base_text_features.dtype

        counts = counts.to(device=device, dtype=dtype)
        gate = self._counts_to_gate(counts)
        self.support_counts.data.copy_(counts)
        self.residual_gate.data.copy_(gate)

        if self.support_init:
            text = self._normalize(
                self.base_text_features.detach().to(device=device, dtype=torch.float32)
            )
            residual = class_means.to(device=device, dtype=torch.float32) - text

            residual = residual / max(float(self.alpha), 1.0e-8)
            residual = residual * float(self.init_strength)

            max_norm = float(self.init_max_norm)
            if max_norm > 0:
                norm = residual.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
                residual = residual * (max_norm / norm).clamp_max(1.0)

            self.residual_mean.data.copy_(
                residual.to(device=device, dtype=self.residual_mean.dtype)
            )

        self.btr_support_initialized.fill_(True)

        print(
            "[BayesianTaskResidualAdapter] support init done: "
            f"alpha={self.alpha}, "
            f"support_init={self.support_init}, "
            f"use_gate={self.use_evidence_gate}, "
            f"gate_mean={float(self.residual_gate.detach().float().mean().cpu()):.4f}, "
            f"gate_min={float(self.residual_gate.detach().float().min().cpu()):.4f}, "
            f"gate_max={float(self.residual_gate.detach().float().max().cpu()):.4f}, "
            f"residual_norm={float(self.residual_mean.detach().float().norm(dim=-1).mean().cpu()):.4f}, "
            f"std={float(torch.exp(self.residual_logstd.detach()).mean().cpu()):.6f}"
        )

        return None

    # ----------------------------------------------------------------------
    # Prototypes
    # ----------------------------------------------------------------------
    def _gate(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.use_evidence_gate:
            return self.residual_gate.to(device=device, dtype=dtype)
        return torch.ones(
            self.num_classes,
            device=device,
            dtype=dtype,
        )

    def _mean_prototypes(self) -> torch.Tensor:
        """
        Deterministic posterior mean prototypes:
            E[W] = T + alpha * g_c * mu_R
        """
        text = self.base_text_features.to(
            device=self.residual_mean.device,
            dtype=self.residual_mean.dtype,
        )
        gate = self._gate(
            dtype=self.residual_mean.dtype,
            device=self.residual_mean.device,
        ).view(self.num_classes, 1)

        mu = text + float(self.alpha) * gate * self.residual_mean
        return mu.to(dtype=self.base_text_features.dtype)

    def get_prototypes(self) -> torch.Tensor:
        return self._mean_prototypes()

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample stochastic prototypes through residual posterior:

            R ~ N(mu_R, sigma_c^2 I)
            W = T + alpha * g_c * R

        Returns:
            [S, C, D]
        """
        n_samples = max(1, int(n_samples))

        dtype = self.residual_mean.dtype
        device = self.residual_mean.device

        eps = torch.randn(
            n_samples,
            self.num_classes,
            self.clip_latent_dim,
            device=device,
            dtype=dtype,
        )

        std = torch.exp(self.residual_logstd).to(
            device=device,
            dtype=dtype,
        ).view(1, self.num_classes, 1)

        residual = self.residual_mean.unsqueeze(0) + eps * std

        text = self.base_text_features.to(device=device, dtype=dtype).unsqueeze(0)
        gate = self._gate(dtype=dtype, device=device).view(1, self.num_classes, 1)

        prototypes = text + float(self.alpha) * gate * residual
        return prototypes.to(dtype=self.base_text_features.dtype)

    # ----------------------------------------------------------------------
    # KL and regularization
    # ----------------------------------------------------------------------
    def kl_divergence(self) -> torch.Tensor:
        """
        KL(q(R) || p(R)).

        q(R):
            mean = residual_mean
            std  = residual_logstd, class-wise scalar

        p(R):
            mean = 0
            std  = BTR_PRIOR_STD
        """
        mean = self.residual_mean.float()

        posterior_logstd = self.residual_logstd.float()
        prior_logstd = self.residual_prior_logstd.float().to(posterior_logstd.device)

        posterior_std = torch.exp(posterior_logstd).clamp_min(1.0e-12)
        prior_std = torch.exp(prior_logstd).clamp_min(1.0e-12)

        posterior_var = posterior_std.pow(2)
        prior_var = prior_std.pow(2)

        trace = self.clip_latent_dim * (posterior_var / prior_var).sum()

        diff_term = (mean.pow(2) / prior_var.unsqueeze(-1)).sum()

        logdet = 2.0 * self.clip_latent_dim * (
            prior_logstd - posterior_logstd
        ).sum()

        dim = float(self.num_classes * self.clip_latent_dim)

        kl = 0.5 * (trace + diff_term - dim + logdet)
        return kl.to(dtype=self.residual_mean.dtype)

    def bayes_kl_base_weight(self) -> float:
        return float(self.btr_beta)

    # ----------------------------------------------------------------------
    # Posterior predictive utilities
    # ----------------------------------------------------------------------
    def _maybe_temperature_scale_logits_all(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        if training and not self.temperature_on_train:
            return logits_all

        if abs(float(self.temperature) - 1.0) <= 1.0e-12:
            return logits_all

        return logits_all / float(self.temperature)

    def posterior_predictive_log_probs(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        log E_q[softmax(logits(W) / T)]
        """
        logits_all = self._maybe_temperature_scale_logits_all(
            logits_all,
            training=training,
        )

        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        out = torch.logsumexp(log_probs, dim=0) - math.log(float(x.shape[0]))
        return out.to(dtype=logits_all.dtype)

    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Used by model.py.

        Training default:
            mean logits, matching existing BayesAdapter optimization path.

        Evaluation default:
            posterior predictive log-probabilities.
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        logits_all = self._maybe_temperature_scale_logits_all(
            logits_all,
            training=training,
        )

        if training and not self.posterior_predictive_on_train:
            return logits_all.mean(dim=0)

        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        out = torch.logsumexp(log_probs, dim=0) - math.log(float(x.shape[0]))
        return out.to(dtype=logits_all.dtype)

    def postprocess_logits(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        No-op.

        Temperature is applied before MC posterior predictive aggregation.
        This avoids double temperature scaling through model.py's postprocess hook.
        """
        return logits

    # ----------------------------------------------------------------------
    # Loss helpers
    # ----------------------------------------------------------------------
    def btr_data_term(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns data term and logging extras.
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"btr_data_term expects logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        mode = self.data_term

        if mode in {"mc_ce", "sample_ce", "samplewise_ce"}:
            s, b, c = logits_all.shape
            flat_logits = logits_all.reshape(s * b, c)
            flat_labels = labels.unsqueeze(0).expand(s, -1).reshape(-1)
            data_term = F.cross_entropy(flat_logits, flat_labels)
            extras = {"btr_data_mode": logits_all.new_tensor(0.0)}
            return data_term, extras

        # Default: posterior predictive NLL.
        log_probs = self.posterior_predictive_log_probs(
            logits_all,
            training=True,
        )
        data_term = F.nll_loss(log_probs, labels)
        extras = {"btr_data_mode": logits_all.new_tensor(1.0)}
        return data_term, extras

    def brier_loss(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.posterior_predictive_log_probs(
            logits_all,
            training=True,
        )
        probs = log_probs.exp()
        one_hot = F.one_hot(labels.to(torch.long), num_classes=probs.shape[-1]).to(
            dtype=probs.dtype,
            device=probs.device,
        )
        return (probs - one_hot).pow(2).sum(dim=-1).mean()

    # ----------------------------------------------------------------------
    # Uncertainty helper
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def btr_scores(
        self,
        features: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> dict:
        if n_samples is None:
            n_samples = int(getattr(self.cfg.CLIP_ADAPTERS, "N_TEST_SAMPLES", 50))
        n_samples = max(1, int(n_samples))

        x = self.adapt_features(features)
        x = self._normalize(x.float())

        p = self.sample_prototypes(n_samples=n_samples).detach().float().to(x.device)
        p = self._normalize(p)

        scale = self.logit_scale.exp().detach().float().to(x.device)
        logits_all = torch.einsum("bd,scd->sbc", x, p) * scale
        logits_all = self._maybe_temperature_scale_logits_all(
            logits_all,
            training=False,
        )

        probs_all = torch.softmax(logits_all, dim=-1)
        probs_mean = probs_all.mean(dim=0).clamp_min(1.0e-12)

        predictive_entropy = -(probs_mean * probs_mean.log()).sum(dim=-1)

        probs_all_safe = probs_all.clamp_min(1.0e-12)
        expected_entropy = -(
            probs_all_safe * probs_all_safe.log()
        ).sum(dim=-1).mean(dim=0)

        mutual_info = predictive_entropy - expected_entropy

        return {
            "probs_mean": probs_mean,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_info": mutual_info,
            "max_probability": probs_mean.max(dim=-1).values,
            "logits_all": logits_all,
        }

    # Compatibility alias.
    hba_scores = btr_scores