from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


class HbaLrAdapter(BaseAdapter):
    """
    Bayesian Low-Rank Task Residual Adapter.

    This version matches the intended design:

        mu_W = T + alpha * A B
        q(W) = N(mu_W, Sigma_q)

    where:
        T      : frozen CLIP text prototypes, [C, D]
        A      : learnable class-wise low-rank coefficients, [C, R]
        B      : learnable shared low-rank basis, [R, D]
        alpha  : residual scale, implemented as HBA_RHO
        Sigma_q: class-wise Gaussian variance over full prototype W

    Important distinction:
        Low-rank TaskRes residual parameterizes only the posterior mean.
        The Bayesian sampling is over full prototypes W, not over A.

    This matches:

        W = t_0 + alpha * A B
        q(W) = N(W, Sigma)
        p(y|x) = E_q softmax(sim(z, W) / T)

    For strict TaskRes-style adaptation, keep image features frozen:
        HBA_USE_VISUAL_ADAPTER: False
    """

    initialization_name = "BAYES_ADAPTER"
    hba_initialization_name = "HBA_LR"
    adapter_kind = "stochastic_prototype"

    needs_support_features = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = cfg.CLIP_ADAPTERS
        n_classes, feat_dim = base_text_features.shape

        self.num_classes = int(n_classes)
        self.clip_latent_dim = int(feat_dim)

        # ------------------------------------------------------------------
        # Low-rank TaskRes posterior mean:
        #     mu_W = T + alpha * A B
        # ------------------------------------------------------------------
        rank = int(getattr(cad, "HBA_RANK", 16))
        self.rank = max(1, min(rank, self.clip_latent_dim))

        # HBA_RHO is alpha in:
        #     W = T + alpha * A B
        self.rho = float(getattr(cad, "HBA_RHO", 0.5))
        if self.rho <= 0:
            raise ValueError(f"HBA_RHO must be > 0, got {self.rho}")

        # Optional coefficient bound on A.
        # For pure TaskRes-LR mean, set HBA_MAX_COEFF_NORM <= 0.
        self.max_coeff_norm = float(getattr(cad, "HBA_MAX_COEFF_NORM", 0.0))

        default_beta = 1.0 / float(
            max(1, 1000 * self.num_classes * self.clip_latent_dim)
        )
        self.hba_beta = float(getattr(cad, "HBA_BETA", default_beta))

        self.lambda_b = float(getattr(cad, "HBA_LAMBDA_B", 0.0))
        self.lambda_orth = float(getattr(cad, "HBA_LAMBDA_ORTH", 1.0e-4))

        self.lambda_proto_anchor = float(
            getattr(cad, "HBA_LAMBDA_PROTO_ANCHOR", 0.0)
        )
        self.proto_anchor_type = str(
            getattr(cad, "HBA_PROTO_ANCHOR_TYPE", "cosine")
        ).lower()

        self.posterior_predictive_on_train = bool(
            getattr(cad, "HBA_POSTERIOR_PREDICTIVE_ON_TRAIN", False)
        )

        # ------------------------------------------------------------------
        # BayesAdapter-style Gaussian posterior over full prototypes W
        # ------------------------------------------------------------------
        prior_std = float(getattr(cad, "BAYES_PRIOR_STD", 0.01))
        if prior_std <= 0:
            raise ValueError(f"BAYES_PRIOR_STD must be > 0, got {prior_std}")

        posterior_init_std = float(getattr(cad, "HBA_POSTERIOR_INIT_STD", prior_std))
        # Backward compatibility with previous HBA config.
        posterior_init_std = float(getattr(cad, "HBA_S0", posterior_init_std))
        if posterior_init_std <= 0:
            raise ValueError(
                f"HBA_POSTERIOR_INIT_STD / HBA_S0 must be > 0, "
                f"got {posterior_init_std}"
            )

        prior_logstd = math.log(prior_std)
        posterior_init_logstd = math.log(posterior_init_std)

        # q(W): class-wise scalar std, shared over feature dimensions.
        # Shape: [C]
        self.text_features_unnorm_logstd = nn.Parameter(
            torch.full(
                (self.num_classes,),
                posterior_init_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        # p(W): zero-shot prototype prior
        self.register_buffer("prior_mean", base_text_features.detach().clone())
        self.register_buffer(
            "prior_logstd",
            torch.full(
                (self.num_classes,),
                prior_logstd,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            ),
        )

        # ------------------------------------------------------------------
        # Low-rank mean parameters A and B
        # ------------------------------------------------------------------
        # A: [C, R]
        self.hba_mean = nn.Parameter(
            torch.zeros(
                self.num_classes,
                self.rank,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        # B^T: [D, R]
        basis = torch.randn(
            self.clip_latent_dim,
            self.rank,
            device=base_text_features.device,
            dtype=torch.float32,
        )
        basis = self._orthonormalize_columns(basis, self.rank)
        self.hba_basis = nn.Parameter(basis.to(dtype=base_text_features.dtype))

        self.register_buffer(
            "hba_support_initialized",
            torch.tensor(False, device=base_text_features.device),
        )

        # ------------------------------------------------------------------
        # Optional temperature scaling
        # ------------------------------------------------------------------
        # To match:
        #     E_q softmax(logits(W) / T)
        # temperature is applied inside bayes_base_logits_from_mc()
        # before posterior predictive aggregation.
        self.temperature = float(getattr(cad, "HBA_TEMPERATURE", 1.0))
        self.temperature_on_train = bool(
            getattr(cad, "HBA_TEMPERATURE_ON_TRAIN", False)
        )
        if self.temperature <= 0:
            raise ValueError(f"HBA_TEMPERATURE must be > 0, got {self.temperature}")

        # ------------------------------------------------------------------
        # Support initialization controls
        # ------------------------------------------------------------------
        self.support_init = bool(getattr(cad, "HBA_SUPPORT_INIT", True))
        self.init_strength = float(getattr(cad, "HBA_INIT_STRENGTH", 1.0))
        self.init_max_norm = float(getattr(cad, "HBA_INIT_MAX_NORM", 0.0))
        self.init_residual = str(
            getattr(cad, "HBA_INIT_RESIDUAL", "euclidean")
        ).lower()

        # ------------------------------------------------------------------
        # Optional visual adapter
        # ------------------------------------------------------------------
        # For strict TaskRes/BayesAdapter-style prototype adaptation, keep off.
        self.use_visual_adapter = bool(
            getattr(cad, "HBA_USE_VISUAL_ADAPTER", False)
        )
        visual_rank = int(getattr(cad, "HBA_VISUAL_RANK", 32))
        self.visual_rank = max(1, min(visual_rank, self.clip_latent_dim))

        self.visual_scale = float(getattr(cad, "HBA_VISUAL_SCALE", 0.1))
        self.visual_dropout_p = float(getattr(cad, "HBA_VISUAL_DROPOUT", 0.1))
        self.visual_use_ln = bool(getattr(cad, "HBA_VISUAL_USE_LN", True))
        self.visual_activation = str(
            getattr(cad, "HBA_VISUAL_ACT", "gelu")
        ).lower()

        self.lambda_visual_anchor = float(
            getattr(cad, "HBA_LAMBDA_VISUAL_ANCHOR", 0.0)
        )
        self.visual_anchor_type = str(
            getattr(cad, "HBA_VISUAL_ANCHOR_TYPE", "cosine")
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

    @staticmethod
    def _orthonormalize_columns(x: torch.Tensor, rank: int) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [D, R], got {tuple(x.shape)}")

        d = int(x.shape[0])
        rank = int(rank)

        if x.shape[1] < rank:
            pad = torch.randn(
                d,
                rank - int(x.shape[1]),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)

        try:
            q, _ = torch.linalg.qr(x[:, :rank], mode="reduced")
            q = q[:, :rank]
        except RuntimeError:
            q = F.normalize(x[:, :rank], dim=0)

        return q.contiguous()

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.visual_activation == "relu":
            return F.relu(x)
        if self.visual_activation == "silu":
            return F.silu(x)
        return F.gelu(x)

    def _bound_coefficients(self, a: torch.Tensor) -> torch.Tensor:
        """
        Optional bound for A.

        For pure low-rank TaskRes mean:
            HBA_MAX_COEFF_NORM <= 0
        """
        max_norm = float(self.max_coeff_norm)
        if max_norm <= 0:
            return a

        norm = a.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        scale = (max_norm / norm).clamp_max(1.0)
        return a * scale

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
    # Posterior mean:
    #     mu_W = T + alpha * A B
    # ----------------------------------------------------------------------
    def _mean_prototypes(self) -> torch.Tensor:
        """
        Deterministic posterior mean prototypes.

        Returns:
            mu_W: [C, D]
        """
        a = self._bound_coefficients(self.hba_mean)

        text = self.base_text_features.to(
            device=a.device,
            dtype=a.dtype,
        )  # [C, D]

        basis = self.hba_basis.to(
            device=a.device,
            dtype=a.dtype,
        )  # [D, R]

        # A B: [C, D]
        residual = torch.einsum("cr,dr->cd", a, basis)

        mu = text + float(self.rho) * residual
        return mu.to(dtype=self.base_text_features.dtype)

    def get_prototypes(self) -> torch.Tensor:
        """
        Deterministic posterior mean prototypes [C, D].
        """
        return self._mean_prototypes()

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        """
        Sample full prototypes W from:

            q(W) = N(mu_W, Sigma_q)

        Returns:
            [S, C, D]
        """
        n_samples = max(1, int(n_samples))

        mu = self._mean_prototypes()  # [C, D]

        eps = torch.randn(
            n_samples,
            self.num_classes,
            self.clip_latent_dim,
            device=mu.device,
            dtype=mu.dtype,
        )

        std = torch.exp(self.text_features_unnorm_logstd).to(
            device=mu.device,
            dtype=mu.dtype,
        )  # [C]

        return mu.unsqueeze(0) + eps * std.view(1, self.num_classes, 1)

    # ----------------------------------------------------------------------
    # Support SVD initialization for posterior mean
    # ----------------------------------------------------------------------
    def _class_means_from_support(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
    ) -> torch.Tensor:
        device = self.base_text_features.device
        dtype = torch.float32

        features = features_train.detach().to(device=device, dtype=dtype)
        labels = labels_train.detach().to(device=device, dtype=torch.long)

        features = self._normalize(features)
        text = self._normalize(
            self.base_text_features.detach().to(device=device, dtype=dtype)
        )

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

        means = sums / counts.clamp_min(1.0)
        missing = counts.squeeze(-1) <= 0
        if missing.any():
            means[missing] = text[missing]

        return self._normalize(means)

    def _support_residuals(self, class_means: torch.Tensor) -> torch.Tensor:
        """
        TaskRes-style residual initialization:

            residual_c = class_mean_c - text_c
        """
        text = self.base_text_features.detach().float().to(
            device=class_means.device,
            dtype=torch.float32,
        )
        means = class_means.detach().float().to(
            device=class_means.device,
            dtype=torch.float32,
        )

        text = self._normalize(text)
        means = self._normalize(means)

        return means - text

    @torch.no_grad()
    def build_cache(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
    ) -> None:
        """
        Optional support-set SVD initialization.

        If enabled:
            residual = class_mean - text
            residual ~= A B

        Since:
            mu_W = T + rho * A B

        initialize:
            A ~= residual @ B.T / rho
        """
        if not self.support_init:
            return None

        if features_train is None or labels_train is None:
            return None

        device = self.base_text_features.device

        class_means = self._class_means_from_support(
            features_train=features_train,
            labels_train=labels_train,
        )
        residual = self._support_residuals(class_means).to(device=device)

        residual_norm = residual.norm(dim=-1, keepdim=True)
        residual = torch.where(
            residual_norm > 1.0e-8,
            residual,
            torch.zeros_like(residual),
        )

        try:
            _, _, vh = torch.linalg.svd(residual.float(), full_matrices=False)
            k = min(self.rank, int(vh.shape[0]))
            top_basis = vh[:k].t().contiguous()
        except RuntimeError:
            k = 0
            top_basis = residual.new_zeros(self.clip_latent_dim, 0)

        if k < self.rank:
            random_pad = torch.randn(
                self.clip_latent_dim,
                self.rank - k,
                device=device,
                dtype=torch.float32,
            )
            basis_init = torch.cat([top_basis.float(), random_pad], dim=1)
        else:
            basis_init = top_basis.float()

        basis_init = self._orthonormalize_columns(basis_init, self.rank)

        coeff = residual.float() @ basis_init.float()
        coeff = coeff / max(float(self.rho), 1.0e-8)
        coeff = coeff * float(self.init_strength)

        coeff_norm = coeff.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)

        max_init_norm = float(self.init_max_norm)
        if self.max_coeff_norm > 0:
            if max_init_norm > 0:
                max_init_norm = min(max_init_norm, float(self.max_coeff_norm))
            else:
                max_init_norm = float(self.max_coeff_norm)

        if max_init_norm > 0:
            coeff = coeff * (max_init_norm / coeff_norm).clamp_max(1.0)

        self.hba_basis.data.copy_(
            basis_init.to(device=device, dtype=self.hba_basis.dtype)
        )
        self.hba_mean.data.copy_(
            coeff.to(device=device, dtype=self.hba_mean.dtype)
        )

        # Keep posterior std initialized independently from support residuals.
        posterior_init_std = float(
            getattr(
                self.cfg.CLIP_ADAPTERS,
                "HBA_POSTERIOR_INIT_STD",
                math.exp(float(self.prior_logstd[0].detach().cpu())),
            )
        )
        posterior_init_std = float(
            getattr(self.cfg.CLIP_ADAPTERS, "HBA_S0", posterior_init_std)
        )
        self.text_features_unnorm_logstd.data.fill_(math.log(posterior_init_std))

        self.hba_support_initialized.fill_(True)

        eye = torch.eye(self.rank, device=device)
        basis_orth = (
            (self.hba_basis.detach().float().t() @ self.hba_basis.detach().float())
            - eye
        ).pow(2).sum()

        print(
            "[Bayesian Low-Rank TaskRes] support SVD init done: "
            f"rank={self.rank}, "
            f"alpha/rho={self.rho}, "
            f"max_coeff_norm={self.max_coeff_norm}, "
            f"init_strength={self.init_strength}, "
            f"mean_norm={float(self.hba_mean.detach().float().norm(dim=-1).mean().cpu()):.4f}, "
            f"basis_orth={float(basis_orth.cpu()):.4e}, "
            f"posterior_std={float(torch.exp(self.text_features_unnorm_logstd.detach()).mean().cpu()):.6f}, "
            f"visual_adapter={self.use_visual_adapter}, "
            f"temperature={self.temperature}"
        )

        return None

    # ----------------------------------------------------------------------
    # KL and regularization
    # ----------------------------------------------------------------------
    def kl_divergence(self) -> torch.Tensor:
        """
        KL(q(W) || p(W)).

        q(W):
            mean = T + alpha * A B
            std  = learned class-wise scalar std, [C]

        p(W):
            mean = T
            std  = BAYES_PRIOR_STD, class-wise scalar std, [C]
        """
        posterior_mean = self._mean_prototypes().float()
        prior_mean = self.prior_mean.float().to(posterior_mean.device)

        posterior_logstd = self.text_features_unnorm_logstd.float()
        prior_logstd = self.prior_logstd.float().to(posterior_logstd.device)

        posterior_std = torch.exp(posterior_logstd).clamp_min(1.0e-12)
        prior_std = torch.exp(prior_logstd).clamp_min(1.0e-12)

        posterior_var = posterior_std.pow(2)
        prior_var = prior_std.pow(2)

        # trace term: sum over C and D
        trace = self.clip_latent_dim * (posterior_var / prior_var).sum()

        # quadratic mean-difference term
        diff = posterior_mean - prior_mean
        diff_term = (diff.pow(2) / prior_var.unsqueeze(-1)).sum()

        # log det term
        logdet = 2.0 * self.clip_latent_dim * (
            prior_logstd - posterior_logstd
        ).sum()

        dim = float(self.num_classes * self.clip_latent_dim)

        kl = 0.5 * (trace + diff_term - dim + logdet)
        return kl.to(dtype=self.hba_mean.dtype)

    def basis_regularization(self) -> torch.Tensor:
        basis = self.hba_basis.float()
        reg = basis.new_tensor(0.0)

        if self.lambda_b > 0:
            reg = reg + float(self.lambda_b) * basis.pow(2).sum()

        if self.lambda_orth > 0:
            gram = basis.t() @ basis
            eye = torch.eye(
                gram.shape[0],
                device=gram.device,
                dtype=gram.dtype,
            )
            reg = reg + float(self.lambda_orth) * (gram - eye).pow(2).sum()

        return reg.to(dtype=self.hba_basis.dtype)

    def prototype_anchor_regularization(self) -> torch.Tensor:
        """
        Optional anchor regularization.

        Usually redundant because KL already anchors q(W) to p(W).
        For pure BayesAdapter-style prior anchoring, set:
            HBA_LAMBDA_PROTO_ANCHOR: 0.0
        """
        text = self._normalize(self.base_text_features.float())
        proto = self._normalize(self.get_prototypes().float())

        cos = (proto * text).sum(dim=-1).clamp(
            -1.0 + 1.0e-6,
            1.0 - 1.0e-6,
        )

        if self.proto_anchor_type == "geodesic":
            reg = torch.acos(cos).pow(2).mean()
        else:
            reg = (1.0 - cos).mean()

        return reg.to(dtype=self.hba_mean.dtype)

    def visual_anchor_regularization(
        self,
        raw_features: torch.Tensor,
        adapted_features: torch.Tensor,
    ) -> torch.Tensor:
        raw = self._normalize(raw_features.float())
        adapted = self._normalize(adapted_features.float())

        cos = (raw * adapted).sum(dim=-1).clamp(
            -1.0 + 1.0e-6,
            1.0 - 1.0e-6,
        )

        if self.visual_anchor_type == "geodesic":
            reg = torch.acos(cos).pow(2).mean()
        else:
            reg = (1.0 - cos).mean()

        return reg.to(dtype=adapted_features.dtype)

    def extra_loss(self) -> torch.Tensor:
        return self.basis_regularization()

    def bayes_kl_base_weight(self) -> float:
        return float(self.hba_beta)

    # ----------------------------------------------------------------------
    # Temperature and MC aggregation
    # ----------------------------------------------------------------------
    def _maybe_temperature_scale_logits_all(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Apply temperature before posterior predictive aggregation.

        This matches:
            E_q softmax(logits(W) / T)

        rather than:
            softmax(E_q logits(W) / T)
        """
        if training and not self.temperature_on_train:
            return logits_all

        if abs(float(self.temperature) - 1.0) <= 1.0e-12:
            return logits_all

        return logits_all / float(self.temperature)

    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
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
        out = torch.logsumexp(log_probs, dim=0) - math.log(
            float(logits_all.shape[0])
        )
        return out.to(dtype=logits_all.dtype)

    def postprocess_logits(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        No-op.

        Temperature is applied before MC posterior predictive aggregation in
        bayes_base_logits_from_mc(), because that matches:

            E_q softmax(logits(W) / T)

        This method is kept to avoid accidental double temperature scaling
        through model.py's postprocess hook.
        """
        return logits

    # ----------------------------------------------------------------------
    # Uncertainty helper
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def hba_scores(
        self,
        features: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> dict:
        if n_samples is None:
            n_samples = int(
                getattr(self.cfg.CLIP_ADAPTERS, "N_TEST_SAMPLES", 50)
            )
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