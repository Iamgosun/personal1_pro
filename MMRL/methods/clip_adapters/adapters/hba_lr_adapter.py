from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


class HbaLrAdapter(BaseAdapter):
    """
    Bounded HBA-VLR.

    Core principle:
        Adaptation should stay close to the original CLIP anchors.

    Prototype side:
        q(a_c) = N(m_c, diag(s_c^2))
        Delta_c = P_{t_c} B a_c
        mu_c = norm(t_c + rho * Delta_c)

    Visual side:
        v_bar = norm(v + eta * P_v g_phi(v))

    Added geometric safeguards:
        1. coefficient norm bound on a_c
        2. prototype anchor regularization: mu_c close to t_c
        3. visual anchor regularization: adapted image feature close to raw image feature
        4. visual adapter zero-init, so it starts as identity
    """

    # Reuse existing BayesAdapter stochastic prototype path.
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
        # Prototype-side HBA
        # ------------------------------------------------------------------
        rank = int(getattr(cad, "HBA_RANK", 16))
        self.rank = max(1, min(rank, self.clip_latent_dim))

        # Keep prototype movement conservative by default.
        self.rho = float(getattr(cad, "HBA_RHO", 0.25))
        self.max_coeff_norm = float(getattr(cad, "HBA_MAX_COEFF_NORM", 1.0))

        self.s_min = float(getattr(cad, "HBA_S_MIN", 1.0e-4))
        self.s0 = float(getattr(cad, "HBA_S0", 0.01))

        if self.rho <= 0:
            raise ValueError(f"HBA_RHO must be > 0, got {self.rho}")
        if self.s_min <= 0:
            raise ValueError(f"HBA_S_MIN must be > 0, got {self.s_min}")
        if self.s0 <= self.s_min:
            raise ValueError(
                f"HBA_S0 must be > HBA_S_MIN, got HBA_S0={self.s0}, "
                f"HBA_S_MIN={self.s_min}"
            )

        default_beta = 1.0 / float(max(1, 1000 * self.num_classes * self.rank))
        self.hba_beta = float(getattr(cad, "HBA_BETA", default_beta))

        self.lambda_b = float(getattr(cad, "HBA_LAMBDA_B", 0.0))
        self.lambda_orth = float(getattr(cad, "HBA_LAMBDA_ORTH", 1.0e-4))

        self.lambda_proto_anchor = float(getattr(cad, "HBA_LAMBDA_PROTO_ANCHOR", 1.0))
        self.proto_anchor_type = str(
            getattr(cad, "HBA_PROTO_ANCHOR_TYPE", "cosine")
        ).lower()

        self.posterior_predictive_on_train = bool(
            getattr(cad, "HBA_POSTERIOR_PREDICTIVE_ON_TRAIN", False)
        )

        # Support initialization controls.
        self.support_init = bool(getattr(cad, "HBA_SUPPORT_INIT", True))
        self.init_strength = float(getattr(cad, "HBA_INIT_STRENGTH", 0.25))
        self.init_max_norm = float(getattr(cad, "HBA_INIT_MAX_NORM", 1.0))
        self.init_residual = str(getattr(cad, "HBA_INIT_RESIDUAL", "logmap")).lower()

        self.hba_mean = nn.Parameter(
            torch.zeros(
                self.num_classes,
                self.rank,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        init_raw_scale = self._inverse_softplus(
            torch.tensor(
                self.s0 - self.s_min,
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

        self.hba_raw_scale = nn.Parameter(
            torch.full(
                (self.num_classes, self.rank),
                float(init_raw_scale.detach().cpu().item()),
                device=base_text_features.device,
                dtype=base_text_features.dtype,
            )
        )

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
        # Visual tangent adapter
        # ------------------------------------------------------------------
        self.use_visual_adapter = bool(getattr(cad, "HBA_USE_VISUAL_ADAPTER", True))
        visual_rank = int(getattr(cad, "HBA_VISUAL_RANK", 32))
        self.visual_rank = max(1, min(visual_rank, self.clip_latent_dim))

        # Small by default: do not break CLIP alignment.
        self.visual_scale = float(getattr(cad, "HBA_VISUAL_SCALE", 0.1))
        self.visual_dropout_p = float(getattr(cad, "HBA_VISUAL_DROPOUT", 0.1))
        self.visual_use_ln = bool(getattr(cad, "HBA_VISUAL_USE_LN", True))
        self.visual_activation = str(getattr(cad, "HBA_VISUAL_ACT", "gelu")).lower()

        self.lambda_visual_anchor = float(getattr(cad, "HBA_LAMBDA_VISUAL_ANCHOR", 1.0))
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

            # Critical: identity start.
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
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1.0e-12)
        return torch.log(torch.expm1(x))

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

    def _std(self) -> torch.Tensor:
        return F.softplus(self.hba_raw_scale) + self.s_min

    def _bound_coefficients(self, a: torch.Tensor) -> torch.Tensor:
        """
        Hard bound for a_c.

        This directly implements:
            adapted prototype should not rotate too far from text prototype.

        If max_coeff_norm <= 0, no bound is applied.
        """
        max_norm = float(self.max_coeff_norm)
        if max_norm <= 0:
            return a

        norm = a.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        scale = (max_norm / norm).clamp_max(1.0)
        return a * scale

    # ----------------------------------------------------------------------
    # Visual tangent adapter
    # ----------------------------------------------------------------------
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        v_bar = norm(v + eta * P_v g_phi(v))

        Both v and v_bar remain on the unit sphere.
        """
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
    # Prototype posterior
    # ----------------------------------------------------------------------
    def _sample_a(self, n_samples: int) -> torch.Tensor:
        eps = torch.randn(
            int(n_samples),
            self.num_classes,
            self.rank,
            device=self.hba_mean.device,
            dtype=self.hba_mean.dtype,
        )
        a = self.hba_mean.unsqueeze(0) + self._std().unsqueeze(0) * eps
        return self._bound_coefficients(a)

    def _prototypes_from_a(self, a: torch.Tensor) -> torch.Tensor:
        if a.ndim != 3:
            raise ValueError(f"Expected a [S, C, R], got {tuple(a.shape)}")

        a = self._bound_coefficients(a)

        text = self._normalize(self.base_text_features.float()).to(
            device=a.device,
            dtype=a.dtype,
        )
        basis = self.hba_basis.to(device=a.device, dtype=a.dtype)

        ba = torch.einsum("dr,scr->scd", basis, a)

        # P_t u = u - t(t^T u)
        dot = (ba * text.unsqueeze(0)).sum(dim=-1, keepdim=True)
        tangent = ba - dot * text.unsqueeze(0)

        proto = text.unsqueeze(0) + float(self.rho) * tangent
        proto = self._normalize(proto)

        return proto.to(dtype=self.base_text_features.dtype)

    def get_prototypes(self) -> torch.Tensor:
        a = self._bound_coefficients(self.hba_mean).unsqueeze(0)
        return self._prototypes_from_a(a).squeeze(0)

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        n_samples = max(1, int(n_samples))
        a = self._sample_a(n_samples)
        return self._prototypes_from_a(a)

    # ----------------------------------------------------------------------
    # Support SVD initialization
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
        text = self._normalize(self.base_text_features.detach().to(device=device, dtype=dtype))

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

    def _support_tangent_residuals(self, class_means: torch.Tensor) -> torch.Tensor:
        text = self._normalize(self.base_text_features.detach().float()).to(
            device=class_means.device,
            dtype=torch.float32,
        )
        u = self._normalize(class_means.float())

        cos = (u * text).sum(dim=-1, keepdim=True).clamp(-0.99, 0.99)
        tangent = u - cos * text
        tangent = tangent - (tangent * text).sum(dim=-1, keepdim=True) * text

        tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)

        if self.init_residual == "linear":
            residual = tangent
        elif self.init_residual == "tan":
            denom = cos.clamp_min(0.15)
            residual = tangent / denom
        else:
            theta = torch.acos(cos)
            residual = tangent * (theta / tangent_norm)

        residual = residual - (residual * text).sum(dim=-1, keepdim=True) * text
        return residual

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        if not self.support_init:
            return None

        if features_train is None or labels_train is None:
            return None

        device = self.base_text_features.device

        class_means = self._class_means_from_support(
            features_train=features_train,
            labels_train=labels_train,
        )
        residual = self._support_tangent_residuals(class_means).to(device=device)

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

        # Respect both init bound and global coefficient bound.
        coeff_norm = coeff.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)

        max_init_norm = float(self.init_max_norm)
        if self.max_coeff_norm > 0:
            max_init_norm = min(max_init_norm, float(self.max_coeff_norm))

        if max_init_norm > 0:
            coeff = coeff * (max_init_norm / coeff_norm).clamp_max(1.0)

        self.hba_basis.data.copy_(basis_init.to(device=device, dtype=self.hba_basis.dtype))
        self.hba_mean.data.copy_(coeff.to(device=device, dtype=self.hba_mean.dtype))

        init_raw_scale = self._inverse_softplus(
            torch.tensor(
                self.s0 - self.s_min,
                device=device,
                dtype=self.hba_raw_scale.dtype,
            )
        )
        self.hba_raw_scale.data.fill_(float(init_raw_scale.detach().cpu().item()))

        self.hba_support_initialized.fill_(True)

        eye = torch.eye(self.rank, device=device)
        basis_orth = (
            (self.hba_basis.detach().float().t() @ self.hba_basis.detach().float()) - eye
        ).pow(2).sum()

        print(
            "[Bounded HBA-VLR] support SVD init done: "
            f"rank={self.rank}, "
            f"rho={self.rho}, "
            f"max_coeff_norm={self.max_coeff_norm}, "
            f"residual={self.init_residual}, "
            f"init_strength={self.init_strength}, "
            f"mean_norm={float(self.hba_mean.detach().float().norm(dim=-1).mean().cpu()):.4f}, "
            f"basis_orth={float(basis_orth.cpu()):.4e}, "
            f"visual_adapter={self.use_visual_adapter}, "
            f"visual_rank={self.visual_rank}, "
            f"visual_scale={self.visual_scale}"
        )

        return None

    # ----------------------------------------------------------------------
    # Regularization
    # ----------------------------------------------------------------------
    def kl_divergence(self) -> torch.Tensor:
        mean = self.hba_mean.float()
        std = self._std().float()
        var = std.pow(2).clamp_min(1.0e-12)

        kl = 0.5 * (var + mean.pow(2) - 1.0 - var.log()).sum()
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
        Keep adapted prototypes close to initial CLIP text prototypes.
        """
        text = self._normalize(self.base_text_features.float())
        proto = self.get_prototypes().float()
        proto = self._normalize(proto)

        cos = (proto * text).sum(dim=-1).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)

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
        """
        Keep adapted image features close to raw CLIP image features.
        """
        raw = self._normalize(raw_features.float())
        adapted = self._normalize(adapted_features.float())

        cos = (raw * adapted).sum(dim=-1).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)

        if self.visual_anchor_type == "geodesic":
            reg = torch.acos(cos).pow(2).mean()
        else:
            reg = (1.0 - cos).mean()

        return reg.to(dtype=adapted_features.dtype)

    def extra_loss(self) -> torch.Tensor:
        # HBA-specific loss.py handles anchor terms explicitly.
        return self.basis_regularization()

    def bayes_kl_base_weight(self) -> float:
        return float(self.hba_beta)

    # ----------------------------------------------------------------------
    # MC aggregation
    # ----------------------------------------------------------------------
    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        if training and not self.posterior_predictive_on_train:
            return logits_all.mean(dim=0)

        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        out = torch.logsumexp(log_probs, dim=0) - math.log(float(logits_all.shape[0]))
        return out.to(dtype=logits_all.dtype)

    @torch.no_grad()
    def hba_scores(
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

        probs_all = torch.softmax(logits_all, dim=-1)
        probs_mean = probs_all.mean(dim=0).clamp_min(1.0e-12)

        predictive_entropy = -(probs_mean * probs_mean.log()).sum(dim=-1)

        probs_all_safe = probs_all.clamp_min(1.0e-12)
        expected_entropy = -(probs_all_safe * probs_all_safe.log()).sum(dim=-1).mean(dim=0)

        mutual_info = predictive_entropy - expected_entropy

        return {
            "probs_mean": probs_mean,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_info": mutual_info,
            "max_probability": probs_mean.max(dim=-1).values,
            "logits_all": logits_all,
        }