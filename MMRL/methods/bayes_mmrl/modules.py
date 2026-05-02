from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from backbones.prompt_builder import CUSTOM_TEMPLATES
from methods.mmrl_family.modules import (
    CLIPTextEncoderPlain,
    MMRLFamilyRepresentationLearner,
    MMRLTextEncoder,
    build_zero_shot_text_features,
)


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_with_fallback(obj, primary: str, fallback: str, default):
    if hasattr(obj, primary):
        return getattr(obj, primary)
    if hasattr(obj, fallback):
        return getattr(obj, fallback)
    return default



def _canonical_eval_mode(mode: str | None) -> str:
    mode = str(mode or "mc_predictive")
    if mode == "mean_only":
        return "posterior_mean"
    if mode == "mc_only":
        return "mc_predictive"
    if mode in {
        "posterior_mean",
        "mc_predictive",
        "mean_plus_mc",
        "mean_main_mc_rep",
        "decoupled_mc_rep",
    }:
        return "mean_main_mc_rep" if mode == "decoupled_mc_rep" else mode
    raise ValueError(f"Unsupported EVAL_MODE: {mode}")

def _canonical_eval_aggregation(mode: str | None) -> str:
    mode = str(mode or "prob_mean")
    if mode in {"prob_mean", "logit_mean"}:
        return mode
    raise ValueError(
        f"Unsupported EVAL_AGGREGATION: {mode}. "
        "Expected one of {'prob_mean', 'logit_mean'}"
    )


def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x))


def _build_positive_lower_triangular(
    raw_tril: torch.Tensor,
    min_diag: float,
) -> torch.Tensor:
    tril = torch.tril(raw_tril)
    diag = torch.diagonal(tril, dim1=-2, dim2=-1)
    diag_pos = F.softplus(diag.float()) + float(min_diag)
    tril = tril - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
    return tril


class BayesianTensorParameter(nn.Module):
    """
    Generic factorized Gaussian tensor posterior / prior.

        q(W) = N(mu_q, diag(sigma_q^2))
        p(W) = N(mu_p, diag(sigma_p^2))

    and we support exact initialization q_0 = p.

    Supported sigma modes:
        - global
        - per_token / row
        - per_dim / col
        - diagonal / full   (elementwise mean-field; "full" kept as a legacy alias)

    For current BayesMMRL usage:
        - rep_tokens : global / per_token / diagonal
        - proj_rep   : global / row
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        sigma_mode: str,
        prior_std: float,
        min_sigma: float = 1e-6,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.sigma_mode = str(sigma_mode)
        self.min_sigma = float(min_sigma)

        rho_shape = self._resolve_rho_shape(self.shape, self.sigma_mode)
        rho_storage_shape = rho_shape if len(rho_shape) > 0 else ()

        self.posterior_mean = nn.Parameter(
            torch.zeros(self.shape, dtype=torch.float32)
        )
        self.posterior_rho = nn.Parameter(
            torch.zeros(rho_storage_shape, dtype=torch.float32)
        )

        self.register_buffer(
            "prior_mean",
            torch.zeros(self.shape, dtype=torch.float32),
        )
        self.register_buffer(
            "prior_std_base",
            torch.full(rho_storage_shape, float(prior_std), dtype=torch.float32),
        )

        self.initialize_posterior_as_prior()

    @staticmethod
    def _resolve_rho_shape(shape: Tuple[int, int], sigma_mode: str):
        if sigma_mode == "global":
            return ()
        if sigma_mode in {"per_token", "row"}:
            return (shape[0], 1)
        if sigma_mode in {"per_dim", "col"}:
            return (1, shape[1])
        if sigma_mode in {"diagonal", "full"}:
            return shape
        raise ValueError(f"Unsupported sigma mode: {sigma_mode}")

    def posterior_sigma(self) -> torch.Tensor:
        return F.softplus(self.posterior_rho.float()) + self.min_sigma

    def expanded_posterior_sigma(self) -> torch.Tensor:
        return self.posterior_sigma().expand_as(self.posterior_mean)

    def prior_sigma(self) -> torch.Tensor:
        return self.prior_std_base.float().expand_as(self.posterior_mean)

    def set_prior_mean(self, tensor: torch.Tensor):
        tensor = tensor.detach().float()
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            self.prior_mean.copy_(tensor)

    def set_prior_std(self, std: float | torch.Tensor):
        if torch.is_tensor(std):
            std_tensor = std.detach().float()
            if tuple(std_tensor.shape) != tuple(self.prior_std_base.shape):
                raise ValueError(
                    "Prior std shape mismatch: "
                    f"expected {tuple(self.prior_std_base.shape)}, "
                    f"got {tuple(std_tensor.shape)}"
                )
            target = std_tensor
        else:
            target = torch.full_like(
                self.prior_std_base,
                float(std),
                dtype=torch.float32,
            )

        if torch.any(target <= 0):
            raise ValueError("prior std must be strictly positive")

        with torch.no_grad():
            self.prior_std_base.copy_(target)

    def initialize_posterior_as_prior(self):
        """
        Enforce q_0 = p exactly:
            posterior_mean <- prior_mean
            posterior_sigma <- prior_sigma
        """
        with torch.no_grad():
            self.posterior_mean.copy_(self.prior_mean)

            sigma_target = (
                self.prior_std_base.float() - self.min_sigma
            ).clamp_min(1e-12)
            rho_target = _softplus_inverse(sigma_target)
            self.posterior_rho.copy_(rho_target)

    def configure_prior_and_initialize(
        self,
        prior_mean: torch.Tensor,
        prior_std: float | torch.Tensor,
    ):
        self.set_prior_mean(prior_mean)
        self.set_prior_std(prior_std)
        self.initialize_posterior_as_prior()

    def sample_tensor(self, use_posterior_mean: bool = False) -> torch.Tensor:
        if use_posterior_mean:
            return self.posterior_mean.float()
        eps = torch.randn_like(self.posterior_mean)
        return self.posterior_mean.float() + self.expanded_posterior_sigma() * eps

    def sample_tensor_many(
        self,
        num_samples: int,
        use_posterior_mean: bool = False,
    ) -> torch.Tensor:
        """
        Batched factorized-Gaussian sampling.

        Returns:
            [S, *self.shape], normally [S, K, D].
        """
        num_samples = max(1, int(num_samples))
        mean = self.posterior_mean.float()

        if use_posterior_mean:
            return mean.unsqueeze(0).expand(num_samples, *mean.shape)

        eps = torch.randn(
            (num_samples, *mean.shape),
            device=mean.device,
            dtype=mean.dtype,
        )
        sigma = self.expanded_posterior_sigma().to(mean.dtype).unsqueeze(0)
        return mean.unsqueeze(0) + sigma * eps

    def kl_divergence(self) -> torch.Tensor:
        mu_q = self.posterior_mean.float()
        sigma_q2 = self.expanded_posterior_sigma().pow(2)

        mu_p = self.prior_mean.float()
        sigma_p2 = self.prior_sigma().pow(2)

        kl = 0.5 * (
            sigma_q2 / sigma_p2
            + (mu_q - mu_p).pow(2) / sigma_p2
            - 1.0
            - torch.log(sigma_q2 / sigma_p2)
        )
        return kl.sum()


class BayesianMatrixNormalParameter(nn.Module):
    """
    Bayesian matrix-normal posterior over shared representation matrix R.

        q(R) = MN(M, U, V)

    Prior:
        p(R) = MN(M_prior, I_K, sigma0^2 I_D)

    Supported feature covariance modes:
        - "diag":         V = diag(v^2)
        - "diag_lowrank": V = diag(d^2) + B B^T

    Token covariance U is full over tokens and parameterized by a Cholesky factor.
    By default we normalize it to tr(U)=K to reduce scale ambiguity between U and V.

    Important:
        - initialization enforces q0 = p exactly
        - initial KL is returned as exact zero so your existing assertion still passes
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        feature_cov_mode: str,
        prior_std: float,
        lowrank_rank: int = 0,
        min_sigma: float = 1e-6,
        enforce_token_trace: bool = True,
    ):
        super().__init__()
        if len(shape) != 2:
            raise ValueError(f"Matrix-normal parameter expects 2D shape, got {shape}")

        self.shape = tuple(shape)
        self.n_tokens, self.rep_dim = self.shape
        self.feature_cov_mode = str(feature_cov_mode)
        self.lowrank_rank = int(lowrank_rank)
        self.min_sigma = float(min_sigma)
        self.enforce_token_trace = bool(enforce_token_trace)

        if self.feature_cov_mode not in {"diag", "diag_lowrank"}:
            raise ValueError(
                "feature_cov_mode must be one of {'diag', 'diag_lowrank'}, "
                f"got {self.feature_cov_mode}"
            )
        if self.feature_cov_mode == "diag_lowrank" and self.lowrank_rank <= 0:
            raise ValueError(
                "diag_lowrank feature covariance requires lowrank_rank > 0"
            )

        self.posterior_mean = nn.Parameter(
            torch.zeros(self.shape, dtype=torch.float32)
        )

        # token covariance U = L_U L_U^T
        self.posterior_token_tril_raw = nn.Parameter(
            torch.zeros(self.n_tokens, self.n_tokens, dtype=torch.float32)
        )

        # feature covariance
        self.posterior_feature_diag_rho = nn.Parameter(
            torch.zeros(self.rep_dim, dtype=torch.float32)
        )
        if self.feature_cov_mode == "diag_lowrank":
            self.posterior_feature_lowrank = nn.Parameter(
                torch.zeros(self.rep_dim, self.lowrank_rank, dtype=torch.float32)
            )
        else:
            self.posterior_feature_lowrank = None

        self.register_buffer(
            "prior_mean",
            torch.zeros(self.shape, dtype=torch.float32),
        )
        self.register_buffer(
            "prior_feature_std",
            torch.full((self.rep_dim,), float(prior_std), dtype=torch.float32),
        )

        # Cache small identities without registering them as buffers, so old
        # checkpoints remain state_dict-compatible.
        self._lowrank_eye_cache = None
        self._token_eye_cache = None

        # These flags do not change the valid model math. They only avoid
        # repeated Python/GPU synchronizations from debug-style checks.
        # - _exact_prior_check_active is reset whenever q is reinitialized as p.
        # - _prior_feature_std_is_isotropic is validated when the prior std is set.
        # - debug_pd_check can be set True for debugging, but is False for speed.
        self._exact_prior_check_active = True
        self._prior_feature_std_is_isotropic = True
        self.debug_pd_check = False

        self.initialize_posterior_as_prior()

    def set_prior_mean(self, tensor: torch.Tensor):
        tensor = tensor.detach().float()
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            self.prior_mean.copy_(tensor)

    def set_prior_std(self, std: float | torch.Tensor):
        if torch.is_tensor(std):
            std_tensor = std.detach().float().flatten()
            if std_tensor.numel() == 1:
                target = std_tensor.expand(self.rep_dim)
            elif tuple(std_tensor.shape) == (self.rep_dim,):
                target = std_tensor
            else:
                raise ValueError(
                    f"Prior std for matrix-normal must be scalar or {(self.rep_dim,)}, "
                    f"got {tuple(std_tensor.shape)}"
                )
        else:
            target = torch.full(
                (self.rep_dim,),
                float(std),
                dtype=torch.float32,
                device=self.prior_feature_std.device,
            )

        if torch.any(target <= 0):
            raise ValueError("prior std must be strictly positive")

        # This check used to happen inside every kl_divergence() call via
        # torch.allclose(...), which synchronizes with Python. The prior std is
        # fixed for a run, so validating once here preserves the same constraint
        # without paying that cost every training batch.
        with torch.no_grad():
            target_for_check = target.detach().float().flatten()
            self._prior_feature_std_is_isotropic = bool(
                torch.allclose(
                    target_for_check,
                    target_for_check[:1].expand_as(target_for_check),
                )
            )
            self.prior_feature_std.copy_(target)

    def _identity_token_tril_raw(self) -> torch.Tensor:
        raw = torch.zeros(
            self.n_tokens,
            self.n_tokens,
            dtype=torch.float32,
            device=self.posterior_mean.device,
        )
        diag_target = torch.full(
            (self.n_tokens,),
            1.0 - self.min_sigma,
            dtype=torch.float32,
            device=raw.device,
        ).clamp_min(1e-12)
        raw_diag = _softplus_inverse(diag_target)
        raw.diagonal().copy_(raw_diag)
        return raw

    def initialize_posterior_as_prior(self):
        # q has just been reset to p, so allow the exact-zero KL fast path again.
        self._exact_prior_check_active = True

        with torch.no_grad():
            self.posterior_mean.copy_(self.prior_mean)

            # q0(U) = I_K
            self.posterior_token_tril_raw.copy_(self._identity_token_tril_raw())

            # q0(V) = sigma0^2 I
            diag_target = (
                self.prior_feature_std.float() - self.min_sigma
            ).clamp_min(1e-12)
            rho_target = _softplus_inverse(diag_target)
            self.posterior_feature_diag_rho.copy_(rho_target)

            if self.posterior_feature_lowrank is not None:
                self.posterior_feature_lowrank.zero_()

    def configure_prior_and_initialize(
        self,
        prior_mean: torch.Tensor,
        prior_std: float | torch.Tensor,
    ):
        self.set_prior_mean(prior_mean)
        self.set_prior_std(prior_std)
        self.initialize_posterior_as_prior()

    def _token_cholesky(self) -> torch.Tensor:
        tril = _build_positive_lower_triangular(
            self.posterior_token_tril_raw,
            self.min_sigma,
        )

        if not self.enforce_token_trace:
            return tril

        # tr(U_raw) = ||L||_F^2
        trace_u = tril.pow(2).sum().clamp_min(1e-12)
        scale = torch.sqrt(tril.new_tensor(float(self.n_tokens)) / trace_u)
        return tril * scale

    def _feature_diag_std(self) -> torch.Tensor:
        return F.softplus(self.posterior_feature_diag_rho.float()) + self.min_sigma

    def _get_lowrank_eye(self, device, dtype) -> torch.Tensor:
        if self.lowrank_rank <= 0:
            raise RuntimeError("_get_lowrank_eye called with lowrank_rank <= 0")

        cache = self._lowrank_eye_cache
        if (
            cache is None
            or cache.device != device
            or cache.dtype != dtype
            or cache.shape != (self.lowrank_rank, self.lowrank_rank)
        ):
            cache = torch.eye(
                self.lowrank_rank,
                device=device,
                dtype=dtype,
            )
            self._lowrank_eye_cache = cache
        return cache

    def _get_token_eye(self, device, dtype) -> torch.Tensor:
        cache = self._token_eye_cache
        if (
            cache is None
            or cache.device != device
            or cache.dtype != dtype
            or cache.shape != (self.n_tokens, self.n_tokens)
        ):
            cache = torch.eye(
                self.n_tokens,
                device=device,
                dtype=dtype,
            )
            self._token_eye_cache = cache
        return cache

    def _feature_stats(self):
        diag_std = self._feature_diag_std()
        diag_var = diag_std.pow(2)

        if self.posterior_feature_lowrank is None:
            trace_v = diag_var.sum()
            logdet_v = torch.log(diag_var).sum()
            marginal_diag_v = diag_var
            return trace_v, logdet_v, marginal_diag_v

        B = self.posterior_feature_lowrank.float()
        trace_v = diag_var.sum() + B.pow(2).sum()

        eye = self._get_lowrank_eye(B.device, B.dtype)
        Bt_Dinv_B = B.transpose(0, 1) @ (B / diag_var.unsqueeze(1))
        sign, logabsdet = torch.linalg.slogdet(eye + Bt_Dinv_B)
        # Mathematically, eye + B^T D^{-1} B is positive definite when D is
        # positive. The check below is useful for debugging but forces a
        # GPU->Python synchronization, so keep it disabled in normal training.
        if self.debug_pd_check and torch.any(sign <= 0):
            raise RuntimeError("Feature covariance lost positive definiteness")

        logdet_v = torch.log(diag_var).sum() + logabsdet
        marginal_diag_v = diag_var + B.pow(2).sum(dim=1)
        return trace_v, logdet_v, marginal_diag_v

    def posterior_sigma(self) -> torch.Tensor:
        # marginal std per entry, shape [K, D]
        L_u = self._token_cholesky()
        diag_u = L_u.pow(2).sum(dim=1)  # diag(U)
        _, _, marginal_diag_v = self._feature_stats()  # diag(V)
        marginal_var = diag_u.unsqueeze(1) * marginal_diag_v.unsqueeze(0)
        return marginal_var.clamp_min(0.0).sqrt()

    def sample_tensor(self, use_posterior_mean: bool = False) -> torch.Tensor:
        if use_posterior_mean:
            return self.posterior_mean.float()

        dtype = self.posterior_mean.dtype
        device = self.posterior_mean.device

        eps = torch.randn(
            self.n_tokens,
            self.rep_dim,
            device=device,
            dtype=dtype,
        )
        z = eps * self._feature_diag_std().to(dtype).unsqueeze(0)

        if self.posterior_feature_lowrank is not None:
            eta = torch.randn(
                self.n_tokens,
                self.lowrank_rank,
                device=device,
                dtype=dtype,
            )
            z = z + eta @ self.posterior_feature_lowrank.float().to(dtype).transpose(
                0,
                1,
            )

        L_u = self._token_cholesky().to(dtype)
        return self.posterior_mean.float().to(dtype) + L_u @ z

    def sample_tensor_many(
        self,
        num_samples: int,
        use_posterior_mean: bool = False,
    ) -> torch.Tensor:
        """
        Batched matrix-normal sampling.

        Returns:
            [S, K, D]

        This has the same distribution as repeatedly calling sample_tensor(),
        but it reduces Python-loop overhead and makes matrix-normal operations
        less fragmented.
        """
        num_samples = max(1, int(num_samples))
        mean = self.posterior_mean.float()

        if use_posterior_mean:
            return mean.unsqueeze(0).expand(num_samples, *mean.shape)

        dtype = self.posterior_mean.dtype
        device = self.posterior_mean.device

        eps = torch.randn(
            num_samples,
            self.n_tokens,
            self.rep_dim,
            device=device,
            dtype=dtype,
        )
        feature_diag_std = self._feature_diag_std().to(dtype)
        z = eps * feature_diag_std.view(1, 1, self.rep_dim)

        if self.posterior_feature_lowrank is not None:
            eta = torch.randn(
                num_samples,
                self.n_tokens,
                self.lowrank_rank,
                device=device,
                dtype=dtype,
            )
            lowrank_t = self.posterior_feature_lowrank.float().to(dtype).transpose(
                0,
                1,
            )
            z = z + eta @ lowrank_t

        L_u = self._token_cholesky().to(dtype)
        z = torch.matmul(L_u.unsqueeze(0), z)

        return mean.to(dtype).unsqueeze(0) + z

    def _is_exact_prior_state(self) -> bool:
        mean_ok = torch.allclose(
            self.posterior_mean.float().detach(),
            self.prior_mean.float().detach(),
            atol=1e-7,
            rtol=0.0,
        )
        token_ok = torch.allclose(
            self._token_cholesky().float().detach(),
            self._get_token_eye(self.posterior_mean.device, torch.float32),
            atol=1e-6,
            rtol=0.0,
        )
        feat_ok = torch.allclose(
            self._feature_diag_std().float().detach(),
            self.prior_feature_std.float().detach(),
            atol=1e-7,
            rtol=0.0,
        )
        if self.posterior_feature_lowrank is None:
            lowrank_ok = True
        else:
            lowrank_ok = torch.allclose(
                self.posterior_feature_lowrank.float().detach(),
                torch.zeros_like(self.posterior_feature_lowrank.float().detach()),
                atol=1e-7,
                rtol=0.0,
            )
        return bool(mean_ok and token_ok and feat_ok and lowrank_ok)

    def kl_divergence(self) -> torch.Tensor:
        # Keep the exact-zero q0=p path, but only while it can still be true.
        # The previous implementation checked torch.allclose(...) on every
        # batch, which synchronizes GPU work with Python and is especially slow
        # for matrix-normal sweeps. Once q is observed to have left p, the exact
        # prior state cannot hold again under normal continuous optimization, so
        # we use the analytic KL directly.
        if self._exact_prior_check_active:
            if self._is_exact_prior_state():
                return self.posterior_mean.float().new_zeros(())
            self._exact_prior_check_active = False

        if not self._prior_feature_std_is_isotropic:
            raise ValueError(
                "Matrix-normal KL currently assumes isotropic feature prior std"
            )

        sigma0_sq = self.prior_feature_std.float()[0].pow(2)
        mean_delta = self.posterior_mean.float() - self.prior_mean.float()

        L_u = self._token_cholesky()
        trace_u = L_u.pow(2).sum()  # tr(U)
        logdet_u = 2.0 * torch.log(torch.diagonal(L_u)).sum()

        trace_v, logdet_v, _ = self._feature_stats()

        D = float(self.rep_dim)
        K = float(self.n_tokens)

        mean_quad = mean_delta.pow(2).sum() / sigma0_sq
        trace_term = (trace_u * trace_v) / sigma0_sq
        const_term = -K * D
        prior_logdet_term = K * D * torch.log(sigma0_sq)
        posterior_logdet_term = -D * logdet_u - K * logdet_v

        return 0.5 * (
            mean_quad
            + trace_term
            + const_term
            + prior_logdet_term
            + posterior_logdet_term
        )


class DeterministicRepresentationLearnerAdapter(nn.Module):
    """
    Wrap original deterministic MMRL representation learner with a compatible API.
    Used when Bayes is moved from R to proj_rep.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.base = MMRLFamilyRepresentationLearner(
            cfg,
            cfg.BAYES_MMRL,
            classnames,
            clip_model,
        )

    @property
    def tokenized_prompts(self):
        return self.base.tokenized_prompts

    @property
    def prompt_embeddings(self):
        return self.base.prompt_embeddings

    def project_mean_tokens(self):
        return self.base()

    def project_sample_tokens(self):
        return self.base()

    def project_sample_tokens_many(
        self,
        num_samples: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compatibility path for bayes_target == proj_rep.

        The deterministic representation learner has no rep-token posterior,
        so repeat the same projected tokens along the MC dimension.
        """
        num_samples = max(1, int(num_samples))
        rep_text, rep_visual = self.base()

        rep_text_many = [
            x.unsqueeze(0).expand(num_samples, *x.shape)
            for x in rep_text
        ]
        rep_visual_many = [
            x.unsqueeze(0).expand(num_samples, *x.shape)
            for x in rep_visual
        ]
        return rep_text_many, rep_visual_many

    def posterior_sigma(self):
        device = self.base.compound_rep_tokens.device
        return torch.zeros(
            self.base.compound_rep_tokens.shape[0],
            1,
            device=device,
            dtype=torch.float32,
        )

    def kl_divergence(self):
        return self.base.compound_rep_tokens.new_zeros(())

    def forward(self):
        return self.base()


class BayesianMultiModalRepresentationLearner(nn.Module):
    """
    Bayesian version of shared representation learner.

    Supported posterior schemes on R:
        1) global
        2) per_token
        3) diagonal
        4) matrix_normal_diag
        5) matrix_normal_diag_lowrank
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        bayes_cfg = cfg.BAYES_MMRL

        n_rep_tokens = int(bayes_cfg.N_REP_TOKENS)
        rep_dim = int(bayes_cfg.REP_DIM)
        self.dtype = clip_model.dtype
        self.rep_layers_length = len(bayes_cfg.REP_LAYERS)

        rep_sigma_mode = str(getattr(bayes_cfg, "REP_SIGMA_MODE", "global"))
        rep_prior_std = float(getattr(bayes_cfg, "REP_PRIOR_STD", 0.05))

        supported_modes = {
            "global",
            "per_token",
            "diagonal",
            "matrix_normal_diag",
            "matrix_normal_diag_lowrank",
        }
        if rep_sigma_mode not in supported_modes:
            raise ValueError(
                f"REP_SIGMA_MODE must be one of {supported_modes}, got {rep_sigma_mode}"
            )

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        token_device = clip_model.token_embedding.weight.device
        tokenized_prompts = [
            clip.tokenize(template.format(text.replace("_", " "))).to(token_device)
            for text in classnames
        ]
        tokenized_prompts = torch.cat(tokenized_prompts, dim=0)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(
                self.dtype
            )
        self.register_buffer("prompt_embeddings", prompt_embeddings)

        shape = (n_rep_tokens, rep_dim)

        if rep_sigma_mode in {"global", "per_token", "diagonal"}:
            self.rep_posterior = BayesianTensorParameter(
                shape=shape,
                sigma_mode=rep_sigma_mode,
                prior_std=rep_prior_std,
            )
        elif rep_sigma_mode == "matrix_normal_diag":
            self.rep_posterior = BayesianMatrixNormalParameter(
                shape=shape,
                feature_cov_mode="diag",
                prior_std=rep_prior_std,
                lowrank_rank=0,
                enforce_token_trace=bool(
                    getattr(bayes_cfg, "REP_MN_ENFORCE_TRACE", True)
                ),
            )
        elif rep_sigma_mode == "matrix_normal_diag_lowrank":
            self.rep_posterior = BayesianMatrixNormalParameter(
                shape=shape,
                feature_cov_mode="diag_lowrank",
                prior_std=rep_prior_std,
                lowrank_rank=int(getattr(bayes_cfg, "REP_MN_LOWRANK_RANK", 8)),
                enforce_token_trace=bool(
                    getattr(bayes_cfg, "REP_MN_ENFORCE_TRACE", True)
                ),
            )
        else:
            raise ValueError(f"Unsupported REP_SIGMA_MODE: {rep_sigma_mode}")

        # default zero-mean prior with q_0 = p
        zero_prior = torch.zeros(n_rep_tokens, rep_dim, dtype=torch.float32)
        self.rep_posterior.configure_prior_and_initialize(
            prior_mean=zero_prior,
            prior_std=rep_prior_std,
        )

        self.compound_rep_tokens_r2vproj = _get_clones(
            nn.Linear(rep_dim, visual_dim),
            self.rep_layers_length,
        )
        self.compound_rep_tokens_r2tproj = _get_clones(
            nn.Linear(rep_dim, text_dim),
            self.rep_layers_length,
        )

    @property
    def posterior_mean(self):
        return self.rep_posterior.posterior_mean

    def posterior_sigma(self):
        return self.rep_posterior.posterior_sigma()

    def kl_divergence(self):
        return self.rep_posterior.kl_divergence()

    def configure_rep_prior_and_initialize(
        self,
        prior_mean: torch.Tensor,
        prior_std: float,
    ):
        self.rep_posterior.configure_prior_and_initialize(
            prior_mean=prior_mean,
            prior_std=prior_std,
        )

    def _project_rep_tokens(
        self,
        rep_tokens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        compound_rep_tokens_text: List[torch.Tensor] = []
        compound_rep_tokens_visual: List[torch.Tensor] = []

        rep_tokens = rep_tokens.to(self.compound_rep_tokens_r2tproj[0].weight.dtype)

        for index in range(self.rep_layers_length):
            rep_text = self.compound_rep_tokens_r2tproj[index](rep_tokens).type(
                self.dtype
            )
            rep_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens).type(
                self.dtype
            )
            compound_rep_tokens_text.append(rep_text)
            compound_rep_tokens_visual.append(rep_visual)

        return compound_rep_tokens_text, compound_rep_tokens_visual

    def _project_rep_tokens_many(
        self,
        rep_tokens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Batched projection for MC samples.

        Args:
            rep_tokens: [S, K, D]

        Returns:
            list length = rep_layers_length;
            each list element is [S, K, text_dim] or [S, K, visual_dim].
        """
        if rep_tokens.dim() != 3:
            raise ValueError(
                f"_project_rep_tokens_many expects [S, K, D], got {tuple(rep_tokens.shape)}"
            )

        compound_rep_tokens_text: List[torch.Tensor] = []
        compound_rep_tokens_visual: List[torch.Tensor] = []

        rep_tokens = rep_tokens.to(self.compound_rep_tokens_r2tproj[0].weight.dtype)

        for index in range(self.rep_layers_length):
            rep_text = self.compound_rep_tokens_r2tproj[index](rep_tokens).type(
                self.dtype
            )
            rep_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens).type(
                self.dtype
            )
            compound_rep_tokens_text.append(rep_text)
            compound_rep_tokens_visual.append(rep_visual)

        return compound_rep_tokens_text, compound_rep_tokens_visual

    def project_mean_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._project_rep_tokens(
            self.rep_posterior.sample_tensor(use_posterior_mean=True)
        )

    def project_sample_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._project_rep_tokens(
            self.rep_posterior.sample_tensor(use_posterior_mean=False)
        )

    def project_sample_tokens_many(
        self,
        num_samples: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample MC rep tokens in one batched call, then project them in one
        batched pass through the representation projection layers.
        """
        num_samples = max(1, int(num_samples))

        if hasattr(self.rep_posterior, "sample_tensor_many"):
            rep_tokens = self.rep_posterior.sample_tensor_many(
                num_samples,
                use_posterior_mean=False,
            )
        else:
            rep_tokens = torch.stack(
                [
                    self.rep_posterior.sample_tensor(use_posterior_mean=False)
                    for _ in range(num_samples)
                ],
                dim=0,
            )

        return self._project_rep_tokens_many(rep_tokens)

    def forward(self, use_posterior_mean: bool = False):
        if use_posterior_mean:
            return self.project_mean_tokens()
        return self.project_sample_tokens()


class BayesianVisualEncoderWrapper(nn.Module):
    """
    Bayesian wrapper for the representation visual projection P_v^r.

    Scheme C:
        random variable is proj_rep

        prior mode = "self_proj_rep":
            p(W) = N(P_v^r_det, Sigma), q_0(W) = p(W)

        prior mode = "clip_proj":
            p(W) = N(P_v^c_clip, Sigma), q_0(W) = p(W)

    where:
        - P_v^r_det is the deterministic MMRL representation head (base.proj_rep)
        - P_v^c_clip is the CLIP class-token visual projection head (base.proj)
    """

    def __init__(self, base_visual: nn.Module, cfg):
        super().__init__()
        if not hasattr(base_visual, "proj_rep"):
            raise ValueError(
                "Bayesian proj_rep requires a ViT visual backbone with proj_rep"
            )
        if not hasattr(base_visual, "proj"):
            raise ValueError(
                "Bayesian proj_rep requires a ViT visual backbone with proj"
            )

        self.base = base_visual
        bayes_cfg = cfg.BAYES_MMRL

        sigma_mode = str(getattr(bayes_cfg, "PROJ_REP_SIGMA_MODE", "row"))
        if sigma_mode not in {"global", "row"}:
            raise ValueError(
                f"PROJ_REP_SIGMA_MODE must be one of {{'global', 'row'}}, got {sigma_mode}"
            )

        prior_mode = str(
            getattr(bayes_cfg, "PROJ_REP_PRIOR_MODE", "clip_proj")
        )
        if prior_mode == "self_proj_rep":
            prior_mean = self.base.proj_rep.detach().float()
        elif prior_mode == "clip_proj":
            prior_mean = self.base.proj.detach().float()
        else:
            raise ValueError(
                "PROJ_REP_PRIOR_MODE must be one of "
                "{'self_proj_rep', 'clip_proj'}, "
                f"got {prior_mode}"
            )

        prior_std = float(getattr(bayes_cfg, "PROJ_REP_PRIOR_STD", 0.01))
        self.proj_rep_prior_mode = prior_mode

        self.bayes_proj_rep = BayesianTensorParameter(
            shape=tuple(self.base.proj_rep.shape),
            sigma_mode=sigma_mode,
            prior_std=prior_std,
        )
        self.bayes_proj_rep.configure_prior_and_initialize(
            prior_mean=prior_mean,
            prior_std=prior_std,
        )

    def posterior_sigma(self):
        return self.bayes_proj_rep.posterior_sigma()

    def kl_divergence(self):
        return self.bayes_proj_rep.kl_divergence()

    def _resolve_proj_rep(self, use_posterior_mean: bool) -> torch.Tensor:
        return self.bayes_proj_rep.sample_tensor(
            use_posterior_mean=use_posterior_mean
        )

    def _forward_vit_with_proj(
        self,
        inputs,
        proj_rep: torch.Tensor,
    ):
        x = inputs[0]
        compound_rep_tokens = list(inputs[1])

        x = self.base.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.base.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.base.positional_embedding.to(x.dtype)
        x = self.base.ln_pre(x)

        x = x.permute(1, 0, 2)
        outputs = self.base.transformer([x, compound_rep_tokens, 0])
        x = outputs[0]
        x = x.permute(1, 0, 2)

        n_tokens = compound_rep_tokens[0].shape[0]
        x_rep = self.base.ln_post(x[:, 1 : 1 + n_tokens, :]).mean(dim=1)
        x_rep = x_rep @ proj_rep.to(x_rep.dtype)

        x_cls = self.base.ln_post(x[:, 0, :])
        x_cls = x_cls @ self.base.proj

        return x_cls, x_rep

    def forward(self, inputs, use_posterior_mean: bool = False):
        proj_rep = self._resolve_proj_rep(use_posterior_mean=use_posterior_mean)
        return self._forward_vit_with_proj(inputs, proj_rep)


# Replace the existing class BayesianCustomMMRLModel with this complete class.
class BayesianCustomMMRLModel(nn.Module):
    """
    Supports:
        - Scheme A: Bayes on R with zero prior
        - Scheme B: Bayes on R with CLIP prior
        - Scheme C: Bayes on P_v^r with deterministic R

    Default training/evaluation path for BayesMMRLMethod is now:
        Mean-main + MC-rep

    Dependency contract for forward_mean_main_mc_rep():
        - logits_main, image_features_main, text_features depend on posterior mean only.
        - logits_rep_stack depends on MC samples.
        - logits_fusion uses clean main + posterior-predictive rep.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        bayes_cfg = cfg.BAYES_MMRL

        self.alpha = float(bayes_cfg.ALPHA)
        self.bayes_target = str(getattr(bayes_cfg, "BAYES_TARGET", "rep_tokens"))
        self.eval_mode = _canonical_eval_mode(
            getattr(bayes_cfg, "EVAL_MODE", "mc_predictive")
        )
        self.eval_use_posterior_mean = bool(
            getattr(bayes_cfg, "EVAL_USE_POSTERIOR_MEAN", False)
        )

        self.eval_aggregation = _canonical_eval_aggregation(
            getattr(bayes_cfg, "EVAL_AGGREGATION", "prob_mean")
        )

        self.n_mc_test = max(1, int(bayes_cfg.N_MC_TEST))

        # This is distribution-equivalent to the old per-sample loop for the
        # rep-token posterior sampling/projection. It can change the exact RNG
        # draw order, so set BAYES_MMRL.BATCH_REP_TOKEN_MC=False to restore the
        # old deterministic sequence if you need bitwise comparability.
        self.batch_rep_token_mc = bool(
            getattr(bayes_cfg, "BATCH_REP_TOKEN_MC", True)
        )

        if self.bayes_target == "rep_tokens":
            self.representation_learner = BayesianMultiModalRepresentationLearner(
                cfg,
                classnames,
                clip_model,
            )
        elif self.bayes_target == "proj_rep":
            self.representation_learner = DeterministicRepresentationLearnerAdapter(
                cfg,
                classnames,
                clip_model,
            )
        else:
            raise ValueError(f"Unsupported BAYES_TARGET: {self.bayes_target}")

        self.register_buffer(
            "tokenized_prompts",
            self.representation_learner.tokenized_prompts.clone(),
        )
        self.register_buffer(
            "prompt_embeddings",
            self.representation_learner.prompt_embeddings.clone(),
        )

        if self.bayes_target == "proj_rep":
            self.image_encoder = BayesianVisualEncoderWrapper(clip_model.visual, cfg)
        else:
            self.image_encoder = clip_model.visual

        self.text_encoder = MMRLTextEncoder(clip_model)
        self.dtype = clip_model.dtype

        self._cached_text_features = None
        self._cached_rep_visual = None
        self._cached_mode = None

    def clear_inference_cache(self):
        self._cached_text_features = None
        self._cached_rep_visual = None
        self._cached_mode = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def kl_terms(self) -> Dict[str, torch.Tensor]:
        zero = torch.zeros(
            (),
            device=self.prompt_embeddings.device,
            dtype=torch.float32,
        )
        rep_kl = (
            self.representation_learner.kl_divergence()
            if hasattr(self.representation_learner, "kl_divergence")
            else zero
        )
        proj_kl = (
            self.image_encoder.kl_divergence()
            if isinstance(self.image_encoder, BayesianVisualEncoderWrapper)
            else zero
        )
        return {
            "rep_tokens": rep_kl,
            "proj_rep": proj_kl,
        }

    def posterior_stats(self) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}
        if hasattr(self.representation_learner, "posterior_sigma"):
            stats["rep_posterior_sigma"] = (
                self.representation_learner.posterior_sigma().detach()
            )
        if isinstance(self.image_encoder, BayesianVisualEncoderWrapper):
            stats["proj_rep_posterior_sigma"] = (
                self.image_encoder.posterior_sigma().detach()
            )
        return stats

    def _encode_image(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        use_posterior_mean_proj: bool = False,
    ):
        if isinstance(self.image_encoder, BayesianVisualEncoderWrapper):
            return self.image_encoder(
                [image.type(self.dtype), list(compound_rep_tokens_visual)],
                use_posterior_mean=use_posterior_mean_proj,
            )
        return self.image_encoder(
            [image.type(self.dtype), list(compound_rep_tokens_visual)]
        )

    def _encode_with_tokens(
        self,
        image: torch.Tensor,
        compound_rep_tokens_text: Sequence[torch.Tensor],
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        use_posterior_mean_proj: bool = False,
    ):
        text_features = self.text_encoder(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features, image_features_rep = self._encode_image(
            image,
            compound_rep_tokens_visual,
            use_posterior_mean_proj=use_posterior_mean_proj,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(
            dim=-1,
            keepdim=True,
        )

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _forward_with_cached_text(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        text_features: torch.Tensor,
        use_posterior_mean_proj: bool = False,
    ):
        image_features, image_features_rep = self._encode_image(
            image,
            compound_rep_tokens_visual,
            use_posterior_mean_proj=use_posterior_mean_proj,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(
            dim=-1,
            keepdim=True,
        )

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _aggregate_train_outputs(self, sample_outputs):
        logits = torch.stack([out[0] for out in sample_outputs], dim=0).mean(dim=0)
        logits_rep = torch.stack([out[1] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        logits_fusion = torch.stack([out[2] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        image_features = torch.stack([out[3] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        text_features = torch.stack([out[4] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return logits, logits_rep, logits_fusion, image_features, text_features

    def _aggregate_eval_outputs(self, sample_outputs):
        eps = 1e-8

        if self.eval_aggregation == "prob_mean":
            probs = torch.stack(
                [torch.softmax(out[0], dim=-1) for out in sample_outputs],
                dim=0,
            ).mean(dim=0)
            probs_rep = torch.stack(
                [torch.softmax(out[1], dim=-1) for out in sample_outputs],
                dim=0,
            ).mean(dim=0)
            probs_fusion = torch.stack(
                [torch.softmax(out[2], dim=-1) for out in sample_outputs],
                dim=0,
            ).mean(dim=0)

            logits = torch.log(probs.clamp_min(eps))
            logits_rep = torch.log(probs_rep.clamp_min(eps))
            logits_fusion = torch.log(probs_fusion.clamp_min(eps))

        elif self.eval_aggregation == "logit_mean":
            logits = torch.stack([out[0] for out in sample_outputs], dim=0).mean(
                dim=0
            )
            logits_rep = torch.stack([out[1] for out in sample_outputs], dim=0).mean(
                dim=0
            )
            logits_fusion = torch.stack([out[2] for out in sample_outputs], dim=0).mean(
                dim=0
            )

        else:
            raise ValueError(f"Unsupported eval aggregation: {self.eval_aggregation}")

        image_features = torch.stack([out[3] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        text_features = torch.stack([out[4] for out in sample_outputs], dim=0).mean(
            dim=0
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return logits, logits_rep, logits_fusion, image_features, text_features

    @torch.no_grad()
    def _get_cached_mean_eval_state(self):
        cache_mode = "posterior_mean"

        if (
            self._cached_text_features is None
            or self._cached_rep_visual is None
            or self._cached_mode != cache_mode
        ):
            rep_text, rep_visual = self.representation_learner.project_mean_tokens()
            text_features = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                rep_text,
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self._cached_text_features = text_features
            self._cached_rep_visual = rep_visual
            self._cached_mode = cache_mode

        return self._cached_text_features, self._cached_rep_visual

    @staticmethod
    def _slice_projected_tokens(
        rep_text_many: Sequence[torch.Tensor],
        rep_visual_many: Sequence[torch.Tensor],
        sample_index: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert batched projected tokens back to the original per-sample format.

        Input list element shape:
            [S, K, D]

        Output list element shape:
            [K, D]
        """
        rep_text = [x[sample_index].contiguous() for x in rep_text_many]
        rep_visual = [x[sample_index].contiguous() for x in rep_visual_many]
        return rep_text, rep_visual

    def forward_train_samples(self, image: torch.Tensor, num_samples: int):
        """Old full-MC path, kept for ablations."""
        num_samples = max(1, int(num_samples))
        outputs = []

        if self.batch_rep_token_mc and hasattr(self.representation_learner, "project_sample_tokens_many"):
            rep_text_many, rep_visual_many = (
                self.representation_learner.project_sample_tokens_many(num_samples)
            )

            for sample_index in range(num_samples):
                rep_text, rep_visual = self._slice_projected_tokens(
                    rep_text_many,
                    rep_visual_many,
                    sample_index,
                )
                outputs.append(
                    self._encode_with_tokens(
                        image,
                        rep_text,
                        rep_visual,
                        use_posterior_mean_proj=False,
                    )
                )

            return outputs

        for _ in range(num_samples):
            rep_text, rep_visual = self.representation_learner.project_sample_tokens()
            outputs.append(
                self._encode_with_tokens(
                    image,
                    rep_text,
                    rep_visual,
                    use_posterior_mean_proj=False,
                )
            )
        return outputs

    def _sample_rep_visual_tokens_many(
        self,
        num_samples: int,
    ) -> List[List[torch.Tensor]]:
        """
        Sample only the visual representation-token path for MC rep logits.

        The text samples, if produced by the representation learner, are
        intentionally ignored so the classifier w stays deterministic.
        """
        num_samples = max(1, int(num_samples))

        if self.batch_rep_token_mc and hasattr(
            self.representation_learner,
            "project_sample_tokens_many",
        ):
            rep_text_many, rep_visual_many = (
                self.representation_learner.project_sample_tokens_many(num_samples)
            )
            del rep_text_many
            return [
                [x[sample_index].contiguous() for x in rep_visual_many]
                for sample_index in range(num_samples)
            ]

        rep_visual_samples = []
        for _ in range(num_samples):
            rep_text, rep_visual = self.representation_learner.project_sample_tokens()
            del rep_text
            rep_visual_samples.append(rep_visual)
        return rep_visual_samples

    def forward_mean_main_mc_rep(
        self,
        image: torch.Tensor,
        num_samples: int,
        use_posterior_mean_for_rep: bool = False,
        aggregation: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Recommended decoupled forward.

        Clean branch:
            R_mu -> text_features_mu, image_features_main, logits_main

        Stochastic branch:
            R_s or sampled proj_rep -> image_features_rep_s -> logits_rep_s
            with text_features_mu fixed.
        """
        num_samples = max(1, int(num_samples))
        aggregation = _canonical_eval_aggregation(aggregation or self.eval_aggregation)
        eps = 1e-8

        # 1) Clean posterior-mean path for class token and text classifier.
        rep_text_mu, rep_visual_mu = self.representation_learner.project_mean_tokens()

        text_features_mu = self.text_encoder(
            self.prompt_embeddings,
            self.tokenized_prompts,
            rep_text_mu,
        )
        text_features_mu = text_features_mu / text_features_mu.norm(
            dim=-1,
            keepdim=True,
        )

        image_features_main, image_features_rep_mu = self._encode_image(
            image,
            rep_visual_mu,
            use_posterior_mean_proj=True,
        )
        image_features_main = image_features_main / image_features_main.norm(
            dim=-1,
            keepdim=True,
        )
        image_features_rep_mu = image_features_rep_mu / image_features_rep_mu.norm(
            dim=-1,
            keepdim=True,
        )

        logits_main = 100.0 * image_features_main @ text_features_mu.t()

        # 2) MC representation path. Text classifier remains fixed at w_mu.
        logits_rep_list = []
        image_features_rep_list = []

        if use_posterior_mean_for_rep:
            rep_visual_samples = [rep_visual_mu for _ in range(num_samples)]
            proj_mean_flags = [True for _ in range(num_samples)]
        else:
            rep_visual_samples = self._sample_rep_visual_tokens_many(num_samples)
            proj_mean_flags = [False for _ in range(num_samples)]

        for rep_visual_s, use_proj_mean_s in zip(rep_visual_samples, proj_mean_flags):
            _, image_features_rep_s = self._encode_image(
                image,
                rep_visual_s,
                use_posterior_mean_proj=use_proj_mean_s,
            )
            image_features_rep_s = image_features_rep_s / image_features_rep_s.norm(
                dim=-1,
                keepdim=True,
            )
            logits_rep_s = 100.0 * image_features_rep_s @ text_features_mu.t()
            logits_rep_list.append(logits_rep_s)
            image_features_rep_list.append(image_features_rep_s)

        logits_rep_stack = torch.stack(logits_rep_list, dim=0)

        if aggregation == "prob_mean":
            probs_main = torch.softmax(logits_main, dim=-1)
            probs_rep = torch.softmax(logits_rep_stack, dim=-1).mean(dim=0)
            probs_fusion = self.alpha * probs_main + (1.0 - self.alpha) * probs_rep

            logits_rep = torch.log(probs_rep.clamp_min(eps))
            logits_fusion = torch.log(probs_fusion.clamp_min(eps))
        elif aggregation == "logit_mean":
            logits_rep = logits_rep_stack.mean(dim=0)
            logits_fusion = self.alpha * logits_main + (1.0 - self.alpha) * logits_rep
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        image_features_rep = torch.stack(image_features_rep_list, dim=0).mean(dim=0)
        image_features_rep = image_features_rep / image_features_rep.norm(
            dim=-1,
            keepdim=True,
        )

        return {
            "logits_main": logits_main,
            "logits_rep": logits_rep,
            "logits_fusion": logits_fusion,
            "logits_rep_stack": logits_rep_stack,
            "image_features_main": image_features_main,
            "image_features_rep": image_features_rep,
            "text_features": text_features_mu,
            "image_features_rep_mu": image_features_rep_mu,
        }

    @torch.no_grad()
    def forward_eval(
        self,
        image: torch.Tensor,
        num_samples: int | None = None,
        use_posterior_mean: bool | None = None,
    ):
        if num_samples is None:
            num_samples = self.n_mc_test
        num_samples = max(1, int(num_samples))

        if use_posterior_mean is not None:
            if use_posterior_mean and num_samples == 1:
                eval_mode = "posterior_mean"
            elif use_posterior_mean and num_samples > 1:
                eval_mode = "mean_plus_mc"
            else:
                eval_mode = "mc_predictive"
        else:
            eval_mode = self.eval_mode
            if eval_mode is None:
                eval_mode = (
                    "posterior_mean"
                    if self.eval_use_posterior_mean
                    else "mc_predictive"
                )

        eval_mode = _canonical_eval_mode(eval_mode)

        if eval_mode == "mean_main_mc_rep":
            out = self.forward_mean_main_mc_rep(
                image,
                num_samples=num_samples,
                use_posterior_mean_for_rep=False,
                aggregation=self.eval_aggregation,
            )
            return (
                out["logits_main"],
                out["logits_rep"],
                out["logits_fusion"],
                out["image_features_main"],
                out["text_features"],
            )

        if eval_mode == "posterior_mean":
            text_features, rep_visual = self._get_cached_mean_eval_state()
            return self._forward_with_cached_text(
                image,
                rep_visual,
                text_features,
                use_posterior_mean_proj=True,
            )

        if eval_mode == "mc_predictive":
            sample_outputs = []

            if self.batch_rep_token_mc and hasattr(self.representation_learner, "project_sample_tokens_many"):
                rep_text_many, rep_visual_many = (
                    self.representation_learner.project_sample_tokens_many(num_samples)
                )

                for sample_index in range(num_samples):
                    rep_text, rep_visual = self._slice_projected_tokens(
                        rep_text_many,
                        rep_visual_many,
                        sample_index,
                    )
                    sample_outputs.append(
                        self._encode_with_tokens(
                            image,
                            rep_text,
                            rep_visual,
                            use_posterior_mean_proj=False,
                        )
                    )

                return self._aggregate_eval_outputs(sample_outputs)

            for _ in range(num_samples):
                rep_text, rep_visual = (
                    self.representation_learner.project_sample_tokens()
                )
                sample_outputs.append(
                    self._encode_with_tokens(
                        image,
                        rep_text,
                        rep_visual,
                        use_posterior_mean_proj=False,
                    )
                )
            return self._aggregate_eval_outputs(sample_outputs)

        if eval_mode == "mean_plus_mc":
            sample_outputs = []

            rep_text, rep_visual = self.representation_learner.project_mean_tokens()
            sample_outputs.append(
                self._encode_with_tokens(
                    image,
                    rep_text,
                    rep_visual,
                    use_posterior_mean_proj=True,
                )
            )

            n_mc = max(0, num_samples - 1)

            if n_mc > 0 and self.batch_rep_token_mc and hasattr(
                self.representation_learner,
                "project_sample_tokens_many",
            ):
                rep_text_many, rep_visual_many = (
                    self.representation_learner.project_sample_tokens_many(n_mc)
                )

                for sample_index in range(n_mc):
                    rep_text, rep_visual = self._slice_projected_tokens(
                        rep_text_many,
                        rep_visual_many,
                        sample_index,
                    )
                    sample_outputs.append(
                        self._encode_with_tokens(
                            image,
                            rep_text,
                            rep_visual,
                            use_posterior_mean_proj=False,
                        )
                    )

                return self._aggregate_eval_outputs(sample_outputs)

            for _ in range(n_mc):
                rep_text, rep_visual = (
                    self.representation_learner.project_sample_tokens()
                )
                sample_outputs.append(
                    self._encode_with_tokens(
                        image,
                        rep_text,
                        rep_visual,
                        use_posterior_mean_proj=False,
                    )
                )

            return self._aggregate_eval_outputs(sample_outputs)

        raise ValueError(f"Unsupported EVAL_MODE: {eval_mode}")





