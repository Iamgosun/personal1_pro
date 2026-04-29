from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseAdapter


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class PPProKeROneHotAdapter(BaseAdapter):
    """
    PP-ProKeR-OneHot adapter.

    This implementation has two different CLIP-logit notions:

    1. Official ProKeR base logits:
         cosine_logits = normalize(image) @ normalize(text).T

       These are used inside the ProKeR residual:
         R = one_hot(y) - cosine_logits(S)

       This follows official ProKeR, whose proker.py computes:
         logits_text_shots = vecs @ clip_weights.T
         logits_text_test  = test_features @ clip_weights.T

       without multiplying by CLIP logit_scale or 100.

    2. Outer ClipAdaptersModel base logits:
         outer_logits = cosine_logits * clip_model.logit_scale.exp()

       The outer model computes these before calling cache_logits().
       Therefore cache_logits() must return:
         proker_final_logits - outer_logits

       so that the actual final output becomes:
         outer_logits + cache_logits = proker_final_logits

    First-round PP-ProKeR-OneHot includes:
      - official one-hot residual
      - official ProKeR-compatible solve
      - optional diagonal GP-style posterior predictive

    It does NOT include:
      - logit-space target
      - class covariance
      - CLAP base
      - support-LOO tuning
    """

    initialization_name = "PP_PROKER_ONEHOT"
    adapter_kind = "prototype"
    uses_cache = True
    closed_form_adapter = True
    requires_training = False

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = getattr(cfg, "CLIP_ADAPTERS", None)

        self.beta = float(getattr(cad, "PP_PROKER_BETA", 1.0))
        self.lmbda = float(getattr(cad, "PP_PROKER_LAMBDA", 1.0))

        # GP variance ridge. If <= 0, it defaults to PP_PROKER_LAMBDA.
        self.gp_delta = float(getattr(cad, "PP_PROKER_GP_DELTA", -1.0))

        # Posterior predictive controls.
        self.rho = float(getattr(cad, "PP_PROKER_RHO", 1.0))
        self.tau = float(getattr(cad, "PP_PROKER_TAU", 1.0))
        self.mc_samples = int(getattr(cad, "PP_PROKER_MC_SAMPLES", 64))
        self.use_mc = bool(getattr(cad, "PP_PROKER_USE_MC", False))

        # If True and MC is enabled, final output becomes log posterior predictive probabilities.
        self.return_log_probs = bool(getattr(cad, "PP_PROKER_RETURN_LOG_PROBS", True))

        # Useful sanity/debug switch:
        #   1.0 = normal ProKeR residual
        #   0.0 = official ProKeR base only, i.e. cosine zero-shot ranking
        self.mean_residual_scale = float(getattr(cad, "PP_PROKER_MEAN_RESIDUAL_SCALE", 1.0))

        self.variance_jitter = float(getattr(cad, "PP_PROKER_VARIANCE_JITTER", 1e-6))

        if self.beta <= 0:
            raise ValueError(f"PP_PROKER_BETA must be positive, got {self.beta}")
        if self.lmbda <= 0:
            raise ValueError(f"PP_PROKER_LAMBDA must be positive, got {self.lmbda}")
        if self.tau <= 0:
            raise ValueError(f"PP_PROKER_TAU must be positive, got {self.tau}")
        if self.mc_samples < 1:
            raise ValueError(f"PP_PROKER_MC_SAMPLES must be >= 1, got {self.mc_samples}")

        self.register_buffer("support_features", torch.empty(0), persistent=True)
        self.register_buffer("support_labels", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("kernel_alpha", torch.empty(0), persistent=True)

        # Matrix used only for posterior variance:
        #   K_SS + delta I
        self.register_buffer("variance_matrix", torch.empty(0), persistent=True)

        self.register_buffer("kernel_beta", torch.tensor(self.beta), persistent=True)
        self.register_buffer("kernel_lambda", torch.tensor(self.lmbda), persistent=True)
        self.register_buffer("variance_delta", torch.tensor(1.0), persistent=True)

        self._is_fitted = False

    def get_prototypes(self) -> torch.Tensor:
        # Keep outer ClipAdaptersModel zero-shot path intact.
        # The outer model will compute scaled CLIP logits first.
        # cache_logits() will then subtract those scaled logits and replace them
        # with official ProKeR logits.
        return self.base_text_features

    def _cosine_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Official ProKeR CLIP logits:
            normalize(image) @ normalize(text).T

        No CLIP logit_scale.
        No hard-coded 100.
        """
        x = _normalize(features.float())
        t = _normalize(self.base_text_features.float()).to(x.device)
        return x @ t.t()

    def _outer_clip_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Exact logits produced by the outer ClipAdaptersModel + lp_logits path:
            normalize(image) @ normalize(text).T * logit_scale.exp()

        This is used only to cancel the outer model's already-added zero-shot logits.
        """
        return self._cosine_logits(features) * self.logit_scale.exp().float()

    def _rbf_kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = _normalize(a.float())
        b = _normalize(b.float())

        # Official ProKeR:
        #   exp(-beta * (1 - X @ Y.T))
        dist = 1.0 - a @ b.t()
        return torch.exp(-self.kernel_beta.to(a.device) * dist)

    def _posterior_variance(self, k_xs: torch.Tensor) -> torch.Tensor:
        """
        Diagonal GP-style predictive variance:

            s2(x) = k(x,x) - k_xS (K_SS + delta I)^-1 k_Sx

        For the normalized ProKeR RBF kernel, k(x,x)=1.
        """
        if self.variance_matrix.numel() == 0:
            return torch.ones(k_xs.shape[0], device=k_xs.device, dtype=k_xs.dtype)

        mat = self.variance_matrix.to(device=k_xs.device, dtype=k_xs.dtype)
        solved = torch.linalg.solve(mat, k_xs.t()).t()

        s2 = 1.0 - (k_xs * solved).sum(dim=1)
        return s2.clamp_min(float(self.variance_jitter))

    def _posterior_predictive_logits(
        self,
        mean_logits: torch.Tensor,
        s2: torch.Tensor,
    ) -> torch.Tensor:
        """
        If MC is disabled:
            returns mean_logits / tau

        If MC is enabled and return_log_probs=True:
            returns log mean_t softmax(sample_t / tau)

        If MC is enabled and return_log_probs=False:
            returns mean probabilities directly. This is mainly for debugging;
            for normal CE/NLL-style evaluation, return_log_probs=True is preferred.
        """
        if (not self.use_mc) or self.rho <= 0:
            return mean_logits / float(self.tau)

        std = torch.sqrt(float(self.rho) * s2).to(mean_logits.dtype)

        eps = torch.randn(
            int(self.mc_samples),
            mean_logits.shape[0],
            mean_logits.shape[1],
            device=mean_logits.device,
            dtype=mean_logits.dtype,
        )

        samples = mean_logits.unsqueeze(0) + std.view(1, -1, 1) * eps
        probs = torch.softmax(samples / float(self.tau), dim=-1).mean(dim=0)

        if self.return_log_probs:
            return torch.log(probs.clamp_min(1e-12))

        return probs

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        device = self.base_text_features.device

        features = _normalize(features_train.to(device=device, dtype=torch.float32))
        labels = labels_train.to(device=device, dtype=torch.long)

        if features.ndim != 2:
            raise ValueError(
                f"PPProKeROneHot expects support features [N, D], got {tuple(features.shape)}"
            )

        n_classes = int(self.base_text_features.shape[0])

        if labels.numel() != features.shape[0]:
            raise ValueError(
                "PPProKeROneHot feature/label count mismatch: "
                f"features={features.shape[0]}, labels={labels.numel()}"
            )

        if labels.min().item() < 0 or labels.max().item() >= n_classes:
            raise ValueError(
                f"PPProKeROneHot labels out of range: "
                f"min={labels.min().item()}, max={labels.max().item()}, "
                f"n_classes={n_classes}"
            )

        self.support_features = features.detach().clone()
        self.support_labels = labels.detach().clone()

        # Official ProKeR one-hot target.
        one_hot = F.one_hot(labels, num_classes=n_classes).to(torch.float32)

        # Official ProKeR residual:
        #   one_hot - cosine_CLIP(S)
        #
        # Important:
        # Do NOT use the outer scaled CLIP logits here.
        proker_logits_support = self._cosine_logits(features)
        residual = one_hot - proker_logits_support

        k_ss = self._rbf_kernel(features, features)
        eye = torch.eye(k_ss.shape[0], device=device, dtype=k_ss.dtype)

        # Official ProKeR-compatible mean solve:
        #   alpha = solve((1/lambda) K + I, one_hot - cosine_CLIP(S))
        solve_mat = (1.0 / float(self.lmbda)) * k_ss + eye
        self.kernel_alpha = torch.linalg.solve(solve_mat, residual).detach()

        # Separate GP-style variance path:
        #   K + delta I
        #
        # This does not alter the official ProKeR-compatible mean solve above.
        delta = float(self.gp_delta) if float(self.gp_delta) > 0 else float(self.lmbda)

        self.variance_matrix = (k_ss + delta * eye).detach()
        self.kernel_beta = torch.tensor(float(self.beta), device=device)
        self.kernel_lambda = torch.tensor(float(self.lmbda), device=device)
        self.variance_delta = torch.tensor(float(delta), device=device)

        self._is_fitted = True

        print(
            "[PP-ProKeR-OneHot] fitted: "
            f"N={features.shape[0]}, C={n_classes}, D={features.shape[1]}, "
            f"beta={float(self.kernel_beta):.6g}, "
            f"lambda={float(self.kernel_lambda):.6g}, "
            f"delta={float(self.variance_delta):.6g}, "
            f"rho={float(self.rho):.6g}, "
            f"tau={float(self.tau):.6g}, "
            f"mc={int(self.mc_samples)}, "
            f"use_mc={bool(self.use_mc)}, "
            f"return_log_probs={bool(self.return_log_probs)}, "
            f"mean_residual_scale={float(self.mean_residual_scale):.6g}"
        )

    def cache_logits(self, features: torch.Tensor):
        if not self._is_fitted or self.support_features.numel() == 0:
            return None

        # This is what the outer ClipAdaptersModel has already computed before
        # calling cache_logits().
        outer_base_logits = self._outer_clip_logits(features)

        # This is the base used by official ProKeR.
        proker_base_logits = self._cosine_logits(features)

        k_xs = self._rbf_kernel(features.float(), self.support_features.float())

        residual_logits = k_xs @ self.kernel_alpha.to(
            device=k_xs.device,
            dtype=k_xs.dtype,
        )

        # Official ProKeR-compatible mean:
        #   m(x) = cosine_CLIP(x) + K_XS alpha
        #
        # mean_residual_scale is only for sanity/debug:
        #   0.0 -> cosine zero-shot ranking
        #   1.0 -> official ProKeR mean
        mean_logits = proker_base_logits + float(self.mean_residual_scale) * residual_logits

        s2 = self._posterior_variance(k_xs)
        final_logits = self._posterior_predictive_logits(mean_logits, s2)

        # Critical adapter-interface correction:
        #
        # Outer model does:
        #   final = outer_scaled_clip_logits + cache_logits
        #
        # We want:
        #   final = official_proker_logits
        #
        # Therefore:
        #   cache_logits = official_proker_logits - outer_scaled_clip_logits
        return (final_logits - outer_base_logits).to(features.dtype)

    def reset_hparams(self, params):
        if "beta" in params:
            self.beta = float(params["beta"])
            self.kernel_beta = torch.tensor(
                float(self.beta),
                device=self.base_text_features.device,
            )

        if "lmbda" in params:
            self.lmbda = float(params["lmbda"])
            self.kernel_lambda = torch.tensor(
                float(self.lmbda),
                device=self.base_text_features.device,
            )

        if "lambda" in params:
            self.lmbda = float(params["lambda"])
            self.kernel_lambda = torch.tensor(
                float(self.lmbda),
                device=self.base_text_features.device,
            )

        if "rho" in params:
            self.rho = float(params["rho"])

        if "tau" in params:
            self.tau = float(params["tau"])

        if "mean_residual_scale" in params:
            self.mean_residual_scale = float(params["mean_residual_scale"])