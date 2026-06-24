from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def _class_centroids(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Compute class visual prototypes from normalized support features."""
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be [N], got {tuple(labels.shape)}")
    if features.shape[0] != labels.numel():
        raise ValueError(
            f"features/labels mismatch: features={features.shape[0]}, labels={labels.numel()}"
        )

    n, d = features.shape
    labels = labels.to(torch.long)
    centroids = torch.zeros(n_classes, d, device=features.device, dtype=features.dtype)
    counts = torch.zeros(n_classes, 1, device=features.device, dtype=features.dtype)

    centroids.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones(n, 1, device=features.device, dtype=features.dtype))

    # Keep absent classes at zero instead of producing NaN. In normal few-shot training
    # each class should be present; this fallback makes error diagnosis safer.
    centroids = centroids / counts.clamp_min(1.0)
    return centroids


@torch.no_grad()
def _estimate_lr_w(features: torch.Tensor) -> float:
    """
    LP++ data-driven visual-classifier step size.

    Official LP++ computes:
        lr_w = 4N / lambda_max(X^T X)

    We use torch.linalg.eigvalsh to avoid scipy dependency.
    """
    x = features.float()
    xtx = x.t() @ x
    eigvals = torch.linalg.eigvalsh(xtx)
    max_eig = float(eigvals.max().clamp_min(1e-12).item())
    return float(4.0 * x.shape[0] / max_eig)


@torch.no_grad()
def _estimate_lr_alpha(features: torch.Tensor, text_features: torch.Tensor) -> float:
    """
    LP++ data-driven alpha step size.

    Official LP++ computes:
        lr_alpha = N / (4 * max_c sum_i (x_i^T t_c)^2)
    """
    x = _normalize(features.float())
    t = _normalize(text_features.float()).to(x.device)
    text_logits = x @ t.t()
    denom = 4.0 * float(text_logits.pow(2).sum(dim=0).max().clamp_min(1e-12).item())
    return float(x.shape[0] / denom)


@torch.no_grad()
def _estimate_init_alpha(
    features: torch.Tensor,
    labels: torch.Tensor,
    shots: Optional[int],
    text_features: torch.Tensor,
) -> float:
    """
    LP++ data-informed alpha initialization.

    Official code:
        alpha_tilde = compute_centroids_alpha((features @ clip_weights), labels)
        alpha_tilde = alpha_tilde * shots
        alpha_init = 250 / shots * alpha_tilde
        final_init_alpha_mean = mean(alpha_init)

    For class c we use the support mean of the correct text logit x_i^T t_c.
    """
    x = _normalize(features.float())
    t = _normalize(text_features.float()).to(x.device)
    labels = labels.to(torch.long)
    n_classes = int(t.shape[0])

    text_logits = x @ t.t()
    values = []
    for c in range(n_classes):
        mask = labels == c
        if bool(mask.any()):
            values.append(text_logits[mask, c].mean())

    if not values:
        return 0.0

    alpha_tilde_mean = torch.stack(values).mean()

    if shots is None or int(shots) <= 0:
        # shots cancels algebraically in the official expression when every class
        # has the same shot count; keep this fallback for uneven support sets.
        return float((250.0 * alpha_tilde_mean).item())

    shots_f = float(int(shots))
    alpha_init = (250.0 / shots_f) * (alpha_tilde_mean * shots_f)
    return float(alpha_init.item())


class LPPlusPlusAdapter(BaseAdapter):
    """
    LP++ adapter for this project's ClipAdapters framework.

    Formula implemented at train/eval:
        logits(x) = Linear(normalize(x); W_v, b_v)
                    + alpha_c * <normalize(x), normalize(t_c)>

    Integration detail:
    ClipAdaptersModel computes outer zero-shot cosine logits first through
    lp_logits(...). This adapter returns additive cache_logits that cancel the
    outer logits and replace them with LP++ logits.

    This follows the LP++ paper/code logic:
    - visual classifier initialized by support visual centroids;
    - text branch uses CLIP text features;
    - class-wise alpha starts from data-informed initialization;
    - optional gradient hooks approximate LP++ data-driven learning rates inside
      the existing trainer/optimizer without introducing a new trainer.
    """

    initialization_name = "LP_PLUS_PLUS"
    adapter_kind = "prototype"
    uses_cache = True
    needs_support_features = True
    requires_training = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = getattr(cfg, "CLIP_ADAPTERS", None)

        self.feat_dim = int(base_text_features.shape[-1])
        self.n_classes = int(base_text_features.shape[0])

        self.use_bias = bool(getattr(cad, "LPPLUS_USE_BIAS", True))
        self.text_scale = float(getattr(cad, "LPPLUS_TEXT_SCALE", 1.0))
        self.alpha_trainable = bool(getattr(cad, "LPPLUS_ALPHA_TRAINABLE", True))
        self.use_data_lr_hooks = bool(getattr(cad, "LPPLUS_USE_DATA_LR_HOOKS", True))
        self.grad_scale_clip = float(getattr(cad, "LPPLUS_GRAD_SCALE_CLIP", 1.0e4))
        self.default_shots = int(getattr(getattr(cfg, "DATASET", None), "NUM_SHOTS", 0))

        # Official LP++ initializes W_v with visual centroids after feature extraction.
        # Before build_cache(), keep a safe zero-shot-shaped parameter so the model can
        # still run feature prefit/cache extraction.
        self.visual_weight = nn.Parameter(base_text_features.detach().clone().float())
        self.visual_bias = nn.Parameter(torch.zeros(self.n_classes, dtype=torch.float32))

        # Official code initializes a class-wise alpha vector from a scalar mean.
        self.alpha_vec = nn.Parameter(torch.zeros(1, self.n_classes, dtype=torch.float32))
        self.alpha_vec.requires_grad_(self.alpha_trainable)

        self.register_buffer("lpplus_lr_w", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("lpplus_lr_alpha", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("lpplus_grad_scale_w", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("lpplus_grad_scale_alpha", torch.tensor(1.0, dtype=torch.float32))

        self._is_fitted = False
        self._register_gradient_hooks()

        print(
            "Using LP++ adapter: visual centroid classifier + class-wise text blending "
            f"(bias={self.use_bias}, data_lr_hooks={self.use_data_lr_hooks})"
        )

    def _register_gradient_hooks(self) -> None:
        def scale_w_grad(grad: torch.Tensor) -> torch.Tensor:
            if not self.use_data_lr_hooks:
                return grad
            scale = self.lpplus_grad_scale_w.to(device=grad.device, dtype=grad.dtype)
            scale = scale.clamp(min=0.0, max=float(self.grad_scale_clip))
            return grad * scale

        def scale_alpha_grad(grad: torch.Tensor) -> torch.Tensor:
            if not self.use_data_lr_hooks:
                return grad
            scale = self.lpplus_grad_scale_alpha.to(device=grad.device, dtype=grad.dtype)
            scale = scale.clamp(min=0.0, max=float(self.grad_scale_clip))
            return grad * scale

        self.visual_weight.register_hook(scale_w_grad)
        self.visual_bias.register_hook(scale_w_grad)
        self.alpha_vec.register_hook(scale_alpha_grad)

    def get_prototypes(self) -> torch.Tensor:
        # Let the outer model compute its usual zero-shot logits; cache_logits()
        # will cancel and replace them by LP++ logits.
        return self.base_text_features

    def _features_for_lp(self, features: torch.Tensor) -> torch.Tensor:
        return _normalize(features.float())

    def _text_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = self._features_for_lp(features)
        t = _normalize(self.base_text_features.float()).to(x.device)
        return (x @ t.t()) * float(self.text_scale)

    def _outer_clip_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = self._features_for_lp(features)
        t = _normalize(self.base_text_features.float()).to(x.device)
        return x @ t.t() * self.logit_scale.exp().float()

    def _vision_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = self._features_for_lp(features)
        weight = self.visual_weight.to(device=x.device, dtype=x.dtype)
        bias = self.visual_bias.to(device=x.device, dtype=x.dtype) if self.use_bias else None
        return F.linear(x, weight, bias)

    def lpplus_logits(self, features: torch.Tensor) -> torch.Tensor:
        vision_logits = self._vision_logits(features)
        text_logits = self._text_logits(features)
        alpha = self.alpha_vec.to(device=vision_logits.device, dtype=vision_logits.dtype)
        return vision_logits + alpha * text_logits

    def cache_logits(self, features: torch.Tensor, base_logits=None):
        # Replace the outer zero-shot logits by LP++ logits.
        final_logits = self.lpplus_logits(features)
        if base_logits is None:
            base_logits = self._outer_clip_logits(features)
        return (final_logits - base_logits.float()).to(features.dtype)

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        device = self.base_text_features.device

        features = self._features_for_lp(features_train.to(device=device, dtype=torch.float32))
        labels = labels_train.to(device=device, dtype=torch.long)

        if labels.numel() != features.shape[0]:
            raise ValueError(
                "LP++ feature/label count mismatch: "
                f"features={features.shape[0]}, labels={labels.numel()}"
            )
        if labels.numel() == 0:
            raise ValueError("LP++ build_cache received an empty support set.")
        if labels.min().item() < 0 or labels.max().item() >= self.n_classes:
            raise ValueError(
                f"LP++ labels out of range: min={labels.min().item()}, "
                f"max={labels.max().item()}, n_classes={self.n_classes}"
            )

        centroids = _class_centroids(features, labels, self.n_classes)
        self.visual_weight.copy_(centroids.to(device=device, dtype=self.visual_weight.dtype))
        self.visual_bias.zero_()

        counts = torch.bincount(labels, minlength=self.n_classes)
        positive_counts = counts[counts > 0]
        inferred_shots = (
            int(positive_counts.float().median().item())
            if positive_counts.numel() > 0
            else int(self.default_shots)
        )
        shots = int(self.default_shots) if int(self.default_shots) > 0 else inferred_shots

        alpha_init = _estimate_init_alpha(
            features=features,
            labels=labels,
            shots=shots,
            text_features=self.base_text_features,
        )
        self.alpha_vec.fill_(float(alpha_init))

        lr_w = _estimate_lr_w(features)
        lr_alpha = _estimate_lr_alpha(features, self.base_text_features)
        self.lpplus_lr_w.fill_(float(lr_w))
        self.lpplus_lr_alpha.fill_(float(lr_alpha))

        base_lr = float(getattr(getattr(self.cfg, "OPTIM", None), "LR", 1.0))
        if not math.isfinite(base_lr) or base_lr <= 0:
            base_lr = 1.0

        self.lpplus_grad_scale_w.fill_(float(lr_w / base_lr))
        self.lpplus_grad_scale_alpha.fill_(float(lr_alpha / base_lr))

        self._is_fitted = True

        print(
            "[LP++] fitted from support features: "
            f"N={features.shape[0]}, C={self.n_classes}, D={features.shape[1]}, "
            f"shots={shots}, alpha_init={alpha_init:.6g}, "
            f"lr_w={lr_w:.6g}, lr_alpha={lr_alpha:.6g}, "
            f"base_lr={base_lr:.6g}, "
            f"grad_scale_w={float(self.lpplus_grad_scale_w.item()):.6g}, "
            f"grad_scale_alpha={float(self.lpplus_grad_scale_alpha.item()):.6g}"
        )

    def reset_hparams(self, params):
        if "text_scale" in params:
            self.text_scale = float(params["text_scale"])
        if "alpha_trainable" in params:
            self.alpha_trainable = bool(params["alpha_trainable"])
            self.alpha_vec.requires_grad_(self.alpha_trainable)
        if "use_data_lr_hooks" in params:
            self.use_data_lr_hooks = bool(params["use_data_lr_hooks"])
        if "grad_scale_clip" in params:
            self.grad_scale_clip = float(params["grad_scale_clip"])
