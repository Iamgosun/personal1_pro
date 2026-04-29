from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _center_logits(z: torch.Tensor) -> torch.Tensor:
    return z - z.mean(dim=-1, keepdim=True)


def _median_range(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r = z.max(dim=-1).values - z.min(dim=-1).values
    return r.median().clamp_min(eps)


def _one_hot_centered(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    y = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(torch.float32)
    return y - 1.0 / float(num_classes)


class ECKAAdapter(BaseAdapter):
    """
    Evidence-Calibrated Kernel-Discriminant Adapter.

    Implementation strategy for current ClipAdapters pipeline:
      - get_prototypes() returns zero-shot text prototypes.
      - cache_logits(features) returns ECKA_logits(features) - ZS_logits(features).
      - build_cache(features_train, labels_train) performs all support-only fitting.

    This keeps ECKA compatible with ClipAdaptersModel.forward_features(),
    which already computes lp_logits(...) + adapter.cache_logits(...).
    """

    initialization_name = "ECKA"
    adapter_kind = "prototype"
    uses_cache = True
    is_ecka_adapter = True
    # 新增
    closed_form_adapter = True
    requires_training = False

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = getattr(cfg, "CLIP_ADAPTERS", None)

        self.ecka_residual_alpha = float(getattr(cad, "ECKA_RESIDUAL_ALPHA", 0.2))
        self.ecka_min_w0 = float(getattr(cad, "ECKA_MIN_W0", 0.5))
        self.ecka_replace_zs = bool(getattr(cad, "ECKA_REPLACE_ZS", False))


        self.ecka_kappa0 = float(getattr(cad, "ECKA_KAPPA0", 2.0))
        self.ecka_cov_shrink = float(getattr(cad, "ECKA_COV_SHRINK", 0.90))
        self.ecka_kernel_lambda = float(getattr(cad, "ECKA_KERNEL_LAMBDA", -1.0))
        self.ecka_beta_scale = float(getattr(cad, "ECKA_KERNEL_BETA_SCALE", 1.0))

        self.ecka_use_gda = bool(getattr(cad, "ECKA_USE_GDA", True))
        self.ecka_use_kernel = bool(getattr(cad, "ECKA_USE_KERNEL", True))
        self.ecka_use_fusion_grid = bool(getattr(cad, "ECKA_USE_FUSION_GRID", True))

        self.ecka_w0 = float(getattr(cad, "ECKA_W0", 1.0 / 3.0))
        self.ecka_wg = float(getattr(cad, "ECKA_WG", 1.0 / 3.0))
        self.ecka_wk = float(getattr(cad, "ECKA_WK", 1.0 / 3.0))
        self.ecka_temperature = float(getattr(cad, "ECKA_TEMPERATURE", 1.0))

        self.ecka_calibrate = bool(getattr(cad, "ECKA_CALIBRATE", True))
        self.ecka_uncertainty_beta = float(getattr(cad, "ECKA_UNCERTAINTY_BETA", 0.0))
        self.ecka_range_delta = float(getattr(cad, "ECKA_RANGE_DELTA", -1.0))

        self.register_buffer("mu", torch.empty(0), persistent=True)
        self.register_buffer("sigma_inv", torch.empty(0), persistent=True)
        self.register_buffer("gda_bias", torch.empty(0), persistent=True)

        self.register_buffer("support_features", torch.empty(0), persistent=True)
        self.register_buffer("support_labels", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("kernel_alpha", torch.empty(0), persistent=True)
        self.register_buffer("kernel_beta", torch.tensor(1.0), persistent=True)
        self.register_buffer("kernel_lambda", torch.tensor(1.0), persistent=True)

        self.register_buffer("scale_zs", torch.tensor(1.0), persistent=True)
        self.register_buffer("scale_gda", torch.tensor(1.0), persistent=True)
        self.register_buffer("scale_kernel", torch.tensor(1.0), persistent=True)

        self.register_buffer("fusion_weights", torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3]), persistent=True)
        self.register_buffer("temperature", torch.tensor(1.0), persistent=True)

        self._is_fitted = False

    def get_prototypes(self) -> torch.Tensor:
        # Keep outer ClipAdaptersModel zero-shot path intact.
        return self.base_text_features

    def _zs_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = _normalize(features.float())
        t = _normalize(self.base_text_features.float())
        return x @ t.t() * self.logit_scale.exp().float()

    def _scaled_zs(self, features: torch.Tensor) -> torch.Tensor:
        z = _center_logits(self._zs_logits(features))
        return z / self.scale_zs.clamp_min(1e-6)

    def _estimate_text_visual_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = features.device
        x = _normalize(features.float())
        t = _normalize(self.base_text_features.float()).to(device)

        n_classes, dim = t.shape
        mu_list = []

        zs = self._zs_logits(x)
        true_logits = zs[torch.arange(labels.numel(), device=device), labels]
        masked = zs.clone()
        masked[torch.arange(labels.numel(), device=device), labels] = -1e30
        margins = true_logits - masked.max(dim=1).values

        margin_scale = torch.median(torch.abs(margins - margins.median())).clamp_min(1e-6)

        for c in range(n_classes):
            idx = labels == c

            if idx.any():
                xc = x[idx]
                vc = _normalize(xc.mean(dim=0, keepdim=True)).squeeze(0)

                mc = margins[idx].median()
                r_text = torch.sigmoid(mc / margin_scale)

                compact = (1.0 - (xc @ vc).clamp(-1, 1)).mean()
                r_vis = torch.exp(-compact / compact.detach().clamp_min(1e-6))

                k_text = self.ecka_kappa0 * r_text
                k_vis = float(idx.sum().item()) * r_vis

                mu_c = _normalize(k_text * t[c] + k_vis * vc, eps=1e-12)
            else:
                mu_c = t[c]

            mu_list.append(mu_c)

        return torch.stack(mu_list, dim=0)

    def _fit_gda(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = _normalize(features.float())
        n, dim = x.shape

        residual = x - mu[labels]
        denom = max(1, n - int(mu.shape[0]))
        cov = residual.t() @ residual / float(denom)

        trace = torch.trace(cov).clamp_min(1e-12)
        sigma2 = trace / float(dim)

        lam = min(max(float(self.ecka_cov_shrink), 0.0), 0.999)
        cov = (1.0 - lam) * cov + lam * sigma2 * torch.eye(dim, device=x.device, dtype=x.dtype)

        sigma_inv = torch.linalg.pinv(cov)
        bias = -0.5 * torch.sum((mu @ sigma_inv) * mu, dim=1)

        return sigma_inv, bias

    def _gda_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = _normalize(features.float())
        z = x @ self.sigma_inv @ self.mu.t() + self.gda_bias.unsqueeze(0)
        z = _center_logits(z)
        return z / self.scale_gda.clamp_min(1e-6)

    def _kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = _normalize(a.float())
        b = _normalize(b.float())
        dist = 1.0 - a @ b.t()
        return torch.exp(-self.kernel_beta.to(a.device) * dist)

    def _median_kernel_beta(self, features: torch.Tensor) -> torch.Tensor:
        x = _normalize(features.float())
        if x.shape[0] <= 1:
            return torch.tensor(1.0, device=x.device)

        sim = x @ x.t()
        dist = 1.0 - sim
        mask = ~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)
        vals = dist[mask]

        med = vals.median().clamp_min(1e-6)
        beta = torch.log(torch.tensor(2.0, device=x.device)) / med
        return beta * float(self.ecka_beta_scale)

    def _fit_kernel(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        phi0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _normalize(features.float())
        n_classes = int(self.base_text_features.shape[0])
        y = _one_hot_centered(labels, n_classes).to(x.device)

        residual = y - phi0.detach()

        beta = self._median_kernel_beta(x)
        self.kernel_beta = beta.detach()

        k = self._kernel(x, x)

        if float(self.ecka_kernel_lambda) > 0:
            lam = torch.tensor(float(self.ecka_kernel_lambda), device=x.device)
        else:
            lam = self._select_kernel_lambda_gcv(k, residual)

        eye = torch.eye(k.shape[0], device=x.device, dtype=x.dtype)
        alpha = torch.linalg.solve(k + lam * eye, residual)

        return alpha, beta, lam

    def _select_kernel_lambda_gcv(self, k: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        grid = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], device=k.device)
        n = k.shape[0]
        eye = torch.eye(n, device=k.device, dtype=k.dtype)

        best_lam = grid[0]
        best_score = None

        for lam in grid:
            mat = k + lam * eye
            solved = torch.linalg.solve(mat, residual)
            pred = k @ solved

            # H = K(K+lambda I)^-1
            h = torch.linalg.solve(mat, k).t()
            tr_h = torch.trace(h)

            num = (residual - pred).pow(2).sum()
            den = (float(n) - tr_h).pow(2).clamp_min(1e-6)
            score = num / den

            if best_score is None or score < best_score:
                best_score = score
                best_lam = lam

        return best_lam.detach()

    def _kernel_logits(self, features: torch.Tensor) -> torch.Tensor:
        phi0 = self._scaled_zs(features)
        kx = self._kernel(features.float(), self.support_features.float())
        z = phi0 + kx @ self.kernel_alpha
        z = _center_logits(z)
        return z / self.scale_kernel.clamp_min(1e-6)

    def _compose_logits(self, features: torch.Tensor) -> torch.Tensor:
        phi0 = self._scaled_zs(features)

        if self.ecka_use_gda:
            phig = self._gda_logits(features)
        else:
            phig = torch.zeros_like(phi0)

        if self.ecka_use_kernel:
            phik = self._kernel_logits(features)
        else:
            phik = torch.zeros_like(phi0)

        w = self.fusion_weights.to(features.device, dtype=phi0.dtype)
        z = w[0] * phi0 + w[1] * phig + w[2] * phik

        if self.ecka_calibrate:
            z = self._apply_range_guard(z, phi0)

        return z / self.temperature.to(features.device).clamp_min(1e-6)

    def _apply_range_guard(self, z: torch.Tensor, phi0: torch.Tensor) -> torch.Tensor:
        delta = float(self.ecka_range_delta)

        # delta <= 0 means disabled.
        if delta <= 0:
            return z

        rz = (z.max(dim=-1, keepdim=True).values - z.min(dim=-1, keepdim=True).values).clamp_min(1e-6)
        r0 = (phi0.max(dim=-1, keepdim=True).values - phi0.min(dim=-1, keepdim=True).values).clamp_min(1e-6)

        factor = torch.minimum(torch.ones_like(rz), delta * r0 / rz)
        return z.mean(dim=-1, keepdim=True) + factor * (z - z.mean(dim=-1, keepdim=True))

    def _fit_fusion_grid(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Lightweight support-only grid.
        # This uses in-sample evidence in v0. For stricter paper experiments,
        # replace this with true LOO / augmentation hold-out.
        phi0 = self._scaled_zs(features)
        phig = self._gda_logits(features) if self.ecka_use_gda else torch.zeros_like(phi0)
        phik = self._kernel_logits(features) if self.ecka_use_kernel else torch.zeros_like(phi0)

        candidates = []
        vals = torch.linspace(0, 1, 11, device=features.device)
        min_w0 = float(self.ecka_min_w0)
        for w0 in vals:
            if float(w0) < min_w0:
                continue

            for wg in vals:
                wk = 1.0 - w0 - wg
                if wk < -1e-6:
                    continue
                if wk < 0:
                    wk = torch.tensor(0.0, device=features.device)
                candidates.append(torch.stack([w0, wg, wk]))

        temps = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], device=features.device)

        best_loss = None
        best_w = candidates[0]
        best_t = temps[2]

        for w in candidates:
            z = w[0] * phi0 + w[1] * phig + w[2] * phik
            for t in temps:
                loss = F.cross_entropy(z / t, labels)
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_w = w.detach()
                    best_t = t.detach()

        return best_w, best_t

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        device = self.base_text_features.device

        features = _normalize(features_train.to(device=device, dtype=torch.float32))
        labels = labels_train.to(device=device, dtype=torch.long)

        if features.ndim != 2:
            raise ValueError(f"ECKA expects support features [N, D], got {tuple(features.shape)}")

        n_classes = int(self.base_text_features.shape[0])
        if labels.min().item() < 0 or labels.max().item() >= n_classes:
            raise ValueError(
                f"ECKA labels out of range: min={labels.min().item()}, "
                f"max={labels.max().item()}, n_classes={n_classes}"
            )

        self.support_features = features.detach().clone()
        self.support_labels = labels.detach().clone()

        # Branch 0: zero-shot normalized score.
        zs_centered = _center_logits(self._zs_logits(features))
        self.scale_zs = _median_range(zs_centered).detach()

        phi0 = self._scaled_zs(features)

        # Branch G: text-shrinkage GDA.
        self.mu = self._estimate_text_visual_prototypes(features, labels).detach()
        self.sigma_inv, self.gda_bias = self._fit_gda(features, labels, self.mu)
        self.sigma_inv = self.sigma_inv.detach()
        self.gda_bias = self.gda_bias.detach()

        gda_centered = _center_logits(features @ self.sigma_inv @ self.mu.t() + self.gda_bias.unsqueeze(0))
        self.scale_gda = _median_range(gda_centered).detach()

        # Branch K: proximal kernel residual.
        self.kernel_alpha, beta, lam = self._fit_kernel(features, labels, phi0)
        self.kernel_alpha = self.kernel_alpha.detach()
        self.kernel_beta = beta.detach()
        self.kernel_lambda = lam.detach()

        kernel_centered = _center_logits(phi0 + self._kernel(features, features) @ self.kernel_alpha)
        self.scale_kernel = _median_range(kernel_centered).detach()

        # Support-only fusion.
        if self.ecka_use_fusion_grid:
            w, temp = self._fit_fusion_grid(features, labels)
            self.fusion_weights = w.detach()
            self.temperature = temp.detach()
        else:
            w = torch.tensor(
                [self.ecka_w0, self.ecka_wg, self.ecka_wk],
                device=device,
                dtype=torch.float32,
            )
            w = w.clamp_min(0)
            w = w / w.sum().clamp_min(1e-6)
            self.fusion_weights = w.detach()
            self.temperature = torch.tensor(float(self.ecka_temperature), device=device)

        self._is_fitted = True

        print(
            "[ECKA] fitted: "
            f"N={features.shape[0]}, C={n_classes}, D={features.shape[1]}, "
            f"beta={float(self.kernel_beta):.6g}, "
            f"lambda={float(self.kernel_lambda):.6g}, "
            f"scale_zs={float(self.scale_zs):.6g}, "
            f"scale_gda={float(self.scale_gda):.6g}, "
            f"scale_kernel={float(self.scale_kernel):.6g}, "
            f"w={self.fusion_weights.detach().cpu().tolist()}, "
            f"T={float(self.temperature):.6g}"
        )

    def cache_logits(self, features):
        if not self._is_fitted or self.support_features.numel() == 0:
            return None

        zs_raw = self._zs_logits(features)
        phi0 = self._scaled_zs(features)
        ecka_norm = self._compose_logits(features)

        if bool(self.ecka_replace_zs):
            # Old behavior: final logits = ECKA logits.
            # Keep only for ablation, not default.
            return ecka_norm.to(zs_raw.dtype) - zs_raw

        residual_norm = ecka_norm - phi0

        # Map normalized residual back to roughly CLIP logit scale.
        residual_raw = (
            float(self.ecka_residual_alpha)
            * self.scale_zs.to(features.device).clamp_min(1e-6)
            * residual_norm
        )

        return residual_raw.to(zs_raw.dtype)