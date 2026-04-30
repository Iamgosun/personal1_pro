from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .bayes_adapter import BayesAdapter


class DreamBayesAdapter(BayesAdapter):
    """
    DREAM-BayesAdapter: BayesAdapter + text-anchored hyperspherical density-ratio evidence.

    This class deliberately inherits the original BayesAdapter parameterization and keeps
    initialization_name == "BAYES_ADAPTER" so the existing ClipAdapters loss path still
    uses the BayesAdapter Monte-Carlo logits and KL term.

    Added behavior:
      1. build_cache(features_train, labels_train): fit a lightweight tangent-space
         shrinkage Gaussian density-ratio head from support CLIP features.
      2. cache_logits(features): add lambda * standardized density-ratio logits at eval.
      3. postprocess_logits(logits, features): optionally apply an ID evidence gate:
            p_final = g * softmax(logits) + (1-g) / K.

    The module is designed as a safe extension: when density fitting is unavailable, when
    DREAM_LAMBDA is 0, or before build_cache() is called, it returns BayesAdapter behavior.
    """

    # Keep exact name for existing model._is_bayes_adapter() and ClipAdaptersLoss checks.
    initialization_name = "BAYES_ADAPTER"
    dream_initialization_name = "DREAM_BAYES_ADAPTER"
    needs_support_features = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        c, d = base_text_features.shape
        self.dream_enabled = bool(getattr(cfg.CLIP_ADAPTERS, "DREAM_ENABLED", True))
        self.dream_rank = int(getattr(cfg.CLIP_ADAPTERS, "DREAM_RANK", 32))
        self.dream_chunk_classes = int(getattr(cfg.CLIP_ADAPTERS, "DREAM_CHUNK_CLASSES", 64))
        self.dream_eps = float(getattr(cfg.CLIP_ADAPTERS, "DREAM_EPS", 1e-4))
        self.dream_text_prior_strength = float(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_TEXT_PRIOR_STRENGTH", 4.0)
        )
        self.dream_mean_prior_strength = float(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_MEAN_PRIOR_STRENGTH", 4.0)
        )
        self.dream_cov_prior_strength = float(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_COV_PRIOR_STRENGTH", 16.0)
        )
        self.dream_density_on_train = bool(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_DENSITY_ON_TRAIN", False)
        )
        self.dream_apply_gate = bool(getattr(cfg.CLIP_ADAPTERS, "DREAM_APPLY_GATE", True))
        self.dream_gate_requires_positive_lambda = bool(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_GATE_REQUIRES_POSITIVE_LAMBDA", True)
        )
        self.dream_gate_on_train = bool(getattr(cfg.CLIP_ADAPTERS, "DREAM_GATE_ON_TRAIN", False))
        self.dream_gate_a = float(getattr(cfg.CLIP_ADAPTERS, "DREAM_GATE_A", 5.0))
        self.dream_gate_quantile = float(getattr(cfg.CLIP_ADAPTERS, "DREAM_GATE_QUANTILE", 0.05))
        self.dream_orthogonal_gamma = float(
            getattr(cfg.CLIP_ADAPTERS, "DREAM_ORTHOGONAL_GAMMA", 0.05)
        )
        self.dream_lambda_beta = float(getattr(cfg.CLIP_ADAPTERS, "DREAM_LAMBDA_BETA", 0.01))
        self.dream_manual_lambda = float(getattr(cfg.CLIP_ADAPTERS, "DREAM_LAMBDA", -1.0))
        self.dream_debug = bool(getattr(cfg.CLIP_ADAPTERS, "DREAM_DEBUG", True))

        # Fitted state. Empty buffers make the module state_dict-safe before fitting.
        self.register_buffer("dream_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dream_class_centers", torch.empty(c, d))
        self.register_buffer("dream_basis", torch.empty(d, 0))
        self.register_buffer("dream_class_basis", torch.empty(c, d, 0))
        self.register_buffer("dream_mu", torch.empty(c, 0))
        self.register_buffer("dream_var", torch.empty(c, 0))
        self.register_buffer("dream_var0", torch.empty(0))
        self.register_buffer("dream_lambda", torch.tensor(0.0))
        self.register_buffer("dream_gate_q", torch.tensor(0.0))
        self.register_buffer("dream_gate_iqr", torch.tensor(1.0))

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _parse_lambda_grid(value) -> List[float]:
        if value is None:
            return [0.0, 0.1, 0.25, 0.5, 1.0]
        if isinstance(value, str):
            cleaned = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
            return [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        if isinstance(value, Iterable):
            return [float(x) for x in value]
        return [float(value)]

    def _log_map(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Sphere log-map Log_m(z). z: [..., D], m: broadcastable [..., D]."""
        dot = (z * m).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(dot)
        sin_theta = torch.sqrt((1.0 - dot.pow(2)).clamp_min(1e-12))
        factor = theta / sin_theta.clamp_min(1e-6)
        return factor * (z - dot * m)

    def _build_centers(self, z: torch.Tensor, y: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        centers = []
        for k in range(self.num_classes):
            idx = y == k
            if bool(idx.any()):
                visual_sum = z.index_select(0, idx.nonzero(as_tuple=False).flatten()).sum(dim=0)
            else:
                visual_sum = torch.zeros_like(text[k])
            center = self.dream_text_prior_strength * text[k] + visual_sum
            centers.append(self._normalize(center.unsqueeze(0)).squeeze(0))
        return torch.stack(centers, dim=0)

    def _build_shared_basis(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        text: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        rows = []
        if int(z.shape[0]) > 0:
            m_y = centers.index_select(0, y)
            rows.append(self._log_map(z, m_y))

        text_dir = text - text.mean(dim=0, keepdim=True)
        rows.append(text_dir)

        a = torch.cat(rows, dim=0).float()
        a = a - a.mean(dim=0, keepdim=True)
        max_rank = max(1, min(int(self.dream_rank), int(a.shape[0]), int(a.shape[1])))

        # A is [M, D]. The right singular vectors are feature-space directions [D, r].
        try:
            _, _, vh = torch.linalg.svd(a, full_matrices=False)
            basis = vh[:max_rank].transpose(0, 1).contiguous()
        except RuntimeError:
            # Very small or numerically degenerate support set: fall back to text directions.
            q, _ = torch.linalg.qr(text_dir.transpose(0, 1), mode="reduced")
            basis = q[:, :max_rank].contiguous()

        return self._normalize(basis.transpose(0, 1)).transpose(0, 1).contiguous()

    def _project_basis_to_class_tangent(
        self, basis: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        class_basis = []
        for k in range(self.num_classes):
            m = centers[k]
            pk_u = basis - m.unsqueeze(-1) * (m.unsqueeze(0) @ basis)
            q, _ = torch.linalg.qr(pk_u.float(), mode="reduced")
            class_basis.append(q[:, : basis.shape[1]].to(basis.dtype))
        return torch.stack(class_basis, dim=0).contiguous()

    def _coords_for_labels(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        centers: torch.Tensor,
        class_basis: torch.Tensor,
    ) -> torch.Tensor:
        coords = []
        for i in range(int(z.shape[0])):
            k = int(y[i].item())
            u = self._log_map(z[i : i + 1], centers[k : k + 1])
            x = u @ class_basis[k]
            coords.append(x.squeeze(0))
        return torch.stack(coords, dim=0) if coords else z.new_empty(0, class_basis.shape[-1])

    def _fit_shrinkage_gaussians(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = int(x.shape[-1])
        if int(x.shape[0]) <= 1:
            var0 = x.new_ones(r)
        else:
            var0 = x.var(dim=0, unbiased=False).clamp_min(0.0) + self.dream_eps

        mu = x.new_zeros(self.num_classes, r)
        var = x.new_zeros(self.num_classes, r)
        for k in range(self.num_classes):
            idx = y == k
            n_k = int(idx.sum().item())
            if n_k > 0:
                x_k = x.index_select(0, idx.nonzero(as_tuple=False).flatten())
                bar = x_k.mean(dim=0)
                mu_k = (float(n_k) / (float(n_k) + self.dream_mean_prior_strength)) * bar
                ss = (x_k - mu_k.unsqueeze(0)).pow(2).sum(dim=0)
                var_k = (
                    self.dream_cov_prior_strength * var0 + ss
                ) / (self.dream_cov_prior_strength + float(n_k))
                mu[k] = mu_k
                var[k] = var_k + self.dream_eps
            else:
                mu[k] = 0.0
                var[k] = var0

        return mu, var, var0

    def _density_ratio_from_fitted(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.dream_fitted.item()) or self.dream_basis.numel() == 0:
            b = int(features.shape[0])
            return features.new_zeros(b, self.num_classes), features.new_zeros(b, self.num_classes)

        z = self._normalize(features.float())
        centers = self.dream_class_centers.to(device=z.device, dtype=z.dtype)
        class_basis = self.dream_class_basis.to(device=z.device, dtype=z.dtype)
        mu = self.dream_mu.to(device=z.device, dtype=z.dtype)
        var = self.dream_var.to(device=z.device, dtype=z.dtype).clamp_min(self.dream_eps)
        var0 = self.dream_var0.to(device=z.device, dtype=z.dtype).clamp_min(self.dream_eps)

        b = int(z.shape[0])
        c = int(centers.shape[0])
        r = int(var0.shape[0])
        out = z.new_empty(b, c)
        rho_out = z.new_empty(b, c)
        chunk = max(1, int(self.dream_chunk_classes))

        logdet0 = torch.log(var0).sum()
        perp_scale = var0.mean().clamp_min(self.dream_eps)

        for start in range(0, c, chunk):
            end = min(c, start + chunk)
            m = centers[start:end]                       # [Cc, D]
            u = self._log_map(z[:, None, :], m[None, :, :])  # [B, Cc, D]
            uk = class_basis[start:end]                  # [Cc, D, R]
            x = torch.einsum("bcd,cdr->bcr", u, uk)      # [B, Cc, R]

            mu_k = mu[start:end].unsqueeze(0)            # [1, Cc, R]
            var_k = var[start:end].unsqueeze(0)          # [1, Cc, R]
            var0_k = var0.view(1, 1, r)

            quad_p = (x - mu_k).pow(2).div(var_k).sum(dim=-1)
            quad_q = x.pow(2).div(var0_k).sum(dim=-1)
            logdet = torch.log(var[start:end]).sum(dim=-1).unsqueeze(0) - logdet0
            ratio = -0.5 * (quad_p - quad_q + logdet)

            # Orthogonal residual catches OOD drift outside the low-rank semantic manifold.
            u_norm2 = u.pow(2).sum(dim=-1)
            x_norm2 = x.pow(2).sum(dim=-1)
            rho = (u_norm2 - x_norm2).clamp_min(0.0)
            if self.dream_orthogonal_gamma > 0:
                ratio = ratio - float(self.dream_orthogonal_gamma) * rho / (2.0 * perp_scale)

            out[:, start:end] = ratio
            rho_out[:, start:end] = rho

        return out, rho_out

    def _standardize_density(self, d: torch.Tensor) -> torch.Tensor:
        mean = d.mean(dim=-1, keepdim=True)
        std = d.std(dim=-1, unbiased=False, keepdim=True).clamp_min(self.dream_eps)
        return (d - mean) / std

    def _select_lambda(self, features: torch.Tensor, labels: torch.Tensor, d: torch.Tensor) -> float:
        if self.dream_manual_lambda >= 0.0:
            return float(self.dream_manual_lambda)

        grid = self._parse_lambda_grid(getattr(self.cfg.CLIP_ADAPTERS, "DREAM_LAMBDA_GRID", None))
        if not grid:
            return 0.0

        z = self._normalize(features.float())
        proto = self._normalize(self.get_prototypes().detach().float()).to(z.device)
        base_logits = z @ proto.t() * self.logit_scale.exp().detach().float().to(z.device)
        d_tilde = self._standardize_density(d.detach())

        best_lmbda = 0.0
        best_obj = None
        for lmbda in grid:
            logits = base_logits + float(lmbda) * d_tilde
            nll = F.cross_entropy(logits, labels, reduction="mean")
            obj = nll + self.dream_lambda_beta * float(lmbda) * float(lmbda)
            obj_value = float(obj.detach().cpu().item())
            if best_obj is None or obj_value < best_obj:
                best_obj = obj_value
                best_lmbda = float(lmbda)
        return best_lmbda

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        if not self.dream_enabled:
            return None

        device = self.base_text_features.device
        z_raw = features_train.detach().to(device=device, dtype=torch.float32)
        y_raw = labels_train.detach().to(device=device, dtype=torch.long).flatten()
        valid = (y_raw >= 0) & (y_raw < self.num_classes)
        z_raw = z_raw.index_select(0, valid.nonzero(as_tuple=False).flatten())
        y = y_raw.index_select(0, valid.nonzero(as_tuple=False).flatten())

        if int(z_raw.shape[0]) == 0:
            if self.dream_debug:
                print("[DREAM-BayesAdapter] no valid support features; fallback to BayesAdapter")
            self.dream_fitted.fill_(False)
            return None

        z = self._normalize(z_raw)
        text = self._normalize(self.base_text_features.detach().float().to(device))

        centers = self._build_centers(z, y, text)
        basis = self._build_shared_basis(z, y, text, centers)
        class_basis = self._project_basis_to_class_tangent(basis, centers)
        x = self._coords_for_labels(z, y, centers, class_basis)
        mu, var, var0 = self._fit_shrinkage_gaussians(x, y)

        self.dream_class_centers = centers.to(self.base_text_features.dtype)
        self.dream_basis = basis.to(self.base_text_features.dtype)
        self.dream_class_basis = class_basis.to(self.base_text_features.dtype)
        self.dream_mu = mu.to(self.base_text_features.dtype)
        self.dream_var = var.to(self.base_text_features.dtype)
        self.dream_var0 = var0.to(self.base_text_features.dtype)
        self.dream_fitted.fill_(True)

        d, _ = self._density_ratio_from_fitted(z_raw)
        lmbda = self._select_lambda(z_raw, y, d)
        self.dream_lambda = torch.tensor(lmbda, device=device, dtype=self.base_text_features.dtype)

        r_score = torch.logsumexp(d - torch.log(d.new_tensor(float(self.num_classes))), dim=-1)
        q = torch.quantile(r_score.float(), float(self.dream_gate_quantile)).to(device)
        q25 = torch.quantile(r_score.float(), 0.25).to(device)
        q75 = torch.quantile(r_score.float(), 0.75).to(device)
        iqr = (q75 - q25).abs().clamp_min(self.dream_eps)
        self.dream_gate_q = q.to(self.base_text_features.dtype)
        self.dream_gate_iqr = iqr.to(self.base_text_features.dtype)

        if self.dream_debug:
            counts = torch.bincount(y, minlength=self.num_classes)
            nonempty = int((counts > 0).sum().item())
            print(
                "[DREAM-BayesAdapter] fitted density head: "
                f"support={int(z.shape[0])}, classes={nonempty}/{self.num_classes}, "
                f"rank={int(self.dream_basis.shape[1])}, lambda={float(self.dream_lambda):.4g}, "
                f"gate_q={float(self.dream_gate_q):.4g}, gate_iqr={float(self.dream_gate_iqr):.4g}"
            )
        return None

    def density_ratio(self, features: torch.Tensor) -> torch.Tensor:
        d, _ = self._density_ratio_from_fitted(features)
        return d

    def dream_scores(self, features: torch.Tensor) -> dict:
        d, rho = self._density_ratio_from_fitted(features)
        r_score = torch.logsumexp(d - torch.log(d.new_tensor(float(self.num_classes))), dim=-1)
        gate = torch.sigmoid(
            float(self.dream_gate_a)
            * (r_score - self.dream_gate_q.to(r_score.device, r_score.dtype))
            / self.dream_gate_iqr.to(r_score.device, r_score.dtype).clamp_min(self.dream_eps)
        )
        return {
            "density_ratio": d,
            "orthogonal_residual": rho,
            "id_evidence": r_score,
            "gate": gate,
            "ood_score": -r_score,
        }

    def cache_logits(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.dream_enabled or not bool(self.dream_fitted.item()):
            return None
        if self.training and not self.dream_density_on_train:
            return None

        lmbda = float(self.dream_lambda.detach().cpu().item())
        if abs(lmbda) <= 0.0:
            return None

        d = self.density_ratio(features)
        return (lmbda * self._standardize_density(d)).to(features.dtype)

    def postprocess_logits(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        if not self.dream_enabled or not self.dream_apply_gate:
            return logits
        if not bool(self.dream_fitted.item()):
            return logits
        if self.dream_gate_requires_positive_lambda and abs(float(self.dream_lambda.detach().cpu().item())) <= 0.0:
            return logits
        if training and not self.dream_gate_on_train:
            return logits

        scores = self.dream_scores(features)
        gate = scores["gate"].to(device=logits.device, dtype=logits.dtype).unsqueeze(-1)
        probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        uniform = probs.new_full(probs.shape, 1.0 / float(self.num_classes))
        final_probs = (gate * probs + (1.0 - gate) * uniform).clamp_min(1e-12)
        return torch.log(final_probs).to(logits.dtype)
