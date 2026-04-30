from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .bayes_adapter import BayesAdapter


class DreamBayesAdapter(BayesAdapter):
    """
    DREAM-BayesAdapter v4.1.

    Strict v4.1 semantics:
      - BayesAdapter remains the classifier-parameter uncertainty base.
      - The base classification logits at eval are posterior predictive
        log-probabilities: b_k(z) = log E_W[softmax_k(logits(W))].
      - DREAM fits a text-anchored hyperspherical tangent density-ratio module.
      - Classification receives only a top-M bounded residual with one constant
        lambda0 selected from support or fixed by config.
      - ID evidence R(z) is decoupled and used only for selective ranking / OOD.
      - No adaptive lambda(z) and no probability-space gate mixture are used.

    Required model.py integration:
      In the stochastic BayesAdapter branch, call bayes_base_logits_from_mc(logits_all)
      when present, and pass the resulting base logits into cache_logits(...,
      base_logits=logits). This guarantees that top-M candidates are selected
      from the same BayesAdapter posterior predictive base used by the final logits.
    """

    # Keep this value so ClipAdaptersModel._is_bayes_adapter() treats DREAM as a
    # stochastic BayesAdapter branch and exposes Bayes KL / MC behavior.
    initialization_name = "BAYES_ADAPTER"
    dream_initialization_name = "DREAM_BAYES_ADAPTER_V41"
    needs_support_features = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = cfg.CLIP_ADAPTERS
        c, d = base_text_features.shape

        self.dream_enabled = bool(
            getattr(cad, "DREAM_V41_ENABLED", getattr(cad, "DREAM_ENABLED", True))
        )
        self.dream_rank = int(getattr(cad, "DREAM_V41_RANK", getattr(cad, "DREAM_RANK", 32)))
        self.dream_chunk_classes = int(
            getattr(cad, "DREAM_V41_CHUNK_CLASSES", getattr(cad, "DREAM_CHUNK_CLASSES", 64))
        )

        self.eps_theta = float(getattr(cad, "DREAM_V41_EPS_THETA", 1.0e-6))
        self.eps_p = float(getattr(cad, "DREAM_V41_EPS_P", 1.0e-12))
        self.eps_sigma = float(getattr(cad, "DREAM_V41_EPS_SIGMA", 1.0e-5))
        self.eps_d = float(getattr(cad, "DREAM_V41_EPS_D", 1.0e-6))
        self.delta = float(getattr(cad, "DREAM_V41_DELTA", 1.0e-5))

        self.nu_c = float(
            getattr(cad, "DREAM_V41_NU_C", getattr(cad, "DREAM_TEXT_PRIOR_STRENGTH", 4.0))
        )
        self.nu_mu = float(
            getattr(cad, "DREAM_V41_NU_MU", getattr(cad, "DREAM_MEAN_PRIOR_STRENGTH", 4.0))
        )
        self.nu_sigma = float(
            getattr(cad, "DREAM_V41_NU_SIGMA", getattr(cad, "DREAM_COV_PRIOR_STRENGTH", 16.0))
        )

        self.alpha_t = float(getattr(cad, "DREAM_V41_ALPHA_T", 0.5))

        self.rho0 = float(getattr(cad, "DREAM_V41_RHO0", 0.1))
        self.rho_min = float(getattr(cad, "DREAM_V41_RHO_MIN", 0.1))
        self.gamma_c = float(getattr(cad, "DREAM_V41_GAMMA_C", 2.0))
        self.gamma_q = float(getattr(cad, "DREAM_V41_GAMMA_Q", 4.0))
        if not (self.gamma_q > self.gamma_c):
            raise ValueError(
                f"DREAM v4.1 requires DREAM_V41_GAMMA_Q > DREAM_V41_GAMMA_C, "
                f"got gamma_q={self.gamma_q}, gamma_c={self.gamma_c}"
            )

        self.top_m = int(getattr(cad, "DREAM_V41_TOP_M", -1))
        self.residual_clip = float(getattr(cad, "DREAM_V41_RESIDUAL_CLIP", 1.5))

        self.manual_lambda = float(
            getattr(cad, "DREAM_V41_LAMBDA", getattr(cad, "DREAM_LAMBDA", -1.0))
        )
        self.lambda_grid = self._parse_grid(
            getattr(cad, "DREAM_V41_LAMBDA_GRID", getattr(cad, "DREAM_LAMBDA_GRID", [0.0, 0.25, 0.5]))
        )
        self.density_on_train = bool(
            getattr(cad, "DREAM_V41_DENSITY_ON_TRAIN", getattr(cad, "DREAM_DENSITY_ON_TRAIN", False))
        )
        self.posterior_base_on_train = bool(getattr(cad, "DREAM_V41_POSTERIOR_BASE_ON_TRAIN", False))
        self.temperature_on_train = bool(getattr(cad, "DREAM_V41_TEMPERATURE_ON_TRAIN", False))
        self.temperature = float(getattr(cad, "DREAM_V41_TEMPERATURE", 1.0))

        self.gate_a = float(getattr(cad, "DREAM_V41_GATE_A", getattr(cad, "DREAM_GATE_A", 5.0)))
        self.gate_quantile = float(getattr(cad, "DREAM_V41_GATE_QUANTILE", 0.1))
        self.gate_min_support = int(getattr(cad, "DREAM_V41_GATE_MIN_SUPPORT", 20))

        # Main v4.1 path uses shared x_k(z)=U^T Log_{m_k}(z).
        # Set true only for A10 ablation: U_k=P_k U G_k^{-1/2}.
        self.use_per_class_basis = bool(getattr(cad, "DREAM_V41_USE_PER_CLASS_BASIS", False))

        self.debug = bool(getattr(cad, "DREAM_V41_DEBUG", getattr(cad, "DREAM_DEBUG", True)))

        self.register_buffer("dream_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dream_class_centers", torch.empty(c, d))
        self.register_buffer("dream_shared_basis", torch.empty(d, 0))
        self.register_buffer("dream_class_basis", torch.empty(c, d, 0))
        self.register_buffer("dream_mu", torch.empty(c, 0))
        self.register_buffer("dream_var", torch.empty(c, 0))
        self.register_buffer("dream_var0", torch.empty(0))

        self.register_buffer("dream_lambda0", torch.tensor(0.0))
        self.register_buffer("dream_s0", torch.tensor(1.0))
        self.register_buffer("dream_gate_q", torch.tensor(0.0))
        self.register_buffer("dream_gate_iqr", torch.tensor(1.0))
        self.register_buffer("dream_support_n", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _parse_grid(value) -> List[float]:
        if value is None:
            return [0.0, 0.25, 0.5]
        if isinstance(value, str):
            cleaned = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
            return [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        if isinstance(value, Iterable):
            return [float(x) for x in value]
        return [float(value)]

    def _effective_top_m(self) -> int:
        if self.top_m > 0:
            return max(1, min(int(self.top_m), self.num_classes))
        return self.num_classes if self.num_classes <= 50 else 50

    def bayes_base_logits_from_mc(self, logits_all: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Return BayesAdapter posterior predictive log-probabilities:
            log mean_s softmax(logits_s).

        During training, keep the original mean-logit path unless explicitly enabled,
        so BayesAdapter training/KL behavior remains unchanged.
        """
        if training and not self.posterior_base_on_train:
            return logits_all.mean(dim=0)
        if logits_all.ndim != 3:
            raise ValueError(
                f"bayes_base_logits_from_mc expects logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )
        x = logits_all.float()
        log_probs = x - torch.logsumexp(x, dim=-1, keepdim=True)
        out = torch.logsumexp(log_probs, dim=0) - math.log(float(logits_all.shape[0]))
        return out.to(dtype=logits_all.dtype)

    @torch.no_grad()
    def _base_log_probs_mc(self, features: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        if n_samples is None:
            n_samples = int(getattr(self.cfg.CLIP_ADAPTERS, "N_TEST_SAMPLES", 128))
        n_samples = max(1, int(n_samples))

        z = self._normalize(features.float())
        prototypes = self.sample_prototypes(n_samples=n_samples).detach().float().to(z.device)
        prototypes = self._normalize(prototypes)
        scale = self.logit_scale.exp().detach().float().to(z.device)
        logits_all = torch.einsum("bd,scd->sbc", z, prototypes) * scale
        return self.bayes_base_logits_from_mc(logits_all, training=False)

    def _log_map(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        dot = (z * m).sum(dim=-1, keepdim=True).clamp(
            -1.0 + self.eps_theta,
            1.0 - self.eps_theta,
        )
        theta = torch.acos(dot)
        sin_theta = torch.sqrt((1.0 - dot.pow(2)).clamp_min(1.0e-12))
        factor = theta / sin_theta.clamp_min(1.0e-6)
        return factor * (z - dot * m)

    def _build_centers(self, z: torch.Tensor, y: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        centers = []
        for k in range(self.num_classes):
            idx = (y == k).nonzero(as_tuple=False).flatten()
            visual_sum = z.index_select(0, idx).sum(dim=0) if int(idx.numel()) > 0 else torch.zeros_like(text[k])
            centers.append(self._normalize((self.nu_c * text[k] + visual_sum).unsqueeze(0)).squeeze(0))
        return torch.stack(centers, dim=0)

    def _build_shared_basis(self, z: torch.Tensor, y: torch.Tensor, text: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        m_y = centers.index_select(0, y)
        a_v = self._log_map(z, m_y).transpose(0, 1).contiguous()  # [D, N]

        t_bar = self._normalize(text.sum(dim=0, keepdim=True)).squeeze(0)
        a_t = (text - t_bar.unsqueeze(0)).transpose(0, 1).contiguous()  # [D, K]

        a_v = a_v / a_v.norm().clamp_min(1.0e-12)
        a_t = a_t / a_t.norm().clamp_min(1.0e-12)
        a = torch.cat([a_v, self.alpha_t * a_t], dim=1).float()

        d = int(a.shape[0])
        n_cols = int(a.shape[1])
        max_rank = max(1, min(int(self.dream_rank), n_cols, max(d - 1, 1)))

        try:
            u, s, _ = torch.linalg.svd(a, full_matrices=False)
            numerical_rank = int((s > 1.0e-8).sum().item())
            rank = max(1, min(max_rank, numerical_rank))
            basis = u[:, :rank].contiguous()
        except RuntimeError:
            q, _ = torch.linalg.qr(a, mode="reduced")
            basis = q[:, :max_rank].contiguous()

        return basis.contiguous()

    def _project_basis_to_class_tangent(self, basis: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
        out = []
        for k in range(self.num_classes):
            m = centers[k]
            pu = basis - m.unsqueeze(-1) * (m @ basis).unsqueeze(0)
            g = pu.transpose(0, 1) @ pu + self.delta * eye
            evals, evecs = torch.linalg.eigh(g.float())
            invsqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(self.delta))) @ evecs.transpose(0, 1)
            uk = pu.float() @ invsqrt
            out.append(uk.to(dtype=basis.dtype))
        return torch.stack(out, dim=0).contiguous()

    def _coords_shared_all(self, z: torch.Tensor, centers: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        dots = (z @ centers.t()).clamp(-1.0 + self.eps_theta, 1.0 - self.eps_theta)
        theta = torch.acos(dots)
        sin_theta = torch.sqrt((1.0 - dots.pow(2)).clamp_min(1.0e-12))
        factor = theta / sin_theta.clamp_min(1.0e-6)

        a = z @ basis          # [B, R]
        h = centers @ basis    # [C, R]
        return factor.unsqueeze(-1) * (a.unsqueeze(1) - dots.unsqueeze(-1) * h.unsqueeze(0))

    def _coords_shared_labels(self, z: torch.Tensor, y: torch.Tensor, centers: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        m = centers.index_select(0, y)
        dots = (z * m).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps_theta, 1.0 - self.eps_theta)
        theta = torch.acos(dots)
        sin_theta = torch.sqrt((1.0 - dots.pow(2)).clamp_min(1.0e-12))
        factor = theta / sin_theta.clamp_min(1.0e-6)

        a = z @ basis
        h = m @ basis
        return factor * (a - dots * h)

    def _coords_for_all_classes(self, z: torch.Tensor, class_start: int = 0, class_end: Optional[int] = None) -> torch.Tensor:
        if class_end is None:
            class_end = self.num_classes

        centers = self.dream_class_centers[class_start:class_end].to(device=z.device, dtype=z.dtype)

        if self.use_per_class_basis:
            bases = self.dream_class_basis[class_start:class_end].to(device=z.device, dtype=z.dtype)
            u = self._log_map(z[:, None, :], centers[None, :, :])
            return torch.einsum("bcd,cdr->bcr", u, bases)

        basis = self.dream_shared_basis.to(device=z.device, dtype=z.dtype)
        return self._coords_shared_all(z, centers, basis)

    def _coords_for_labels(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        centers: torch.Tensor,
        basis: torch.Tensor,
        class_basis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_per_class_basis and class_basis is not None:
            xs = []
            for i in range(int(z.shape[0])):
                k = int(y[i].item())
                u = self._log_map(z[i : i + 1], centers[k : k + 1])
                xs.append((u @ class_basis[k]).squeeze(0))
            return torch.stack(xs, dim=0) if xs else z.new_empty(0, class_basis.shape[-1])

        return self._coords_shared_labels(z, y, centers, basis)

    def _fit_shrinkage_gaussians(self, x_true: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, r = int(x_true.shape[0]), int(x_true.shape[-1])

        if n <= 1:
            global_var = x_true.new_ones(r) * self.eps_sigma
        else:
            global_var = x_true.var(dim=0, unbiased=False).clamp_min(0.0)

        class_means = x_true.new_zeros(self.num_classes, r)
        within_ss = x_true.new_zeros(r)

        for k in range(self.num_classes):
            idx = (y == k).nonzero(as_tuple=False).flatten()
            if int(idx.numel()) == 0:
                continue
            xk = x_true.index_select(0, idx)
            bar = xk.mean(dim=0)
            class_means[k] = bar
            within_ss = within_ss + (xk - bar.unsqueeze(0)).pow(2).sum(dim=0)

        within_var = within_ss / float(max(n - self.num_classes, 1))
        eps_vec = x_true.new_full((r,), self.eps_sigma)
        var0 = torch.maximum(torch.maximum(within_var, self.rho0 * global_var), eps_vec).clamp_min(self.eps_sigma)

        mu = x_true.new_zeros(self.num_classes, r)
        var = x_true.new_zeros(self.num_classes, r)
        lo = self.rho_min * var0
        hi = self.gamma_c * var0

        for k in range(self.num_classes):
            idx = (y == k).nonzero(as_tuple=False).flatten()
            n_k = int(idx.numel())
            if n_k == 0:
                mu[k] = 0.0
                var[k] = var0
                continue

            xk = x_true.index_select(0, idx)
            bar = xk.mean(dim=0)
            mu_k = (float(n_k) / (float(n_k) + self.nu_mu)) * bar
            ss = (xk - mu_k.unsqueeze(0)).pow(2).sum(dim=0)
            raw = (self.nu_sigma * var0 + ss) / (self.nu_sigma + float(n_k)) + self.eps_sigma

            mu[k] = mu_k
            var[k] = torch.maximum(torch.minimum(raw, hi), lo).clamp_min(self.eps_sigma)

        return mu, var, var0

    def _density_ratio_from_fitted(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.dream_fitted.item()) or self.dream_shared_basis.numel() == 0:
            return features.new_zeros(int(features.shape[0]), self.num_classes)

        z = self._normalize(features.float())
        var0 = self.dream_var0.to(device=z.device, dtype=z.dtype).clamp_min(self.eps_sigma)
        mu = self.dream_mu.to(device=z.device, dtype=z.dtype)
        var = self.dream_var.to(device=z.device, dtype=z.dtype).clamp_min(self.eps_sigma)
        r = int(var0.shape[0])

        out = z.new_empty(int(z.shape[0]), self.num_classes)
        bg_var = (self.gamma_q * var0).view(1, 1, r).clamp_min(self.eps_sigma)
        bg_logdet = torch.log(self.gamma_q * var0).sum()

        chunk = max(1, int(self.dream_chunk_classes))
        for start in range(0, self.num_classes, chunk):
            end = min(self.num_classes, start + chunk)
            x = self._coords_for_all_classes(z, start, end)
            mu_k = mu[start:end].unsqueeze(0)
            var_k = var[start:end].unsqueeze(0).clamp_min(self.eps_sigma)

            quad_p = ((x - mu_k).pow(2) / var_k).sum(dim=-1)
            logdet_p = torch.log(var[start:end].clamp_min(self.eps_sigma)).sum(dim=-1).unsqueeze(0)
            quad_q = (x.pow(2) / bg_var).sum(dim=-1)

            # Constants -r/2 log(2pi) cancel in log p_k - log q_k.
            d = -0.5 * quad_p - 0.5 * logdet_p + 0.5 * quad_q + 0.5 * bg_logdet
            out[:, start:end] = d

        return out

    def _raw_id_evidence(self, d: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(d, dim=-1) - math.log(float(self.num_classes))

    def _candidate_density_std(self, d: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        m = self._effective_top_m()
        idx = torch.topk(base_logits.float(), k=m, dim=-1).indices
        cand = d.gather(dim=1, index=idx)
        return cand.std(dim=-1, unbiased=False)

    def _bounded_topm_residual(self, d: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        m = self._effective_top_m()
        idx = torch.topk(base_logits.float(), k=m, dim=-1).indices
        cand = d.gather(dim=1, index=idx)

        mean = cand.mean(dim=-1, keepdim=True)
        std = cand.std(dim=-1, unbiased=False, keepdim=True)
        s0 = self.dream_s0.to(device=d.device, dtype=d.dtype).view(1, 1)
        std = torch.maximum(std, s0).clamp_min(self.eps_d)

        residual = ((cand - mean) / std).clamp(-float(self.residual_clip), float(self.residual_clip))
        delta = torch.zeros_like(d)
        delta.scatter_(dim=1, index=idx, src=residual)
        return delta

    def _select_lambda(self, labels: torch.Tensor, d: torch.Tensor, base_log_probs: torch.Tensor) -> float:
        if self.manual_lambda >= 0.0:
            return float(self.manual_lambda)

        grid = list(self.lambda_grid)
        if 0.0 not in grid:
            grid = [0.0] + grid

        delta = self._bounded_topm_residual(d, base_log_probs)
        best_lambda = 0.0
        best_nll = None

        for lam in grid:
            logits = base_log_probs + float(lam) * delta
            nll = F.cross_entropy(logits, labels, reduction="mean")
            value = float(nll.detach().cpu().item())
            if best_nll is None or value < best_nll:
                best_nll = value
                best_lambda = float(lam)

        return best_lambda

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        if not self.dream_enabled:
            return None

        device = self.base_text_features.device
        z_raw = features_train.detach().to(device=device, dtype=torch.float32)
        y = labels_train.detach().to(device=device, dtype=torch.long).flatten()
        valid = ((y >= 0) & (y < self.num_classes)).nonzero(as_tuple=False).flatten()
        z_raw = z_raw.index_select(0, valid)
        y = y.index_select(0, valid)

        if int(z_raw.shape[0]) == 0:
            self.dream_fitted.fill_(False)
            return None

        z = self._normalize(z_raw)
        text = self._normalize(self.base_text_features.detach().float().to(device))

        centers = self._build_centers(z, y, text)
        basis = self._build_shared_basis(z, y, text, centers)

        if self.use_per_class_basis:
            class_basis = self._project_basis_to_class_tangent(basis, centers)
        else:
            class_basis = torch.empty(
                self.num_classes,
                int(basis.shape[0]),
                int(basis.shape[1]),
                device=device,
                dtype=basis.dtype,
            )

        x_true = self._coords_for_labels(
            z=z,
            y=y,
            centers=centers,
            basis=basis,
            class_basis=class_basis if self.use_per_class_basis else None,
        )
        mu, var, var0 = self._fit_shrinkage_gaussians(x_true, y)

        self.dream_class_centers = centers.to(self.base_text_features.dtype)
        self.dream_shared_basis = basis.to(self.base_text_features.dtype)
        self.dream_class_basis = class_basis.to(self.base_text_features.dtype)
        self.dream_mu = mu.to(self.base_text_features.dtype)
        self.dream_var = var.to(self.base_text_features.dtype)
        self.dream_var0 = var0.to(self.base_text_features.dtype)
        self.dream_fitted.fill_(True)
        self.dream_support_n = torch.tensor(int(z_raw.shape[0]), device=device, dtype=torch.long)

        d_support = self._density_ratio_from_fitted(z_raw)
        base_support = self._base_log_probs_mc(z_raw)

        s0 = self._candidate_density_std(d_support, base_support).median().clamp_min(self.eps_d)
        self.dream_s0 = s0.to(device=device, dtype=self.base_text_features.dtype)

        lam = self._select_lambda(y, d_support, base_support)
        self.dream_lambda0 = torch.tensor(lam, device=device, dtype=self.base_text_features.dtype)

        r_support = self._raw_id_evidence(d_support)
        q = torch.quantile(r_support.float(), float(self.gate_quantile)).to(device)
        q25 = torch.quantile(r_support.float(), 0.25).to(device)
        q75 = torch.quantile(r_support.float(), 0.75).to(device)
        iqr = (q75 - q25).abs().clamp_min(self.eps_d)
        self.dream_gate_q = q.to(self.base_text_features.dtype)
        self.dream_gate_iqr = iqr.to(self.base_text_features.dtype)

        if self.debug:
            counts = torch.bincount(y, minlength=self.num_classes)
            nonempty = int((counts > 0).sum().item())
            print(
                "[DREAM-BayesAdapter v4.1] fitted: "
                f"support={int(z.shape[0])}, classes={nonempty}/{self.num_classes}, "
                f"rank={int(basis.shape[1])}, lambda0={float(self.dream_lambda0):.4g}, "
                f"topM={self._effective_top_m()}, s0={float(self.dream_s0):.4g}, "
                f"gate_q={float(self.dream_gate_q):.4g}, gate_iqr={float(self.dream_gate_iqr):.4g}, "
                f"gamma_q={self.gamma_q}, gamma_c={self.gamma_c}, "
                f"clip=[-{self.residual_clip},{self.residual_clip}], "
                f"use_per_class_basis={self.use_per_class_basis}"
            )
        return None

    def density_ratio(self, features: torch.Tensor) -> torch.Tensor:
        return self._density_ratio_from_fitted(features)

    def cache_logits(self, features: torch.Tensor, base_logits: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if not self.dream_enabled or not bool(self.dream_fitted.item()):
            return None
        if self.training and not self.density_on_train:
            return None

        lam = float(self.dream_lambda0.detach().cpu().item())
        if abs(lam) <= 0.0:
            return None

        if base_logits is None:
            base_logits = self._base_log_probs_mc(features)

        d = self.density_ratio(features)
        delta = self._bounded_topm_residual(d, base_logits.detach())
        return (lam * delta).to(dtype=features.dtype)

    def postprocess_logits(self, logits: torch.Tensor, features: torch.Tensor, training: bool = False) -> torch.Tensor:
        # v4.1 never applies a probability-space gate mixture.
        # Temperature is a global calibration variant only.
        if training and not self.temperature_on_train:
            return logits
        if self.temperature <= 0:
            raise ValueError(f"DREAM_V41_TEMPERATURE must be > 0, got {self.temperature}")
        if abs(self.temperature - 1.0) <= 1.0e-12:
            return logits
        return logits / float(self.temperature)

    @torch.no_grad()
    def dream_scores(self, features: torch.Tensor, logits: Optional[torch.Tensor] = None) -> dict:
        d = self.density_ratio(features)
        r = self._raw_id_evidence(d)

        support_n = int(self.dream_support_n.detach().cpu().item())
        if support_n < self.gate_min_support:
            gate = torch.ones_like(r)
        else:
            gate = torch.sigmoid(
                float(self.gate_a)
                * (r - self.dream_gate_q.to(device=r.device, dtype=r.dtype))
                / self.dream_gate_iqr.to(device=r.device, dtype=r.dtype).clamp_min(self.eps_d)
            )

        out = {
            "density_ratio": d,
            "id_evidence": r,
            "gate": gate,
            "ood_score": -r,
        }
        if logits is not None:
            probs = torch.softmax(logits.float(), dim=-1)
            max_prob = probs.max(dim=-1).values.to(gate.dtype)
            out["max_probability"] = max_prob
            out["selective_score"] = gate * max_prob
        return out
