from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .bayes_adapter import BayesAdapter


class DreamBayesAdapter(BayesAdapter):
    """
    DREAM-BayesAdapter v3, implemented in the existing ClipAdapters adapter tree.

    Strict v3 semantics:
      - BayesAdapter is the classifier-parameter uncertainty base.
      - DREAM adds only a bounded, adaptive density-ratio log-prob residual.
      - ID evidence R(z) and gate g(z) are returned for selective/OOD use only.
      - No probability-space mixture g*p + (1-g)/K is applied.

    Required model.py integration for exact fallback:
      In the stochastic BayesAdapter branch, call bayes_base_logits_from_mc(logits_all)
      when present. That makes the base logits equal to
          log E_W[softmax(logits(W))]
      at eval time, rather than E_W[logits(W)].
    """

    # Keep this for the existing BayesAdapter KL/loss checks.
    initialization_name = "BAYES_ADAPTER"
    dream_initialization_name = "DREAM_BAYES_ADAPTER"
    needs_support_features = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)

        cad = cfg.CLIP_ADAPTERS
        c, d = base_text_features.shape

        self.dream_enabled = bool(getattr(cad, "DREAM_V3_ENABLED", getattr(cad, "DREAM_ENABLED", True)))
        self.dream_rank = int(getattr(cad, "DREAM_V3_RANK", getattr(cad, "DREAM_RANK", 32)))
        self.dream_chunk_classes = int(getattr(cad, "DREAM_V3_CHUNK_CLASSES", getattr(cad, "DREAM_CHUNK_CLASSES", 64)))

        self.eps_theta = float(getattr(cad, "DREAM_V3_EPS_THETA", 1.0e-6))
        self.eps_p = float(getattr(cad, "DREAM_V3_EPS_P", 1.0e-12))
        self.eps_sigma = float(getattr(cad, "DREAM_V3_EPS_SIGMA", 1.0e-5))
        self.eps_d = float(getattr(cad, "DREAM_V3_EPS_D", 1.0e-6))
        self.eps_r = float(getattr(cad, "DREAM_V3_EPS_R", 1.0e-6))
        self.delta = float(getattr(cad, "DREAM_V3_DELTA", 1.0e-5))

        self.nu_c = float(getattr(cad, "DREAM_V3_NU_C", getattr(cad, "DREAM_TEXT_PRIOR_STRENGTH", 4.0)))
        self.nu_mu = float(getattr(cad, "DREAM_V3_NU_MU", getattr(cad, "DREAM_MEAN_PRIOR_STRENGTH", 4.0)))
        self.nu_sigma = float(getattr(cad, "DREAM_V3_NU_SIGMA", getattr(cad, "DREAM_COV_PRIOR_STRENGTH", 16.0)))

        self.alpha_v = float(getattr(cad, "DREAM_V3_ALPHA_V", 1.0))
        self.alpha_t = float(getattr(cad, "DREAM_V3_ALPHA_T", 0.5))

        self.rho0 = float(getattr(cad, "DREAM_V3_RHO0", 0.05))
        self.rho_min = float(getattr(cad, "DREAM_V3_RHO_MIN", 0.05))
        self.gamma_c = float(getattr(cad, "DREAM_V3_GAMMA_C", 2.0))
        self.gamma_q = float(getattr(cad, "DREAM_V3_GAMMA_Q", 4.0))
        if not (self.gamma_q > self.gamma_c):
            raise ValueError(
                f"DREAM v3 requires DREAM_V3_GAMMA_Q > DREAM_V3_GAMMA_C, "
                f"got gamma_q={self.gamma_q}, gamma_c={self.gamma_c}"
            )

        self.c_minus = float(getattr(cad, "DREAM_V3_C_MINUS", 1.5))
        self.c_plus = float(getattr(cad, "DREAM_V3_C_PLUS", 2.0))
        self.temperature = float(getattr(cad, "DREAM_V3_TEMPERATURE", 1.0))

        self.manual_lambda = float(getattr(cad, "DREAM_V3_LAMBDA", getattr(cad, "DREAM_LAMBDA", -1.0)))
        self.lambda_beta = float(getattr(cad, "DREAM_V3_LAMBDA_BETA", getattr(cad, "DREAM_LAMBDA_BETA", 0.01)))
        self.lambda_margin = float(getattr(cad, "DREAM_V3_LAMBDA_MARGIN", 0.0))
        self.density_on_train = bool(getattr(cad, "DREAM_V3_DENSITY_ON_TRAIN", getattr(cad, "DREAM_DENSITY_ON_TRAIN", False)))
        self.posterior_base_on_train = bool(getattr(cad, "DREAM_V3_POSTERIOR_BASE_ON_TRAIN", False))
        self.temperature_on_train = bool(getattr(cad, "DREAM_V3_TEMPERATURE_ON_TRAIN", False))

        self.gate_a = float(getattr(cad, "DREAM_V3_GATE_A", getattr(cad, "DREAM_GATE_A", 5.0)))
        self.max_all_pairs = int(getattr(cad, "DREAM_V3_MAX_ALL_PAIRS", 262144))
        self.debug = bool(getattr(cad, "DREAM_V3_DEBUG", getattr(cad, "DREAM_DEBUG", True)))

        self.register_buffer("dream_fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dream_class_centers", torch.empty(c, d))
        self.register_buffer("dream_shared_basis", torch.empty(d, 0))
        self.register_buffer("dream_class_basis", torch.empty(c, d, 0))
        self.register_buffer("dream_mu", torch.empty(c, 0))
        self.register_buffer("dream_var", torch.empty(c, 0))
        self.register_buffer("dream_var0", torch.empty(0))

        self.register_buffer("dream_lambda0", torch.tensor(0.0))
        self.register_buffer("dream_q_alpha", torch.tensor(0.0))
        self.register_buffer("dream_q_median", torch.tensor(1.0))
        self.register_buffer("dream_s_r", torch.tensor(1.0))
        self.register_buffer("dream_s_min", torch.tensor(1.0))

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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

    def bayes_base_logits_from_mc(self, logits_all: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Return BayesAdapter posterior predictive log-probabilities at eval time:
            log mean_s softmax(logits_s).

        This is the required base b_k(z) for strict DREAM v3 fallback.
        During training the default keeps the original mean-logit path unchanged.
        """
        if training and not self.posterior_base_on_train:
            return logits_all.mean(dim=0)
        probs = torch.softmax(logits_all.float(), dim=-1).mean(dim=0).clamp_min(self.eps_p)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(self.eps_p)
        return torch.log(probs).to(dtype=logits_all.dtype)

    @torch.no_grad()
    def _base_log_probs_mc(self, features: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        if n_samples is None:
            n_samples = int(getattr(self.cfg.CLIP_ADAPTERS, "N_TEST_SAMPLES", 10))
        n_samples = max(1, int(n_samples))

        z = self._normalize(features.float())
        prototypes = self.sample_prototypes(n_samples=n_samples).detach().float().to(z.device)
        prototypes = self._normalize(prototypes)
        scale = self.logit_scale.exp().detach().float().to(z.device)
        logits_all = torch.einsum("bd,scd->sbc", z, prototypes) * scale
        return self.bayes_base_logits_from_mc(logits_all, training=False)

    def _log_map(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        dot = (z * m).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps_theta, 1.0 - self.eps_theta)
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
        a = torch.cat([self.alpha_v * a_v, self.alpha_t * a_t], dim=1).float()

        max_rank = max(1, min(int(self.dream_rank), int(a.shape[0]), int(a.shape[1])))
        try:
            u, s, _ = torch.linalg.svd(a, full_matrices=False)
            rank = int((s > 1.0e-8).sum().item())
            rank = max(1, min(max_rank, rank))
            basis = u[:, :rank].contiguous()
        except RuntimeError:
            q, _ = torch.linalg.qr(a, mode="reduced")
            basis = q[:, :max_rank].contiguous()
        return self._normalize(basis.transpose(0, 1)).transpose(0, 1).contiguous()

    def _project_basis_to_class_tangent(self, basis: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
        out = []
        for k in range(self.num_classes):
            m = centers[k]
            pu = basis - m.unsqueeze(-1) * (m @ basis).unsqueeze(0)
            g = basis.transpose(0, 1) @ pu + self.delta * eye
            evals, evecs = torch.linalg.eigh(g.float())
            invsqrt = evecs @ torch.diag(torch.rsqrt(evals.clamp_min(self.delta))) @ evecs.transpose(0, 1)
            uk = pu.float() @ invsqrt
            out.append(uk.to(dtype=basis.dtype))
        return torch.stack(out, dim=0).contiguous()

    def _coords_for_all_classes(self, z: torch.Tensor, class_start: int = 0, class_end: Optional[int] = None) -> torch.Tensor:
        if class_end is None:
            class_end = self.num_classes
        centers = self.dream_class_centers[class_start:class_end].to(device=z.device, dtype=z.dtype)
        bases = self.dream_class_basis[class_start:class_end].to(device=z.device, dtype=z.dtype)
        u = self._log_map(z[:, None, :], centers[None, :, :])
        return torch.einsum("bcd,cdr->bcr", u, bases)

    def _coords_for_labels(self, z: torch.Tensor, y: torch.Tensor, centers: torch.Tensor, class_basis: torch.Tensor) -> torch.Tensor:
        xs = []
        for i in range(int(z.shape[0])):
            k = int(y[i].item())
            u = self._log_map(z[i : i + 1], centers[k : k + 1])
            xs.append((u @ class_basis[k]).squeeze(0))
        return torch.stack(xs, dim=0) if xs else z.new_empty(0, class_basis.shape[-1])

    def _robust_var(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[0]) <= 1:
            return x.new_ones(x.shape[-1]) * self.eps_sigma
        med = torch.median(x, dim=0).values
        mad = torch.median((x - med.unsqueeze(0)).abs(), dim=0).values
        var = (1.4826 * mad).pow(2)
        fallback = x.var(dim=0, unbiased=False)
        return torch.where(var > self.eps_sigma, var, fallback).clamp_min(self.eps_sigma)

    def _estimate_all_class_var(self, z: torch.Tensor) -> torch.Tensor:
        n, c = int(z.shape[0]), self.num_classes
        total_pairs = n * c
        values = []

        if total_pairs <= self.max_all_pairs:
            for start in range(0, c, max(1, self.dream_chunk_classes)):
                end = min(c, start + max(1, self.dream_chunk_classes))
                values.append(self._coords_for_all_classes(z, start, end).reshape(-1, self.dream_class_basis.shape[-1]))
        else:
            # Deterministic support-only approximation for very large ImageNet-style K*N.
            # The exact all-class MAD can be memory-prohibitive; this preserves the intended
            # background floor without changing inference formulas.
            g = torch.Generator(device=z.device)
            g.manual_seed(0)
            m = min(self.max_all_pairs, total_pairs)
            flat = torch.randperm(total_pairs, generator=g, device=z.device)[:m]
            sample_i = torch.div(flat, c, rounding_mode="floor")
            sample_k = flat % c
            z_s = z.index_select(0, sample_i)
            centers = self.dream_class_centers.to(device=z.device, dtype=z.dtype).index_select(0, sample_k)
            bases = self.dream_class_basis.to(device=z.device, dtype=z.dtype).index_select(0, sample_k)
            u = self._log_map(z_s, centers)
            x = torch.einsum("bd,bdr->br", u, bases)
            values.append(x)

        return self._robust_var(torch.cat(values, dim=0))

    def _fit_gaussians(self, x_true: torch.Tensor, y: torch.Tensor, var0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = int(x_true.shape[-1])
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
        return mu, var

    def _density_ratio_from_fitted(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.dream_fitted.item()) or self.dream_class_basis.numel() == 0:
            return features.new_zeros(int(features.shape[0]), self.num_classes)

        z = self._normalize(features.float())
        var0 = self.dream_var0.to(device=z.device, dtype=z.dtype).clamp_min(self.eps_sigma)
        mu = self.dream_mu.to(device=z.device, dtype=z.dtype)
        var = self.dream_var.to(device=z.device, dtype=z.dtype).clamp_min(self.eps_sigma)
        r = int(var0.shape[0])
        out = z.new_empty(int(z.shape[0]), self.num_classes)
        bg_var = (self.gamma_q * var0).view(1, 1, r)
        bg_logdet = torch.log(self.gamma_q * var0).sum()

        chunk = max(1, int(self.dream_chunk_classes))
        for start in range(0, self.num_classes, chunk):
            end = min(self.num_classes, start + chunk)
            x = self._coords_for_all_classes(z, start, end)
            mu_k = mu[start:end].unsqueeze(0)
            var_k = var[start:end].unsqueeze(0)
            quad_p = ((x - mu_k).pow(2) / var_k).sum(dim=-1)
            logdet_p = torch.log(var[start:end]).sum(dim=-1).unsqueeze(0)
            quad_q = (x.pow(2) / bg_var).sum(dim=-1)
            d = -0.5 * quad_p - 0.5 * logdet_p + 0.5 * quad_q + 0.5 * bg_logdet
            out[:, start:end] = d
        return out

    def _raw_id_evidence(self, d: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(d - math.log(float(self.num_classes)), dim=-1)

    def _standardize_density(self, d: torch.Tensor) -> torch.Tensor:
        mean = d.mean(dim=-1, keepdim=True)
        std = d.std(dim=-1, unbiased=False, keepdim=True)
        floor = self.dream_s_min.to(device=d.device, dtype=d.dtype).clamp_min(self.eps_d)
        std = torch.maximum(std, floor.view(1, 1)).clamp_min(self.eps_d)
        return ((d - mean) / std).clamp(-self.c_minus, self.c_plus)

    def _lambda_z(self, r_score: torch.Tensor, lambda0: Optional[float] = None) -> torch.Tensor:
        if lambda0 is None:
            lambda0 = float(self.dream_lambda0.detach().cpu().item())
        q_a = self.dream_q_alpha.to(device=r_score.device, dtype=r_score.dtype)
        q_m = self.dream_q_median.to(device=r_score.device, dtype=r_score.dtype)
        return float(lambda0) * ((r_score - q_a) / (q_m - q_a + self.eps_r)).clamp(0.0, 1.0)

    def _select_lambda(self, features: torch.Tensor, labels: torch.Tensor, d: torch.Tensor) -> float:
        if self.manual_lambda >= 0.0:
            return float(self.manual_lambda)

        grid = self._parse_grid(getattr(self.cfg.CLIP_ADAPTERS, "DREAM_V3_LAMBDA_GRID", getattr(self.cfg.CLIP_ADAPTERS, "DREAM_LAMBDA_GRID", None)))
        if 0.0 not in grid:
            grid = [0.0] + grid

        base_log_probs = self._base_log_probs_mc(features)
        r_score = self._raw_id_evidence(d)
        d_tilde = self._standardize_density(d)

        results = []
        for lam in grid:
            lam_z = self._lambda_z(r_score, float(lam)).unsqueeze(-1)
            logits = base_log_probs + lam_z * d_tilde
            nll = F.cross_entropy(logits, labels, reduction="mean")
            obj = nll + self.lambda_beta * float(lam) * float(lam)
            results.append((float(lam), float(obj.detach().cpu().item()), float(nll.detach().cpu().item())))

        baseline = next((x for x in results if abs(x[0]) <= 1.0e-12), results[0])
        best = min(results, key=lambda x: x[1])
        if self.lambda_margin > 0.0 and baseline[1] - best[1] <= self.lambda_margin:
            return 0.0
        return float(best[0])

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
        class_basis = self._project_basis_to_class_tangent(basis, centers)

        self.dream_class_centers = centers.to(self.base_text_features.dtype)
        self.dream_shared_basis = basis.to(self.base_text_features.dtype)
        self.dream_class_basis = class_basis.to(self.base_text_features.dtype)
        self.dream_fitted.fill_(True)

        x_true = self._coords_for_labels(z, y, centers, class_basis)
        s_true = self._robust_var(x_true)
        s_all = self._estimate_all_class_var(z)
        var0 = torch.maximum(s_true, self.rho0 * s_all).clamp_min(self.eps_sigma)
        mu, var = self._fit_gaussians(x_true, y, var0)

        self.dream_mu = mu.to(self.base_text_features.dtype)
        self.dream_var = var.to(self.base_text_features.dtype)
        self.dream_var0 = var0.to(self.base_text_features.dtype)

        d_support = self._density_ratio_from_fitted(z_raw)
        r_support = self._raw_id_evidence(d_support)
        n = int(z_raw.shape[0])
        alpha = min(0.2, max(0.05, 3.0 / float(n + 1)))
        q_alpha = torch.quantile(r_support.float(), alpha).to(device)
        q_median = torch.quantile(r_support.float(), 0.5).to(device)
        if float((q_median - q_alpha).detach().cpu().item()) <= self.eps_r:
            q_median = q_alpha + self.eps_r
        q25 = torch.quantile(r_support.float(), 0.25).to(device)
        q75 = torch.quantile(r_support.float(), 0.75).to(device)
        s_r = (q75 - q25).abs().clamp_min(self.eps_r)
        s_min = torch.median(d_support.std(dim=-1, unbiased=False).float()).to(device).clamp_min(self.eps_d)

        self.dream_q_alpha = q_alpha.to(self.base_text_features.dtype)
        self.dream_q_median = q_median.to(self.base_text_features.dtype)
        self.dream_s_r = s_r.to(self.base_text_features.dtype)
        self.dream_s_min = s_min.to(self.base_text_features.dtype)

        lam = self._select_lambda(z_raw, y, d_support)
        self.dream_lambda0 = torch.tensor(lam, device=device, dtype=self.base_text_features.dtype)

        if self.debug:
            counts = torch.bincount(y, minlength=self.num_classes)
            nonempty = int((counts > 0).sum().item())
            print(
                "[DREAM-BayesAdapter v3] fitted: "
                f"support={n}, classes={nonempty}/{self.num_classes}, rank={int(basis.shape[1])}, "
                f"lambda0={float(self.dream_lambda0):.4g}, q_alpha={float(self.dream_q_alpha):.4g}, "
                f"q_median={float(self.dream_q_median):.4g}, s_min={float(self.dream_s_min):.4g}, "
                f"gamma_q={self.gamma_q}, gamma_c={self.gamma_c}, clip=[-{self.c_minus},{self.c_plus}]"
            )
        return None

    def density_ratio(self, features: torch.Tensor) -> torch.Tensor:
        return self._density_ratio_from_fitted(features)

    def cache_logits(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.dream_enabled or not bool(self.dream_fitted.item()):
            return None
        if self.training and not self.density_on_train:
            return None
        if abs(float(self.dream_lambda0.detach().cpu().item())) <= 0.0:
            return None

        d = self.density_ratio(features)
        r = self._raw_id_evidence(d)
        lam_z = self._lambda_z(r).unsqueeze(-1)
        return (lam_z * self._standardize_density(d)).to(dtype=features.dtype)

    def postprocess_logits(self, logits: torch.Tensor, features: torch.Tensor, training: bool = False) -> torch.Tensor:
        # v3 does NOT apply evidence mixture to probabilities.
        # Optional temperature is a calibration variant only.
        if training and not self.temperature_on_train:
            return logits
        if self.temperature <= 0:
            raise ValueError(f"DREAM_V3_TEMPERATURE must be > 0, got {self.temperature}")
        if abs(self.temperature - 1.0) <= 1.0e-12:
            return logits
        return logits / float(self.temperature)

    @torch.no_grad()
    def dream_scores(self, features: torch.Tensor, logits: Optional[torch.Tensor] = None) -> dict:
        d = self.density_ratio(features)
        d_tilde = self._standardize_density(d)
        r = self._raw_id_evidence(d)
        lam_z = self._lambda_z(r)
        gate = torch.sigmoid(
            self.gate_a
            * (r - self.dream_q_alpha.to(device=r.device, dtype=r.dtype))
            / self.dream_s_r.to(device=r.device, dtype=r.dtype).clamp_min(self.eps_r)
        )
        out = {
            "density_ratio": d,
            "standardized_density_ratio": d_tilde,
            "id_evidence": r,
            "lambda_z": lam_z,
            "gate": gate,
            "ood_score": -r,
        }
        if logits is not None:
            probs = torch.softmax(logits.float(), dim=-1)
            out["max_probability"] = probs.max(dim=-1).values
            out["selective_score"] = gate * out["max_probability"].to(gate.dtype)
        return out
