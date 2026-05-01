from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .bayes_adapter import BayesAdapter


def _cad_get(cfg, name: str, default):
    return getattr(cfg.CLIP_ADAPTERS, name, default)


def _normalize_deba_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "none": "off",
        "false": "off",
        "disabled": "off",
        "disable": "off",
        "0": "off",
        "posthoc": "p",
        "post-hoc": "p",
        "deba-p": "p",
        "joint": "j",
        "deba-j": "j",
        "mix": "interp",
        "mixed": "interp",
        "deba-mix": "interp",
        "deba-interp": "interp",
    }
    return aliases.get(mode, mode)


class DEBAAdapter(BayesAdapter):
    """
    Dirichlet Evidence Bayesian Adapter.

    DEBA is a separate adapter method. It reuses BayesAdapter's stochastic
    prototype posterior parameterization, but uses a separate loss branch
    selected by deba_initialization_name == "DEBA".

    Important implementation note:
        initialization_name intentionally remains "BAYES_ADAPTER" so the
        existing ClipAdaptersModel MC path still produces bayes_logits_all.
        The old BayesAdapter loss is not used because loss.py checks
        deba_initialization_name before the generic BayesAdapter branch.

    Modes:
        DEBA_MODE=p:
            DEBA-P. Training data term stays MC supervised CE.
            Eval can return Dirichlet projected predictive mean.

        DEBA_MODE=j:
            DEBA-J. Training data term is Dirichlet expected NLL:
                E_{pi ~ Dir(alpha)}[-log pi_y]
              = digamma(sum alpha) - digamma(alpha_y)

        DEBA_MODE=interp:
            Stable DEBA-Interp:
                (1 - lambda) * NLL(mean MC probabilities)
              + lambda * Dirichlet expected NLL

    This is a projection/approximation of the stochastic posterior predictive
    distribution, not an exact closed-form posterior.
    """

    initialization_name = "BAYES_ADAPTER"
    deba_initialization_name = "DEBA"
    adapter_kind = "stochastic_prototype"

    @staticmethod
    def _mc_log_probs(logits_all: torch.Tensor) -> torch.Tensor:
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )
        x = logits_all.float()
        return x - torch.logsumexp(x, dim=-1, keepdim=True)

    @staticmethod
    def _mc_predictive_log_probs(logits_all: torch.Tensor) -> torch.Tensor:
        log_probs = DEBAAdapter._mc_log_probs(logits_all)
        n_samples = torch.tensor(
            float(logits_all.shape[0]),
            device=logits_all.device,
            dtype=log_probs.dtype,
        )
        return torch.logsumexp(log_probs, dim=0) - torch.log(n_samples)

    def _mc_supervised_ce(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        n_samples = int(logits_all.shape[0])
        flat_logits = logits_all.reshape(
            n_samples * logits_all.shape[1],
            logits_all.shape[2],
        )
        flat_labels = labels.unsqueeze(0).expand(n_samples, -1).reshape(-1)
        return F.cross_entropy(flat_logits, flat_labels, reduction="none").mean()

    def _mc_predictive_nll(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        log_mu = self._mc_predictive_log_probs(logits_all)
        return F.nll_loss(log_mu, labels, reduction="mean")

    def deba_mode(self) -> str:
        return _normalize_deba_mode(_cad_get(self.cfg, "DEBA_MODE", "interp"))



    def deba_projection(self, logits_all: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Project MC categorical probabilities onto a Dirichlet family.

        MC:
            p_s = softmax(logits_s)
            mu = mean_s p_s
            v_c = var_s p_{s,c}

        Dirichlet with mean mu and scalar concentration S:
            Var(pi_c) = mu_c * (1 - mu_c) / (S + 1)

        Important:
            DEBA_ALPHA0 is interpreted as total symmetric prior mass, not
            per-class prior evidence.

            alpha_c = alpha0 / C + S(x) * mu_c
            sum_c alpha_c = alpha0 + S(x)

        This avoids the old behavior:
            alpha_c = alpha0 + S(x) * mu_c
            sum_c alpha_c = C * alpha0 + S(x)

        The old behavior becomes severely under-confident when C is large.
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        eps = float(_cad_get(self.cfg, "DEBA_EPS", 1.0e-6))
        alpha0 = max(0.0, float(_cad_get(self.cfg, "DEBA_ALPHA0", 1.0)))
        s_min = float(_cad_get(self.cfg, "DEBA_S_MIN", 1.0))
        s_max = float(_cad_get(self.cfg, "DEBA_S_MAX", 1000.0))
        s_agg = str(_cad_get(self.cfg, "DEBA_S_AGG", "weighted_mean")).lower()
        grad_scale = float(_cad_get(self.cfg, "DEBA_S_GRAD_SCALE", 0.0))

        probs = torch.softmax(logits_all.float(), dim=-1)

        mu = probs.mean(dim=0).clamp_min(eps)
        mu = mu / mu.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Biased variance is intentional here: this is a deterministic MC projection
        # statistic, not an unbiased population estimator.
        var = probs.var(dim=0, unbiased=False).clamp_min(eps)

        per_class_s = mu * (1.0 - mu) / var - 1.0
        per_class_s = torch.nan_to_num(
            per_class_s,
            nan=s_max,
            posinf=s_max,
            neginf=s_min,
        ).clamp(min=s_min, max=s_max)

        if s_agg == "median":
            evidence = per_class_s.median(dim=-1).values
        elif s_agg == "mean":
            evidence = per_class_s.mean(dim=-1)
        else:
            weights = (mu * (1.0 - mu)).clamp_min(eps)
            evidence = (
                weights * per_class_s
            ).sum(dim=-1) / weights.sum(dim=-1).clamp_min(eps)

        evidence = evidence.clamp(min=s_min, max=s_max)

        # Default stop-gradient through S(x). This keeps gradients through mu while
        # reducing the incentive to collapse MC variance just to inflate evidence.
        if grad_scale <= 0.0:
            evidence_for_alpha = evidence.detach()
        elif grad_scale >= 1.0:
            evidence_for_alpha = evidence
        else:
            evidence_for_alpha = evidence.detach() + grad_scale * (
                evidence - evidence.detach()
            )

        num_classes = torch.tensor(
            float(logits_all.shape[-1]),
            device=logits_all.device,
            dtype=logits_all.float().dtype,
        )

        # Correct interpretation:
        # alpha0 is total symmetric Dirichlet prior mass.
        alpha_prior = alpha0 / num_classes

        alpha = (
            alpha_prior + evidence_for_alpha.unsqueeze(-1) * mu
        ).clamp_min(eps)

        alpha_sum = alpha.sum(dim=-1).clamp_min(eps)
        dirichlet_mean = alpha / alpha_sum.unsqueeze(-1)

        projected_var = mu * (1.0 - mu) / (
            evidence.unsqueeze(-1) + 1.0
        ).clamp_min(eps)

        projection_error = torch.sqrt(
            ((var - projected_var) ** 2).mean(dim=-1).clamp_min(0.0)
        )

        vacuity = num_classes / alpha_sum

        return {
            "alpha": alpha,
            "alpha_sum": alpha_sum,
            "mu": mu,
            "dirichlet_mean": dirichlet_mean,
            "evidence": evidence,
            "vacuity": vacuity,
            "mc_variance_mean": var.mean(dim=-1),
            "dirichlet_projection_error": projection_error,
        }



    def deba_expected_nll(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = self.deba_projection(logits_all)
        alpha = state["alpha"]
        alpha_sum = state["alpha_sum"]

        eps = float(_cad_get(self.cfg, "DEBA_EPS", 1.0e-6))
        y_alpha = alpha.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(eps)
        loss = (torch.digamma(alpha_sum) - torch.digamma(y_alpha)).mean()
        return loss, state

    def deba_data_term(
        self,
        logits_all: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mode = self.deba_mode()

        if mode in {"off", "p"}:
            data_term = self._mc_supervised_ce(logits_all, labels)
            return data_term, {
                "loss_ce": data_term.detach(),
                "data_term": data_term.detach(),
            }

        deba_nll, state = self.deba_expected_nll(logits_all, labels)
        mean_nll = self._mc_predictive_nll(logits_all, labels)

        if mode == "j":
            data_term = deba_nll
        elif mode == "interp":
            lam = float(_cad_get(self.cfg, "DEBA_LAMBDA", 0.5))
            lam = max(0.0, min(1.0, lam))
            data_term = (1.0 - lam) * mean_nll + lam * deba_nll
        else:
            raise ValueError(
                f"Unsupported CLIP_ADAPTERS.DEBA_MODE={mode!r}. "
                "Use p, j, or interp."
            )

        extras = {
            "loss_ce": data_term.detach(),
            "data_term": data_term.detach(),
            "loss_deba_dirichlet_nll": deba_nll.detach(),
            "loss_deba_mean_nll": mean_nll.detach(),
            "deba_evidence_mean": state["evidence"].mean().detach(),
            "deba_evidence_min": state["evidence"].min().detach(),
            "deba_evidence_max": state["evidence"].max().detach(),
            "deba_vacuity_mean": state["vacuity"].mean().detach(),
            "deba_mc_variance_mean": state["mc_variance_mean"].mean().detach(),
            "deba_projection_error": state[
                "dirichlet_projection_error"
            ].mean().detach(),
        }
        return data_term, extras

    def bayes_base_logits_from_mc(
        self,
        logits_all: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Return predictive logits used by training forward/evaluation.

        Default DEBA eval output is log Dirichlet predictive mean. Since the
        evaluator later applies softmax/log_softmax, log probabilities are a
        valid calibrated-logit representation.
        """
        if logits_all.ndim != 3:
            raise ValueError(
                f"Expected logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        if training and not bool(
            _cad_get(self.cfg, "DEBA_DIRICHLET_ON_TRAIN_LOGITS", False)
        ):
            return logits_all.mean(dim=0)

        eval_return = str(
            _cad_get(self.cfg, "DEBA_EVAL_RETURN", "dirichlet_mean")
        ).lower()

        if eval_return in {"mc", "mc_predictive", "posterior_predictive"}:
            return self._mc_predictive_log_probs(logits_all).to(dtype=logits_all.dtype)

        if eval_return in {"mean_logits", "logit_mean"}:
            return logits_all.mean(dim=0)

        state = self.deba_projection(logits_all)
        eps = float(_cad_get(self.cfg, "DEBA_EPS", 1.0e-6))
        return state["dirichlet_mean"].clamp_min(eps).log().to(dtype=logits_all.dtype)