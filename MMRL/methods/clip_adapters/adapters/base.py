from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAdapter(nn.Module):
    """
    Unified adapter contract for ClipAdapters.

    Adapter categories:
    - prototype            : deterministic class prototypes, shape [C, D]
    - stochastic_prototype : sampled class prototypes, shape [S, C, D]

    Optional extension points:
    - adapt_features(features) -> features'
    - cache_logits(features)   -> [B, C] or None
    - build_cache(features_train, labels_train)
    """

    initialization_name = "BASE"
    adapter_kind = "prototype"  # or "stochastic_prototype"
    uses_cache = False

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.device = base_text_features.device
        self.logit_scale = clip_model.logit_scale

        clip_cfg = getattr(cfg, "CLIP_ADAPTERS", None)
        legacy_cfg = getattr(getattr(cfg, "TRAINER", None), "ClipADAPTER", None)

        self.initialization = self._cfg_get("INIT", "ZS", clip_cfg, legacy_cfg)
        self.apply_constraint = self._cfg_get("CONSTRAINT", "none", clip_cfg, legacy_cfg)
        self.modeltype = self._cfg_get("TYPE", "MP", clip_cfg, legacy_cfg)

        self.distance = "KL"
        self.register_buffer("base_text_features", base_text_features.detach().clone())

        n_classes = int(base_text_features.shape[0])
        self.register_buffer(
            "alpha_constraint",
            torch.zeros(n_classes, dtype=base_text_features.dtype, device=base_text_features.device),
        )
        self.register_buffer(
            "penalty_parameter",
            torch.zeros(n_classes, dtype=base_text_features.dtype, device=base_text_features.device),
        )

        self.augmentations = True
        self.epochs_aumentation = 20
        self.kl_weight = 0.0
        self.anneal_start_epoch = 20
        self.total_epochs = 300
        self.anneal_rate = 1.0 / max(1, self.total_epochs - self.anneal_start_epoch)

        self.grid_search_param: Dict[str, list] = {}

    @staticmethod
    def _cfg_get(name: str, default, *cfgs):
        for cfg in cfgs:
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
        return default

    def forward(self, n_samples: int = 1):
        if self.adapter_kind == "stochastic_prototype":
            return self.sample_prototypes(n_samples)
        return self.get_prototypes()

    def get_prototypes(self) -> torch.Tensor:
        """
        Deterministic prototypes with shape [C, D].
        """
        raise NotImplementedError

    def sample_prototypes(self, n_samples: int = 1) -> torch.Tensor:
        """
        Stochastic prototypes with shape [S, C, D].
        Deterministic adapters fall back to repeated samples.
        """
        prototypes = self.get_prototypes()
        if prototypes.ndim != 2:
            raise ValueError(
                f"{self.__class__.__name__}.get_prototypes() must return [C, D], got {tuple(prototypes.shape)}"
            )
        return prototypes.unsqueeze(0).expand(n_samples, -1, -1)

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Optional feature adaptation before logits are computed.
        """
        return features

    def cache_logits(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Optional additive cache logits for Tip-Adapter style methods.
        """
        return None

    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        """
        Optional cache construction hook called after train features are extracted.
        """
        return None

    def get_constraint_reference(self) -> torch.Tensor:
        """
        Returns the deterministic reference prototypes used by zero-shot constraints.
        """
        if hasattr(self, "variational_mu"):
            return self.variational_mu

        prototypes = self.get_prototypes()
        if prototypes.ndim != 2:
            raise ValueError(
                f"Constraint reference must be [C, D], got {tuple(prototypes.shape)} from {self.__class__.__name__}"
            )
        return prototypes

    def zero_shot_constraint(self) -> torch.Tensor:
        reference = self.get_constraint_reference()

        if "l2" in self.apply_constraint:
            dissimilarity = (reference - self.base_text_features).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            dissimilarity = 1.0 - F.cosine_similarity(reference, self.base_text_features, dim=-1)
        else:
            raise NotImplementedError("Dissimilarity metric for constraint not implemented")

        return torch.mean(self.alpha_constraint * dissimilarity)

    def init_lagrangian_multipliers(self, labels_ds: torch.Tensor, logits_ds: torch.Tensor):
        n_classes = int(self.base_text_features.shape[0])

        if "balanced" in self.apply_constraint:
            performance = torch.ones(
                n_classes, dtype=torch.float32, device=logits_ds.device
            )
        else:
            with torch.no_grad():
                labels_one_hot = F.one_hot(labels_ds.to(torch.long), num_classes=n_classes).to(torch.float32)
                denom = labels_one_hot.sum(0).clamp_min(1.0)
                performance = torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot) / denom

                if "corrected" in self.apply_constraint:
                    performance *= n_classes / torch.sum(performance).clamp_min(1e-12).item()

                if "constant" in self.apply_constraint:
                    performance = torch.ones_like(performance) * torch.mean(performance).item()

        self.alpha_constraint = performance.to(self.base_text_features.device)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint)

    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1.0 / (2.0 * rho) * (x ** 2 - lambd ** 2)
            y_inf = -1.0 / (2.0 * rho) * (lambd ** 2)
            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)
            sup = x >= 0
            return (
                torch.where(sup, y_sup, y_inf),
                torch.where(sup, grad_y_sup, grad_y_inf),
            )

        reference = self.get_constraint_reference()
        dissimilarity = (reference - self.base_text_features).pow(2).sum(-1)

        _, phr_grad = phr(
            dissimilarity,
            self.alpha_constraint,
            torch.clamp(self.penalty_parameter + 1e-8, min=1e-8),
        )
        self.alpha_constraint = phr_grad.detach().clone()
        self.penalty_parameter = dissimilarity.detach().clone()

    def extra_loss(self) -> Optional[torch.Tensor]:
        return None

    def reset_hparams(self, params: Dict) -> None:
        return None