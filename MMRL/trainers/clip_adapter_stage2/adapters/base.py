from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseAdapter(nn.Module):
    """Base adapter contract for clip_adapters stage-2 refactor.

    External behavior intentionally matches the original AdapterMethod as closely
    as possible so the trainer can keep using `self.model.adapter`.
    """

    initialization_name = "BASE"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.device = base_text_features.device
        self.logit_scale = clip_model.logit_scale
        self.initialization = cfg.TRAINER.ClipADAPTER.INIT
        self.apply_constraint = cfg.TRAINER.ClipADAPTER.CONSTRAINT
        self.modeltype = "MP"
        self.distance = "KL"
        self.register_buffer("base_text_features", base_text_features)
        self.alpha_constraint = torch.zeros((base_text_features.shape[0]), device=base_text_features.device)
        self.augmentations = True
        self.epochs_aumentation = 20
        self.kl_weight = 0.0
        self.anneal_start_epoch = 20
        self.total_epochs = 300
        self.anneal_rate = 1.0 / max(1, self.total_epochs - self.anneal_start_epoch)
        self.grid_search_param: Dict[str, list] = {}
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint)

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def get_constraint_reference(self) -> torch.Tensor:
        if hasattr(self, "prototypes"):
            return self.prototypes
        if hasattr(self, "variational_mu"):
            return self.variational_mu
        raise AttributeError("Adapter has neither `prototypes` nor `variational_mu`.")

    def zero_shot_constraint(self) -> torch.Tensor:
        reference = self.get_constraint_reference()
        if "l2" in self.apply_constraint:
            disimilitude = (reference - self.base_text_features.clone()).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            disimilitude = 1 - torch.nn.functional.cosine_similarity(reference, self.base_text_features.clone())
        else:
            raise NotImplementedError("Dissimilitude metric for constraint not implemented")
        return torch.mean(self.alpha_constraint * disimilitude)

    def init_lagrangian_multipliers(self, labels_ds: torch.Tensor, logits_ds: torch.Tensor):
        if "balanced" in self.apply_constraint:
            performance = torch.ones(logits_ds.shape[-1], dtype=torch.float32, device=logits_ds.device)
        else:
            with torch.no_grad():
                labels_one_hot = torch.nn.functional.one_hot(labels_ds)
                performance = (
                    torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32))
                    / labels_one_hot.sum(0)
                )
                if "corrected" in self.apply_constraint:
                    performance *= logits_ds.shape[-1] / torch.sum(performance).item()
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1], dtype=torch.float32, device=logits_ds.device) * torch.mean(performance).item()
        self.alpha_constraint = performance.to(self.base_text_features.device)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint)

    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
            y_inf = -1 / (2 * rho) * (lambd ** 2)
            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)
            sup = x >= 0
            return (
                torch.where(sup, y_sup, y_inf),
                torch.where(sup, grad_y_sup, grad_y_inf),
            )

        reference = self.get_constraint_reference()
        disimilitude = (reference - self.base_text_features.clone()).pow(2).sum(-1)
        _, phr_grad = phr(disimilitude, self.alpha_constraint, torch.clamp(self.penalty_parameter + 1e-8, min=1e-8))
        self.alpha_constraint = phr_grad.detach().clone()
        self.penalty_parameter = disimilitude.detach().clone()

    def extra_loss(self) -> Optional[torch.Tensor]:
        return None

    def reset_hparams(self, params: Dict) -> None:
        """Optional hook used by grid search."""
        return None
