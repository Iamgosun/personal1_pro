import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter


class TaskResidualAdapter(BaseAdapter):
    initialization_name = "TR"
    adapter_kind = "prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, alpha: float = 0.5):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using TaskRes-Adapter")

        self.alpha = float(alpha)
        self.grid_search_param = {
            "lr": [1e-1, 1e-2, 1e-3],
            "alpha": list(np.arange(0.2, 1.2, 0.2)),
        }

        self.residual = nn.Parameter(torch.zeros_like(base_text_features))

    def get_prototypes(self) -> torch.Tensor:
        return self.base_text_features + self.alpha * self.residual

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])