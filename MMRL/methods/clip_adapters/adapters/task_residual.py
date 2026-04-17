import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter


class TaskResidualAdapter(BaseAdapter):
    initialization_name = "TR"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, alpha: float = 0.5):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using TaskRes-Adapter")
        self.alpha = alpha
        self.grid_search_param = {
            "lr": [1e-1, 1e-2, 1e-3],
            "alpha": list(np.arange(0.2, 1.2, 0.2)),
        }
        self.prototypes = nn.Parameter(torch.zeros_like(base_text_features.clone()))

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        return self.prototypes.unsqueeze(0).expand(n_samples, -1, -1)

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = params["alpha"]
