import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter


class ClipAdapterResidual(BaseAdapter):
    initialization_name = "ClipA"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, ratio: float = 0.2):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using CLIP-Adapter")
        self.grid_search_param = {
            "lr": [1e-1, 1e-2, 1e-3],
            "ratio": list(np.arange(0.2, 1, 0.2)),
        }
        self.ratio = ratio
        self.prototypes = nn.Parameter(base_text_features.clone())
        self.prototypes.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(base_text_features.shape[-1], base_text_features.shape[-1] // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(base_text_features.shape[-1] // 4, base_text_features.shape[-1], bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        return self.prototypes.unsqueeze(0).expand(n_samples, -1, -1)

    def reset_hparams(self, params):
        if "ratio" in params:
            self.ratio = params["ratio"]
