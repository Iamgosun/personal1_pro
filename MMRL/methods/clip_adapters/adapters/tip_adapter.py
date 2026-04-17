import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter


class TipAdapter(BaseAdapter):
    initialization_name = "TipA"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, beta: float = 1.0, alpha: float = 1.0):
        super().__init__(cfg, clip_model, base_text_features)
        if "-f-" in self.initialization:
            self.grid_search_param = {
                "lr": [1e-1, 1e-2],
                "alpha": list(np.arange(1, 50, 50 / 10)),
                "beta": list(np.arange(1, 28, 28 / 10)),
            }
        else:
            self.grid_search_param = {
                "alpha": list(np.arange(1, 50, 50 / 20)),
                "beta": list(np.arange(1, 28, 28 / 20)),
            }
        print("Using Tip-Adapter")
        self.beta = beta
        self.alpha = alpha
        self.prototypes = nn.Parameter(base_text_features.clone())
        self.prototypes.requires_grad = False
        self.cache_keys = None
        self.cache_values = None

    def init_tipadapter(self, features_train: torch.Tensor, labels_train: torch.Tensor):
        self.cache_keys = nn.Parameter(features_train.clone())
        self.cache_keys.requires_grad = True
        self.cache_values = nn.Parameter(torch.nn.functional.one_hot(labels_train).clone().to(torch.float32))
        self.cache_values.requires_grad = False

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        return self.prototypes.unsqueeze(0).expand(n_samples, -1, -1)

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = params["alpha"]
        if "beta" in params:
            self.beta = params["beta"]
