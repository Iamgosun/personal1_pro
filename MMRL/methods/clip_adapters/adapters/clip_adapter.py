import numpy as np
import torch
import torch.nn as nn

from .base import BaseAdapter


class ClipAdapterResidual(BaseAdapter):
    initialization_name = "ClipA"
    adapter_kind = "prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, ratio: float = 0.2):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using CLIP-Adapter")

        feat_dim = int(base_text_features.shape[-1])
        hidden_dim = max(1, feat_dim // 4)

        self.grid_search_param = {
            "lr": [1e-1, 1e-2, 1e-3],
            "ratio": list(np.arange(0.2, 1.0, 0.2)),
        }

        self.ratio = float(ratio)

        # CLIP-Adapter keeps the classifier anchored to the zero-shot text prototypes.
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def get_prototypes(self) -> torch.Tensor:
        return self.base_text_features

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.mlp(features)
        return self.ratio * adapted + (1.0 - self.ratio) * features

    def reset_hparams(self, params):
        if "ratio" in params:
            self.ratio = float(params["ratio"])