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

        self.feat_dim = int(base_text_features.shape[-1])
        self.hidden_dim = max(1, self.feat_dim // 4)

        self.grid_search_param = {
            "lr": [1e-1, 1e-2, 1e-3],
            "ratio": list(np.arange(0.2, 1.0, 0.2)),
        }

        self.ratio = float(ratio)
        self.mlp = self._build_mlp()

    def _build_mlp(self):
        return nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feat_dim, bias=False),
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

    def reset_for_grid(self, params, features_train=None, labels_train=None):
        self.reset_hparams(params)

        # CLAP reset_hyperparams() re-calls init_clipA(), which rebuilds the MLP.
        self.mlp = self._build_mlp().to(self.base_text_features.device)