import torch
import torch.nn as nn

from .base import BaseAdapter


class CrossModalProbeAdapter(BaseAdapter):
    initialization_name = "CrossModal"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using CrossModal for Linear Probing")
        self.prototypes = nn.Parameter(base_text_features.clone())

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        return self.prototypes.unsqueeze(0).expand(n_samples, -1, -1)
