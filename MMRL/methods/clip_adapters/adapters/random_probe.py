import torch
import torch.nn as nn

from .base import BaseAdapter


class RandomProbeAdapter(BaseAdapter):
    initialization_name = "RANDOM"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using RANDOM initialization in Linear Probing")
        self.prototypes = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty_like(base_text_features)))

    def forward(self, n_samples: int = 1) -> torch.Tensor:
        return self.prototypes.unsqueeze(0).expand(n_samples, -1, -1)
