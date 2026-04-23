import torch
import torch.nn as nn

from .base import BaseAdapter


class ZeroShotProbeAdapter(BaseAdapter):
    initialization_name = "ZS"
    adapter_kind = "prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using Zero-Shot initialization in Linear Probing")
        self.prototypes = nn.Parameter(base_text_features.clone())

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes