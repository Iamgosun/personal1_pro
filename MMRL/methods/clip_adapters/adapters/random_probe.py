import torch
import torch.nn as nn

from .base import BaseAdapter


class RandomProbeAdapter(BaseAdapter):
    initialization_name = "RANDOM"
    adapter_kind = "prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using RANDOM initialization in Linear Probing")
        self.prototypes = nn.Parameter(
            torch.nn.init.kaiming_normal_(torch.empty_like(base_text_features))
        )

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes