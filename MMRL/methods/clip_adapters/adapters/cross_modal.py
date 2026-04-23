import torch
import torch.nn as nn

from .base import BaseAdapter


class CrossModalProbeAdapter(BaseAdapter):
    initialization_name = "CrossModal"
    adapter_kind = "prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using CrossModal Probe Adapter")

        feat_dim = int(base_text_features.shape[-1])

        # Distinct from ZS probe:
        # - trainable class prototypes
        # - trainable image-side linear projector initialized to identity
        self.prototypes = nn.Parameter(base_text_features.clone())
        self.feature_proj = nn.Linear(feat_dim, feat_dim, bias=False)

        with torch.no_grad():
            self.feature_proj.weight.copy_(torch.eye(feat_dim, device=base_text_features.device, dtype=base_text_features.dtype))

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.feature_proj(features)