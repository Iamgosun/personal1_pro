import torch
import torch.nn as nn

from .base import BaseAdapter


class CrossModalProbeAdapter(BaseAdapter):
    """
    CLAP/CrossModal-aligned adapter.

    Important:
    - CLAP CrossModal does NOT add an image-side projection.
    - The semantic difference comes from training data:
      image features + text prompt features are concatenated in the cache executor.
    """

    initialization_name = "CrossModal"
    adapter_kind = "prototype"
    uses_cross_modal = True

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor):
        super().__init__(cfg, clip_model, base_text_features)
        print("Using CrossModal for Linear Probing")

        # CLAP init_MultiModal(): prototypes = base_text_features.clone()
        self.prototypes = nn.Parameter(base_text_features.clone())

    def get_prototypes(self) -> torch.Tensor:
        return self.prototypes

    def reset_for_grid(self, params, features_train=None, labels_train=None):
        # Reinitialize between grid-search candidates to avoid parameter carry-over.
        if "prototypes" in self._parameters:
            del self._parameters["prototypes"]
        self.prototypes = nn.Parameter(self.base_text_features.clone())