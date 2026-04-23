import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseAdapter


class TipAdapter(BaseAdapter):
    initialization_name = "TipA"
    adapter_kind = "prototype"
    uses_cache = True

    def __init__(
        self,
        cfg,
        clip_model,
        base_text_features: torch.Tensor,
        beta: float = 1.0,
        alpha: float = 1.0,
    ):
        super().__init__(cfg, clip_model, base_text_features)

        init_name = str(self.initialization).upper()
        self.finetune_cache = ("-F-" in init_name) or init_name.endswith("-F") or ("TIPA-F" in init_name)

        if self.finetune_cache:
            self.grid_search_param = {
                "lr": [1e-1, 1e-2],
                "alpha": list(np.arange(1, 50, 50 / 10)),
                "beta": list(np.arange(1, 28, 28 / 10)),
            }
            print("Using Tip-Adapter-F")
        else:
            self.grid_search_param = {
                "alpha": list(np.arange(1, 50, 50 / 20)),
                "beta": list(np.arange(1, 28, 28 / 20)),
            }
            print("Using Tip-Adapter")

        self.beta = float(beta)
        self.alpha = float(alpha)

        self.cache_keys = None
        self.cache_values = None

    def get_prototypes(self) -> torch.Tensor:
        # Tip-Adapter keeps the zero-shot classifier fixed.
        return self.base_text_features

    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        features_train = features_train.to(self.base_text_features.device).to(torch.float32)
        labels_train = labels_train.to(self.base_text_features.device).to(torch.long)

        if features_train.ndim != 2:
            raise ValueError(
                f"TipAdapter expects cached train features with shape [N, D], got {tuple(features_train.shape)}"
            )

        n_classes = int(self.base_text_features.shape[0])
        cache_keys = F.normalize(features_train, dim=-1)
        cache_values = F.one_hot(labels_train, num_classes=n_classes).to(cache_keys.dtype)

        if self.finetune_cache:
            self.cache_keys = torch.nn.Parameter(cache_keys)
        else:
            self.cache_keys = cache_keys

        self.cache_values = cache_values

    def cache_logits(self, features: torch.Tensor):
        if self.cache_keys is None or self.cache_values is None:
            return None

        features = F.normalize(features, dim=-1)

        if isinstance(self.cache_keys, torch.nn.Parameter):
            cache_keys = F.normalize(self.cache_keys, dim=-1)
        else:
            cache_keys = F.normalize(self.cache_keys, dim=-1)

        affinity = features @ cache_keys.t()
        cache_logits = torch.exp(self.beta * (affinity - 1.0)) @ self.cache_values.to(features.dtype)
        return self.alpha * cache_logits

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "beta" in params:
            self.beta = float(params["beta"])