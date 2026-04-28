import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


class TipAdapter(BaseAdapter):
    """
    CLAP-aligned Tip-Adapter.

    Important semantic choices:
    - In jusiro/CLAP, both TipA and TipA-f- register cache_keys as nn.Parameter.
      Plain TipA is still run for one epoch; TipA-f- is run for longer.
    - cache_values are frozen one-hot parameters in CLAP.
    - cache affinity in CLAP uses raw features @ normalized cache_keys.T.
      This is preserved by default through CLAP_TIPA_RAW_AFFINITY=True.
    """

    initialization_name = "TipA"
    adapter_kind = "prototype"
    uses_cache = True
    is_tip_adapter = True

    def __init__(
        self,
        cfg,
        clip_model,
        base_text_features: torch.Tensor,
        beta: float = 1.0,
        alpha: float = 1.0,
    ):
        super().__init__(cfg, clip_model, base_text_features)

        init_name = str(self.initialization)
        init_upper = init_name.upper()

        self.finetune_cache = (
            ("-F-" in init_upper)
            or init_upper.endswith("-F")
            or ("TIPA-F" in init_upper)
            or ("TIPA_F" in init_upper)
        )

        clip_cfg = getattr(cfg, "CLIP_ADAPTERS", None)

        # CLAP behavior: TipA cache keys are trainable parameters too.
        self.clap_trainable_cache = bool(
            getattr(clip_cfg, "CLAP_TIPA_TRAINABLE_CACHE", True)
        )

        # CLAP forward_tipadapter uses raw features for cache affinity.
        # Official Tip-Adapter formula usually normalizes query features first.
        self.clap_raw_affinity = bool(
            getattr(clip_cfg, "CLAP_TIPA_RAW_AFFINITY", True)
        )

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

        # These are filled by build_cache().
        self.cache_keys = None
        self.cache_values = None

    def _clear_cache_state(self):
        """Safely remove old Parameter/buffer cache state before rebuilding."""
        for name in ("cache_keys", "cache_values"):
            if name in self._parameters:
                del self._parameters[name]
            if name in self._buffers:
                del self._buffers[name]
            object.__setattr__(self, name, None)

    def get_prototypes(self) -> torch.Tensor:
        # CLAP TipA keeps zero-shot classifier fixed.
        return self.base_text_features

    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        self._clear_cache_state()

        device = self.base_text_features.device
        features_train = features_train.to(device=device, dtype=torch.float32)
        labels_train = labels_train.to(device=device, dtype=torch.long)

        if features_train.ndim != 2:
            raise ValueError(
                f"TipAdapter expects cached train features [N, D], got {tuple(features_train.shape)}"
            )

        n_classes = int(self.base_text_features.shape[0])
        cache_values = F.one_hot(labels_train, num_classes=n_classes).to(torch.float32)

        # CLAP stores raw features as trainable cache_keys and normalizes in forward.
        # If user disables CLAP_TIPA_TRAINABLE_CACHE, keep a non-trainable buffer.
        trainable_cache = self.finetune_cache or self.clap_trainable_cache

        if trainable_cache:
            self.cache_keys = nn.Parameter(features_train.clone())
            self.cache_values = nn.Parameter(cache_values.clone(), requires_grad=False)
        else:
            self.register_buffer(
                "cache_keys",
                F.normalize(features_train, dim=-1).clone(),
                persistent=True,
            )
            self.register_buffer(
                "cache_values",
                cache_values.clone(),
                persistent=True,
            )

    def cache_logits(self, features: torch.Tensor):
        if self.cache_keys is None or self.cache_values is None:
            return None

        cache_keys = F.normalize(self.cache_keys.to(features.device), dim=-1)
        cache_values = self.cache_values.to(device=features.device, dtype=torch.float32)

        if self.clap_raw_affinity:
            query = features.to(torch.float32)
        else:
            query = F.normalize(features.to(torch.float32), dim=-1)

        affinity = query @ cache_keys.t().to(torch.float32)

        # exp(-beta * (1 - affinity)) == exp(beta * (affinity - 1))
        cache_logits = torch.exp(self.beta * (affinity - 1.0)) @ cache_values
        return self.alpha * cache_logits.to(features.dtype)

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "beta" in params:
            self.beta = float(params["beta"])

    def reset_for_grid(self, params, features_train=None, labels_train=None):
        self.reset_hparams(params)
        if features_train is not None and labels_train is not None:
            self.build_cache(features_train, labels_train)