import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAdapter


class TipAdapter(BaseAdapter):
    """
    Official Tip-Adapter / Tip-Adapter-F aligned implementation.

    This version is aligned with the *actual* current repository execution path:

    - CacheExecutor.run_epoch() sets model mode to eval before cache-mode training.
      Therefore a training-flag based dummy grad is not reliable for plain TipA.
    - CacheExecutor.forward_backward() already has a closed_form_adapter branch
      that skips backward/optimizer step.
    - Plain TipA is a frozen-cache, non-training method in the official algorithm,
      so it should use closed_form_adapter=True.
    - TipA-f- is trainable and must keep closed_form_adapter=False.

    Shape convention:
      cache_keys:   [N, D]
      cache_values: [N, C]
    Official Tip-Adapter stores cache_keys as [D, N]; using cache_keys.T below
    is mathematically equivalent.
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

        init_upper = str(self.initialization).upper()
        self.finetune_cache = (
            ("-F-" in init_upper)
            or init_upper.endswith("-F")
            or ("TIPA-F" in init_upper)
            or ("TIPA_F" in init_upper)
        )

        clip_cfg = getattr(cfg, "CLIP_ADAPTERS", None)

        # Official path: False means normalize image/query features before affinity.
        self.raw_affinity = bool(
            getattr(clip_cfg, "CLAP_TIPA_RAW_AFFINITY", False)
        )

        # Official plain TipA: frozen cache. Keep this only for legacy ablations.
        self.plain_trainable_cache = bool(
            getattr(clip_cfg, "CLAP_TIPA_TRAINABLE_CACHE", False)
        )

        # The key compatibility fix:
        # plain official TipA is closed-form/no-backward in cache mode.
        # Your CacheExecutor already supports this branch.
        self.closed_form_adapter = bool(
            (not self.finetune_cache) and (not self.plain_trainable_cache)
        )
        self.requires_training = not self.closed_form_adapter

        if clip_cfg is not None and hasattr(clip_cfg, "TIPA_ALPHA"):
            alpha = getattr(clip_cfg, "TIPA_ALPHA")
        if clip_cfg is not None and hasattr(clip_cfg, "TIPA_BETA"):
            beta = getattr(clip_cfg, "TIPA_BETA")

        self.alpha = float(alpha)
        self.beta = float(beta)

        # Official Tip-Adapter-F uses fixed alpha/beta during cache-key training.
        # HPO alpha/beta affect eval/test logits, not the training trajectory.
        self.train_alpha = float(
            getattr(clip_cfg, "TIPA_TRAIN_ALPHA", self.alpha)
            if clip_cfg is not None
            else self.alpha
        )
        self.train_beta = float(
            getattr(clip_cfg, "TIPA_TRAIN_BETA", self.beta)
            if clip_cfg is not None
            else self.beta
        )

        # Save one-hot cache values in fp16 to reduce checkpoint size.
        # They are cast to fp32 inside cache_logits().
        self.cache_values_fp16 = bool(
            getattr(clip_cfg, "TIPA_CACHE_VALUES_FP16", True)
            if clip_cfg is not None
            else True
        )

        if self.finetune_cache:
            self.grid_search_param = {
                "lr": [1e-3],
                "alpha": list(np.arange(1, 50, 50 / 10)),
                "beta": list(np.arange(1, 28, 28 / 10)),
            }
            print("Using Tip-Adapter-F (official-aligned)")
        else:
            self.grid_search_param = {
                "alpha": list(np.arange(1, 50, 50 / 20)),
                "beta": list(np.arange(1, 28, 28 / 20)),
            }
            print("Using Tip-Adapter (official-aligned)")

        # Kept harmlessly so optimizer construction never sees an empty adapter
        # parameter list in unusual legacy settings. In closed-form plain TipA,
        # CacheExecutor skips backward/step.
        self._tipa_noop = nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=True)

    def _drop_module_tensor(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        if name in self._buffers:
            del self._buffers[name]
        if name in self._modules:
            del self._modules[name]
        if name in self.__dict__:
            del self.__dict__[name]

    def _clear_cache_state(self) -> None:
        self._drop_module_tensor("cache_keys")
        self._drop_module_tensor("cache_values")

    def _cache_is_ready(self) -> bool:
        return (
            ("cache_keys" in self._parameters or "cache_keys" in self._buffers)
            and ("cache_values" in self._parameters or "cache_values" in self._buffers)
        )

    def get_prototypes(self) -> torch.Tensor:
        # Tip-Adapter keeps the zero-shot CLIP classifier fixed.
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

        # Official cache keys are normalized at construction. For TipA-f- they
        # are then optimized freely, without per-forward re-normalization.
        cache_keys = F.normalize(features_train, dim=-1)

        values_dtype = torch.float16 if self.cache_values_fp16 else torch.float32
        cache_values = F.one_hot(labels_train, num_classes=n_classes).to(values_dtype)

        trainable_cache = bool(self.finetune_cache or self.plain_trainable_cache)

        if trainable_cache:
            self.cache_keys = nn.Parameter(cache_keys.clone(), requires_grad=True)
        else:
            self.register_buffer("cache_keys", cache_keys.clone(), persistent=True)

        self.register_buffer("cache_values", cache_values.clone(), persistent=True)

    def _current_alpha_beta(self, training: bool) -> tuple[float, float]:
        if training and self.finetune_cache:
            return self.train_alpha, self.train_beta
        return self.alpha, self.beta

    def cache_logits(self, features: torch.Tensor):
        if not self._cache_is_ready():
            return None

        cache_keys = self.cache_keys.to(device=features.device, dtype=torch.float32)
        cache_values = self.cache_values.to(device=features.device, dtype=torch.float32)

        if self.raw_affinity:
            query = features.to(torch.float32)
        else:
            query = F.normalize(features.to(torch.float32), dim=-1)

        affinity = query @ cache_keys.t()

        alpha, beta = self._current_alpha_beta(training=bool(self.training))

        # Official formula:
        #   exp(-beta * (1 - affinity)) @ cache_values
        # equals:
        #   exp(beta * (affinity - 1)) @ cache_values
        cache_logits = torch.exp(float(beta) * (affinity - 1.0)) @ cache_values
        cache_logits = float(alpha) * cache_logits

        return cache_logits.to(features.dtype)

    def extra_loss(self):
        # No extra numerical objective is needed. This method is intentionally
        # kept for compatibility with ClipAdaptersLoss, but plain TipA now uses
        # CacheExecutor.closed_form_adapter and therefore does not backward.
        return None

    def reset_hparams(self, params):
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "beta" in params:
            self.beta = float(params["beta"])

        if "CLIP_ADAPTERS.TIPA_ALPHA" in params:
            self.alpha = float(params["CLIP_ADAPTERS.TIPA_ALPHA"])
        if "CLIP_ADAPTERS.TIPA_BETA" in params:
            self.beta = float(params["CLIP_ADAPTERS.TIPA_BETA"])

    def reset_for_grid(self, params, features_train=None, labels_train=None):
        self.reset_hparams(params)
        if features_train is not None and labels_train is not None:
            self.build_cache(features_train, labels_train)
