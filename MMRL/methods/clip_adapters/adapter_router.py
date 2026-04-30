from __future__ import annotations

from .adapters import (
    BayesAdapter,
    CapelAdapter,
    ClipAdapterResidual,
    CrossModalProbeAdapter,
    DreamBayesAdapter,
    RandomProbeAdapter,
    TaskResidualAdapter,
    TipAdapter,
    ZeroShotProbeAdapter,
)


def build_adapter(cfg, clip_model, base_text_features, classnames=None):
    init = str(cfg.CLIP_ADAPTERS.INIT)
    init_upper = init.upper()

    if init_upper == "CAPEL":
        if classnames is None:
            raise ValueError("CapelAdapter requires classnames.")
        return CapelAdapter(cfg, clip_model, base_text_features, classnames)

    if init_upper == "RANDOM":
        return RandomProbeAdapter(cfg, clip_model, base_text_features)

    if init_upper.startswith("ZS") or init_upper == "ZS":
        return ZeroShotProbeAdapter(cfg, clip_model, base_text_features)

    if "TR" in init_upper:
        return TaskResidualAdapter(cfg, clip_model, base_text_features)

    if "CLIPA" in init_upper:
        return ClipAdapterResidual(cfg, clip_model, base_text_features)

    if "TIPA" in init_upper:
        return TipAdapter(cfg, clip_model, base_text_features)

    if "CROSSMODAL" in init_upper:
        return CrossModalProbeAdapter(cfg, clip_model, base_text_features)

    # IMPORTANT: check DREAM before the generic BAYES_ADAPTER substring branch.
    # Otherwise "DREAM_BAYES_ADAPTER" is swallowed by the plain BayesAdapter branch.
    if init_upper in {"DREAM_BAYES_ADAPTER", "DREAMBAYES", "DREAM_BA"}:
        return DreamBayesAdapter(cfg, clip_model, base_text_features)

    if "BAYES_ADAPTER" in init_upper:
        return BayesAdapter(cfg, clip_model, base_text_features)

    raise NotImplementedError(f"Unknown clip adapter init: {init}")
