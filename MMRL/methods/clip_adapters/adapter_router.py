from __future__ import annotations

from .adapters import (
    BayesAdapter,
    CapelAdapter,
    ECKAAdapter,
    ClipAdapterResidual,
    CrossModalProbeAdapter,
    RandomProbeAdapter,
    TaskResidualAdapter,
    TipAdapter,
    VncCapelAdapter,
    ZeroShotProbeAdapter,
)


def build_adapter(cfg, clip_model, base_text_features, classnames=None):
    init = str(cfg.CLIP_ADAPTERS.INIT)
    init_upper = init.upper()

    if init_upper == "VNC_CAPEL" or init_upper == "VNCCAPEL":
        if classnames is None:
            raise ValueError("VncCapelAdapter requires classnames.")
        return VncCapelAdapter(cfg, clip_model, base_text_features, classnames)

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

    if "BAYES_ADAPTER" in init_upper:
        return BayesAdapter(cfg, clip_model, base_text_features)

    if init_upper == "ECKA" or init_upper == "ECKA_CORE" or init_upper == "ECKA_CAL":
        return ECKAAdapter(cfg, clip_model, base_text_features)

    raise NotImplementedError(f"Unknown clip adapter init: {init}")