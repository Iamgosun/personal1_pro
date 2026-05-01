from __future__ import annotations

from .adapters import (
    BayesAdapter,
    BayesianTaskResidualAdapter,
    CapelAdapter,
    ClipAdapterResidual,
    CrossModalProbeAdapter,
    DEBAAdapter,
    DreamBayesAdapter,
    HbaLrAdapter,
    RandomProbeAdapter,
    TaskResidualAdapter,
    TipAdapter,
    ZeroShotProbeAdapter,
    SBEAAdapter,
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

    if "TR" in init_upper and init_upper not in {
        "BTR",
        "BAYESIAN_TASK_RESIDUAL",
        "BAYESIAN_TASKRES",
        "BAYESIAN_TASK_RESIDUAL_ADAPTER",
        "TASKRES_BAYES",
        "BAYES_TASKRES",
    }:
        return TaskResidualAdapter(cfg, clip_model, base_text_features)

    if "CLIPA" in init_upper:
        return ClipAdapterResidual(cfg, clip_model, base_text_features)

    if "TIPA" in init_upper:
        return TipAdapter(cfg, clip_model, base_text_features)

    if "CROSSMODAL" in init_upper:
        return CrossModalProbeAdapter(cfg, clip_model, base_text_features)

    # Bayesian Task Residual should be checked before generic HBA/BAYES routing.
    if init_upper in {
        "BTR",
        "BAYESIAN_TASK_RESIDUAL",
        "BAYESIAN_TASKRES",
        "BAYESIAN_TASK_RESIDUAL_ADAPTER",
        "TASKRES_BAYES",
        "BAYES_TASKRES",
    }:
        return BayesianTaskResidualAdapter(cfg, clip_model, base_text_features)

    # HBA must be checked before the generic BAYES_ADAPTER branch.
    if init_upper in {"HBA", "HBA_LR", "HBALR", "HBA-LR"}:
        return HbaLrAdapter(cfg, clip_model, base_text_features)

    # DEBA is a separate method-level adapter, not the old BayesAdapter branch.
    if init_upper in {
        "DEBA",
        "DEBA_ADAPTER",
        "DEBA-P",
        "DEBA_P",
        "DEBA-J",
        "DEBA_J",
        "DEBA_INTERP",
        "DEBA-MIX",
        "DEBA_MIX",
    }:
        return DEBAAdapter(cfg, clip_model, base_text_features)

    if init_upper in {"SBEA", "SBEA_ARD", "SPARSE_BAYES_ENERGY_ADAPTER"}:
        return SBEAAdapter(cfg, clip_model, base_text_features)

    # IMPORTANT: check DREAM before the generic BAYES_ADAPTER substring branch.
    if init_upper in {"DREAM_BAYES_ADAPTER", "DREAMBAYES", "DREAM_BA"}:
        return DreamBayesAdapter(cfg, clip_model, base_text_features)

    if "BAYES_ADAPTER" in init_upper:
        return BayesAdapter(cfg, clip_model, base_text_features)

    raise NotImplementedError(f"Unknown clip adapter init: {init}")