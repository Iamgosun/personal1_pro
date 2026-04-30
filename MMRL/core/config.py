from __future__ import annotations

from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default

from core.overrides import resolve_method_dataset_overrides


def _copy_mmrl_family(src, sec):
    sec.PREC = src.PREC
    sec.ALPHA = src.ALPHA
    sec.REG_WEIGHT = src.REG_WEIGHT
    sec.N_REP_TOKENS = src.N_REP_TOKENS
    sec.REP_LAYERS = src.REP_LAYERS
    sec.REP_DIM = src.REP_DIM
    if hasattr(src, "LOSS_MAIN_WEIGHT"):
        sec.LOSS_MAIN_WEIGHT = src.LOSS_MAIN_WEIGHT
    if hasattr(src, "LOSS_REP_WEIGHT"):
        sec.LOSS_REP_WEIGHT = src.LOSS_REP_WEIGHT
    if hasattr(src, "LOSS_FUSE_WEIGHT"):
        sec.LOSS_FUSE_WEIGHT = src.LOSS_FUSE_WEIGHT


def _as_legacy_clipadapter(cfg):
    if not hasattr(cfg.TRAINER, "ClipADAPTER"):
        cfg.TRAINER.ClipADAPTER = CN()
    sec = cfg.TRAINER.ClipADAPTER
    cad = cfg.CLIP_ADAPTERS
    sec.PREC = cad.PREC
    sec.INIT = cad.INIT
    sec.CONSTRAINT = cad.CONSTRAINT
    sec.ENHANCED_BASE = cad.ENHANCED_BASE
    sec.TYPE = cad.TYPE
    sec.ALLOW_CACHE = cad.ALLOW_CACHE
    sec.GRID_SEARCH = cad.GRID_SEARCH
    sec.GRID_SEARCH_SPLIT = cad.GRID_SEARCH_SPLIT
    sec.CLAP_TIPA_TRAINABLE_CACHE = cad.CLAP_TIPA_TRAINABLE_CACHE
    sec.CLAP_TIPA_RAW_AFFINITY = cad.CLAP_TIPA_RAW_AFFINITY
    sec.CLAP_TIPA_ONE_EPOCH = cad.CLAP_TIPA_ONE_EPOCH
    sec.TIPA_F_GRID_EPOCHS = cad.TIPA_F_GRID_EPOCHS
    sec.CROSS_MODAL_RESAMPLE_TEXT = cad.CROSS_MODAL_RESAMPLE_TEXT
    sec.CROSS_MODAL_EPOCH_SUBSAMPLE = cad.CROSS_MODAL_EPOCH_SUBSAMPLE

    sec.CACHE_REPS = cad.CACHE_REPS
    sec.CACHE_TRAIN_AUG = cad.CACHE_TRAIN_AUG
    sec.CACHE_AGGREGATION = cad.CACHE_AGGREGATION
    sec.CACHE_POOL_EPOCH_SUBSAMPLE = cad.CACHE_POOL_EPOCH_SUBSAMPLE


    if hasattr(cad, "CACHE_AGGREGATION"):
        sec.CACHE_AGGREGATION = cad.CACHE_AGGREGATION
    if hasattr(cad, "CACHE_POOL_EPOCH_SUBSAMPLE"):
        sec.CACHE_POOL_EPOCH_SUBSAMPLE = cad.CACHE_POOL_EPOCH_SUBSAMPLE

    sec.ONLINE_PREFIT_REPS = cad.ONLINE_PREFIT_REPS
    sec.ONLINE_PREFIT_TRAIN_AUG = cad.ONLINE_PREFIT_TRAIN_AUG

    if hasattr(cad, "CAPEL_PROMPT_BANK"):
        sec.CAPEL_PROMPT_BANK = cad.CAPEL_PROMPT_BANK
        sec.CAPEL_PROMPTS_PER_CLASS = cad.CAPEL_PROMPTS_PER_CLASS
        sec.CAPEL_PC_LAMBDA = cad.CAPEL_PC_LAMBDA
        sec.CAPEL_STRICT_PROMPT_BANK = cad.CAPEL_STRICT_PROMPT_BANK
        sec.CAPEL_FALLBACK_ORDER = cad.CAPEL_FALLBACK_ORDER
        sec.CAPEL_USE_FEATURE_CACHE = cad.CAPEL_USE_FEATURE_CACHE
        sec.CAPEL_REBUILD_FEATURE_CACHE = cad.CAPEL_REBUILD_FEATURE_CACHE
        sec.CAPEL_FEATURE_CACHE_DIR = cad.CAPEL_FEATURE_CACHE_DIR

    if hasattr(cad, "VNC_CAPEL_VNC_LAMBDA"):
        sec.VNC_CAPEL_VNC_LAMBDA = cad.VNC_CAPEL_VNC_LAMBDA

    if hasattr(cad, "PP_PROKER_BETA"):
        sec.PP_PROKER_BETA = cad.PP_PROKER_BETA
        sec.PP_PROKER_LAMBDA = cad.PP_PROKER_LAMBDA
        sec.PP_PROKER_GP_DELTA = cad.PP_PROKER_GP_DELTA
        sec.PP_PROKER_USE_MC = cad.PP_PROKER_USE_MC
        sec.PP_PROKER_MC_SAMPLES = cad.PP_PROKER_MC_SAMPLES
        sec.PP_PROKER_RHO = cad.PP_PROKER_RHO
        sec.PP_PROKER_TAU = cad.PP_PROKER_TAU
        sec.PP_PROKER_RETURN_LOG_PROBS = cad.PP_PROKER_RETURN_LOG_PROBS
        sec.PP_PROKER_VARIANCE_JITTER = cad.PP_PROKER_VARIANCE_JITTER
        sec.PP_PROKER_MEAN_RESIDUAL_SCALE = cad.PP_PROKER_MEAN_RESIDUAL_SCALE


def _as_legacy_mmrl(cfg):
    if not hasattr(cfg.TRAINER, "MMRL"):
        cfg.TRAINER.MMRL = CN()
    sec = cfg.TRAINER.MMRL
    _copy_mmrl_family(cfg.MMRL, sec)


def _as_legacy_mmrl_mix(cfg):
    if not hasattr(cfg.TRAINER, "MMRLMix"):
        cfg.TRAINER.MMRLMix = CN()
    sec = cfg.TRAINER.MMRLMix
    _copy_mmrl_family(cfg.MMRL_MIX, sec)


def _as_legacy_mmrlpp(cfg):
    if not hasattr(cfg.TRAINER, "MMRLpp"):
        cfg.TRAINER.MMRLpp = CN()
    sec = cfg.TRAINER.MMRLpp
    src = cfg.MMRLPP
    sec.PREC = src.PREC
    sec.ALPHA = src.ALPHA
    sec.BETA = src.BETA
    sec.REG_WEIGHT = src.REG_WEIGHT
    sec.N_REP_TOKENS = src.N_REP_TOKENS
    sec.REP_LAYERS = src.REP_LAYERS
    sec.REP_DIM = src.REP_DIM
    sec.PROJ_LORA_DIM = src.PROJ_LORA_DIM
    sec.RES_LORA_DIM = src.RES_LORA_DIM



def _as_legacy_bayes_mmrl(cfg):
    if not hasattr(cfg.TRAINER, "BayesMMRL"):
        cfg.TRAINER.BayesMMRL = CN()
    sec = cfg.TRAINER.BayesMMRL
    src = cfg.BAYES_MMRL

    sec.PREC = src.PREC
    sec.ALPHA = src.ALPHA
    sec.REG_WEIGHT = src.REG_WEIGHT
    sec.N_REP_TOKENS = src.N_REP_TOKENS
    sec.REP_LAYERS = src.REP_LAYERS
    sec.REP_DIM = src.REP_DIM

    sec.KL_WARMUP_EPOCHS = src.KL_WARMUP_EPOCHS
    
    sec.BAYES_TARGET = src.BAYES_TARGET

    sec.N_MC_TRAIN = src.N_MC_TRAIN
    sec.N_MC_TEST = src.N_MC_TEST
    sec.EVAL_MODE = src.EVAL_MODE
    sec.EVAL_USE_POSTERIOR_MEAN = src.EVAL_USE_POSTERIOR_MEAN
    sec.EVAL_AGGREGATION = src.EVAL_AGGREGATION

    sec.REP_SIGMA_MODE = src.REP_SIGMA_MODE
    sec.REP_PRIOR_MODE = src.REP_PRIOR_MODE
    sec.REP_PRIOR_STD = src.REP_PRIOR_STD
    sec.REP_KL_WEIGHT = src.REP_KL_WEIGHT
    sec.REP_MN_ENFORCE_TRACE = src.REP_MN_ENFORCE_TRACE
    sec.REP_MN_LOWRANK_RANK = src.REP_MN_LOWRANK_RANK

    sec.CLIP_PRIOR_SCALE = src.CLIP_PRIOR_SCALE
    sec.CLIP_PRIOR_BLEND = src.CLIP_PRIOR_BLEND

    sec.PROJ_REP_SIGMA_MODE = src.PROJ_REP_SIGMA_MODE
    sec.PROJ_REP_PRIOR_MODE = src.PROJ_REP_PRIOR_MODE
    sec.PROJ_REP_PRIOR_STD = src.PROJ_REP_PRIOR_STD
    sec.PROJ_REP_KL_WEIGHT = src.PROJ_REP_KL_WEIGHT

    # lightweight backward-compatible aliases
    sec.KL_WEIGHT = src.REP_KL_WEIGHT
    sec.PRIOR_STD = src.REP_PRIOR_STD
    sec.SIGMA_MODE = src.REP_SIGMA_MODE

def _as_legacy_vcrm_mmrl(cfg):
    if not hasattr(cfg.TRAINER, "VCRMMMRL"):
        cfg.TRAINER.VCRMMMRL = CN()

    sec = cfg.TRAINER.VCRMMMRL
    src = cfg.VCRM_MMRL

    sec.PREC = src.PREC
    sec.ALPHA = src.ALPHA
    sec.REG_WEIGHT = src.REG_WEIGHT
    sec.N_REP_TOKENS = src.N_REP_TOKENS
    sec.REP_LAYERS = src.REP_LAYERS
    sec.REP_DIM = src.REP_DIM

    sec.VCRM_CONTEXT_LAYER = src.VCRM_CONTEXT_LAYER
    sec.VCRM_HIDDEN_DIM = src.VCRM_HIDDEN_DIM
    sec.VCRM_ETA = src.VCRM_ETA
    sec.VCRM_DETACH_CONTEXT = src.VCRM_DETACH_CONTEXT
    sec.VCRM_MOD_WEIGHT = src.VCRM_MOD_WEIGHT



def _sync_active_mmrl_family(cfg):
    if cfg.METHOD.NAME == "MMRLMix":
        _copy_mmrl_family(cfg.MMRL_MIX, cfg.MMRL)


def get_refactor_defaults():
    cfg = get_cfg_default()
    cfg.TRAINER.NAME = "RefactorRunner"

    cfg.METHOD = CN()
    cfg.METHOD.NAME = "MMRL"
    cfg.METHOD.FAMILY = "internal_adaptation"
    cfg.METHOD.EXEC_MODE = "online"
    cfg.METHOD.TAG = "default"

    cfg.PROTOCOL = CN()
    cfg.PROTOCOL.NAME = "B2N"
    cfg.PROTOCOL.PHASE = "train_base"

    cfg.RUNTIME = CN()
    cfg.RUNTIME.USE_DASSL_TRAINER_BRIDGE = True
    cfg.RUNTIME.OUTPUT_ROOT = "output_refactor"

    cfg.MMRL = CN()
    cfg.MMRL.PREC = "amp"
    cfg.MMRL.ALPHA = 0.7
    cfg.MMRL.REG_WEIGHT = 1.0
    cfg.MMRL.N_REP_TOKENS = 5
    cfg.MMRL.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.MMRL.REP_DIM = 512

    cfg.VCRM_MMRL = CN()
    cfg.VCRM_MMRL.PREC = "amp"
    cfg.VCRM_MMRL.ALPHA = 0.7
    cfg.VCRM_MMRL.REG_WEIGHT = 0.5
    cfg.VCRM_MMRL.N_REP_TOKENS = 5
    cfg.VCRM_MMRL.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.VCRM_MMRL.REP_DIM = 512

    cfg.VCRM_MMRL.VCRM_CONTEXT_LAYER = 3
    cfg.VCRM_MMRL.VCRM_HIDDEN_DIM = 512
    cfg.VCRM_MMRL.VCRM_ETA = 0.1
    cfg.VCRM_MMRL.VCRM_DETACH_CONTEXT = True
    cfg.VCRM_MMRL.VCRM_MOD_WEIGHT = 0.0



    cfg.MMRL_MIX = CN()
    cfg.MMRL_MIX.PREC = "amp"
    cfg.MMRL_MIX.ALPHA = 0.7
    cfg.MMRL_MIX.REG_WEIGHT = 1.0
    cfg.MMRL_MIX.LOSS_MAIN_WEIGHT = 1.0
    cfg.MMRL_MIX.LOSS_REP_WEIGHT = 1.0
    cfg.MMRL_MIX.LOSS_FUSE_WEIGHT = 0.2
    cfg.MMRL_MIX.N_REP_TOKENS = 5
    cfg.MMRL_MIX.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.MMRL_MIX.REP_DIM = 512

    cfg.MMRLPP = CN()
    cfg.MMRLPP.PREC = "amp"
    cfg.MMRLPP.ALPHA = 0.7
    cfg.MMRLPP.BETA = 0.9
    cfg.MMRLPP.REG_WEIGHT = 1.0
    cfg.MMRLPP.N_REP_TOKENS = 5
    cfg.MMRLPP.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.MMRLPP.REP_DIM = 512
    cfg.MMRLPP.PROJ_LORA_DIM = 64
    cfg.MMRLPP.RES_LORA_DIM = 4

    cfg.BAYES_MMRL = CN()
    cfg.BAYES_MMRL.PREC = "amp"
    cfg.BAYES_MMRL.ALPHA = 0.7
    cfg.BAYES_MMRL.REG_WEIGHT = 1.0
    cfg.BAYES_MMRL.N_REP_TOKENS = 5
    cfg.BAYES_MMRL.REP_LAYERS = [6, 7, 8, 9, 10, 11, 12]
    cfg.BAYES_MMRL.REP_DIM = 512

    cfg.BAYES_MMRL.KL_WARMUP_EPOCHS = 5
    # which parameter block is Bayesian
    # "rep_tokens" -> schemes A/B
    # "proj_rep"   -> scheme C
    cfg.BAYES_MMRL.BAYES_TARGET = "rep_tokens"

    cfg.BAYES_MMRL.N_MC_TRAIN = 3
    cfg.BAYES_MMRL.N_MC_TEST = 5

    # eval modes
    cfg.BAYES_MMRL.EVAL_MODE = "mc_predictive"   # posterior_mean | mc_predictive | mean_plus_mc
    cfg.BAYES_MMRL.EVAL_USE_POSTERIOR_MEAN = False
    cfg.BAYES_MMRL.EVAL_AGGREGATION = "prob_mean"   # prob_mean | logit_mean

    # ----- schemes A/B: Bayes on representation tokens R -----
    # q_0(R) = p(R) is enforced automatically in code
    cfg.BAYES_MMRL.REP_SIGMA_MODE = "global"     # global | per_token
    cfg.BAYES_MMRL.REP_PRIOR_MODE = "zero"       # zero | clip_joint
    cfg.BAYES_MMRL.REP_PRIOR_STD = 0.05
    cfg.BAYES_MMRL.REP_KL_WEIGHT = 5e-4

    # matrix-normal specific defaults
    cfg.BAYES_MMRL.REP_MN_ENFORCE_TRACE = True
    cfg.BAYES_MMRL.REP_MN_LOWRANK_RANK = 8

    # used only when REP_PRIOR_MODE == clip_joint
    cfg.BAYES_MMRL.CLIP_PRIOR_SCALE = 0.05
    cfg.BAYES_MMRL.CLIP_PRIOR_BLEND = 0.5

    # ----- scheme C: Bayes on proj_rep -----
    # q_0(W) = p(W) is enforced automatically in code
    # self_proj_rep: prior mean = deterministic MMRL proj_rep
    # clip_proj    : prior mean = CLIP visual projection head proj
    cfg.BAYES_MMRL.PROJ_REP_SIGMA_MODE = "row"   # global | row
    cfg.BAYES_MMRL.PROJ_REP_PRIOR_MODE = "clip_proj"   # self_proj_rep | clip_proj
    cfg.BAYES_MMRL.PROJ_REP_PRIOR_STD = 0.01
    cfg.BAYES_MMRL.PROJ_REP_KL_WEIGHT = 1e-6

    # lightweight backward-compatible aliases
    cfg.BAYES_MMRL.KL_WEIGHT = cfg.BAYES_MMRL.REP_KL_WEIGHT
    cfg.BAYES_MMRL.PRIOR_STD = cfg.BAYES_MMRL.REP_PRIOR_STD
    cfg.BAYES_MMRL.SIGMA_MODE = cfg.BAYES_MMRL.REP_SIGMA_MODE

    cfg.CLIP_ADAPTERS = CN()
    cfg.CLIP_ADAPTERS.PREC = "fp32"
    cfg.CLIP_ADAPTERS.TYPE = "MP"
    cfg.CLIP_ADAPTERS.INIT = "ZS"
    cfg.CLIP_ADAPTERS.CONSTRAINT = "none"
    cfg.CLIP_ADAPTERS.ENHANCED_BASE = "none"


    cfg.CLIP_ADAPTERS.ALLOW_CACHE = True

    # Cache extraction defaults. For closer CLAP behavior, override CACHE_REPS=20.
    cfg.CLIP_ADAPTERS.CACHE_REPS = 1
    cfg.CLIP_ADAPTERS.CACHE_TRAIN_AUG = True
    # "pool" matches jusiro/CLAP-style repeated augmentation features.
    # "mean" keeps the old behavior: average repeated features into one feature.
    cfg.CLIP_ADAPTERS.CACHE_AGGREGATION = "pool"
    # When using a feature pool, train on approximately one original train set
    # per epoch instead of consuming all augmented views every epoch.
    cfg.CLIP_ADAPTERS.CACHE_POOL_EPOCH_SUBSAMPLE = True
    cfg.CLIP_ADAPTERS.CACHE_FEATURE_ONLY_KEY = True
    # Online mode:
    # Build transient support features/statistics for CLAP constraint and TipA bank.
    # This does not write disk cache.
    cfg.CLIP_ADAPTERS.ONLINE_PREFIT_REPS = 1
    cfg.CLIP_ADAPTERS.ONLINE_PREFIT_TRAIN_AUG = True

    # CLAP-aligned adapter behavior
    cfg.CLIP_ADAPTERS.GRID_SEARCH = False
    cfg.CLIP_ADAPTERS.GRID_SEARCH_SPLIT = "val"

    # CLAP TipA semantics:
    # - plain TipA also uses trainable cache keys
    # - plain TipA trains for one epoch
    # - cache affinity uses raw query features, as in jusiro/CLAP
    cfg.CLIP_ADAPTERS.CLAP_TIPA_TRAINABLE_CACHE = True
    cfg.CLIP_ADAPTERS.CLAP_TIPA_RAW_AFFINITY = True
    cfg.CLIP_ADAPTERS.CLAP_TIPA_ONE_EPOCH = True
    cfg.CLIP_ADAPTERS.TIPA_F_GRID_EPOCHS = 20

    # CrossModal semantics:
    # - concatenate text prompt features into training feature pool
    # - resample text features to match number of image features
    # - each epoch samples half of the expanded pool, matching CLAP's trainer
    cfg.CLIP_ADAPTERS.CROSS_MODAL_RESAMPLE_TEXT = True
    cfg.CLIP_ADAPTERS.CROSS_MODAL_EPOCH_SUBSAMPLE = True



    # train / eval MC sampling
    cfg.CLIP_ADAPTERS.N_SAMPLES = 3
    cfg.CLIP_ADAPTERS.N_TEST_SAMPLES = 10

    # generic KL weight (kept for compatibility)
    cfg.CLIP_ADAPTERS.KL_WEIGHT = 1e-4

    # BayesAdapter-specific defaults
    cfg.CLIP_ADAPTERS.BAYES_PRIOR_STD = 0.01
    cfg.CLIP_ADAPTERS.BAYES_KL_SCALE = 1.0


    # DREAM-BayesAdapter-specific defaults
    # BayesAdapter + text-anchored tangent density-ratio evidence head.
    cfg.CLIP_ADAPTERS.DREAM_ENABLED = True

    # Tangent density geometry
    cfg.CLIP_ADAPTERS.DREAM_RANK = 32
    cfg.CLIP_ADAPTERS.DREAM_CHUNK_CLASSES = 64
    cfg.CLIP_ADAPTERS.DREAM_EPS = 1.0e-4

    # Text/support shrinkage priors
    cfg.CLIP_ADAPTERS.DREAM_TEXT_PRIOR_STRENGTH = 4.0
    cfg.CLIP_ADAPTERS.DREAM_MEAN_PRIOR_STRENGTH = 4.0
    cfg.CLIP_ADAPTERS.DREAM_COV_PRIOR_STRENGTH = 16.0

    # Density-ratio product-of-experts
    # If DREAM_LAMBDA < 0, the adapter chooses lambda from DREAM_LAMBDA_GRID.
    # Include 0.0 so the method can fall back to pure BayesAdapter.
    cfg.CLIP_ADAPTERS.DREAM_LAMBDA = -1.0
    cfg.CLIP_ADAPTERS.DREAM_LAMBDA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]
    cfg.CLIP_ADAPTERS.DREAM_LAMBDA_BETA = 0.01

    # Penalize tangent-space orthogonal residuals for OOD robustness.
    # Set 0.0 to disable.
    cfg.CLIP_ADAPTERS.DREAM_ORTHOGONAL_GAMMA = 0.05

    # Keep BayesAdapter training unchanged by default.
    cfg.CLIP_ADAPTERS.DREAM_DENSITY_ON_TRAIN = False

    # Evidence gate
    cfg.CLIP_ADAPTERS.DREAM_APPLY_GATE = True
    cfg.CLIP_ADAPTERS.DREAM_GATE_REQUIRES_POSITIVE_LAMBDA = True
    cfg.CLIP_ADAPTERS.DREAM_GATE_ON_TRAIN = False
    cfg.CLIP_ADAPTERS.DREAM_GATE_A = 5.0
    cfg.CLIP_ADAPTERS.DREAM_GATE_QUANTILE = 0.05

    # Logging
    cfg.CLIP_ADAPTERS.DREAM_DEBUG = True



    # CAPEL-specific defaults
    cfg.CLIP_ADAPTERS.CAPEL_PROMPT_BANK = "/root/autodl-tmp/MMRL/prompts/capel_prompt_bank_all.json"
    cfg.CLIP_ADAPTERS.CAPEL_PROMPTS_PER_CLASS = 50
    cfg.CLIP_ADAPTERS.CAPEL_PC_LAMBDA = 3.0
    cfg.CLIP_ADAPTERS.CAPEL_STRICT_PROMPT_BANK = True
    cfg.CLIP_ADAPTERS.CAPEL_FALLBACK_ORDER = False

    # CAPEL prompt feature cache.
    # This avoids re-encoding C x K prompts with CLIP text encoder every run.
    cfg.CLIP_ADAPTERS.CAPEL_USE_FEATURE_CACHE = True
    cfg.CLIP_ADAPTERS.CAPEL_REBUILD_FEATURE_CACHE = False
    cfg.CLIP_ADAPTERS.CAPEL_FEATURE_CACHE_DIR = "/root/autodl-tmp/MMRL/prompts/capel_feature_cache"
    cfg.CLIP_ADAPTERS.VNC_CAPEL_VNC_LAMBDA = 0.2


    # ECKA-specific defaults
    cfg.CLIP_ADAPTERS.ECKA_KAPPA0 = 2.0
    cfg.CLIP_ADAPTERS.ECKA_COV_SHRINK = 0.95
    cfg.CLIP_ADAPTERS.ECKA_KERNEL_LAMBDA = -1.0
    cfg.CLIP_ADAPTERS.ECKA_KERNEL_BETA_SCALE = 1.0

    cfg.CLIP_ADAPTERS.ECKA_USE_GDA = True
    cfg.CLIP_ADAPTERS.ECKA_USE_KERNEL = True
    cfg.CLIP_ADAPTERS.ECKA_USE_FUSION_GRID = True

    cfg.CLIP_ADAPTERS.ECKA_W0 = 0.3333333333
    cfg.CLIP_ADAPTERS.ECKA_WG = 0.3333333333
    cfg.CLIP_ADAPTERS.ECKA_WK = 0.3333333333
    cfg.CLIP_ADAPTERS.ECKA_TEMPERATURE = 1.0

    # Conservative residual mode.
    # Default False means final logits = zero-shot logits + residual,
    # not full ECKA replacement.
    cfg.CLIP_ADAPTERS.ECKA_REPLACE_ZS = False
    cfg.CLIP_ADAPTERS.ECKA_RESIDUAL_ALPHA = 0.1
    cfg.CLIP_ADAPTERS.ECKA_MIN_W0 = 0.6

    # Calibration switches. Keep disabled until accuracy sanity checks pass.
    cfg.CLIP_ADAPTERS.ECKA_CALIBRATE = False
    cfg.CLIP_ADAPTERS.ECKA_RANGE_DELTA = -1.0
    cfg.CLIP_ADAPTERS.ECKA_UNCERTAINTY_BETA = 0.0


    # PP-ProKeR-OneHot defaults
    #
    # Official ProKeR-compatible mean:
    #   alpha = solve((1 / lambda) * K + I, one_hot - f_clip(S))
    #
    # Posterior predictive:
    #   p(y|x,S) ~= mean_t softmax((m(x) + sqrt(rho * s2(x)) eps_t) / tau)
    cfg.CLIP_ADAPTERS.PP_PROKER_BETA = 1.0
    cfg.CLIP_ADAPTERS.PP_PROKER_LAMBDA = 1.0

    # GP-style variance ridge. If <= 0, the adapter uses PP_PROKER_LAMBDA.
    cfg.CLIP_ADAPTERS.PP_PROKER_GP_DELTA = -1.0

    # Posterior predictive controls.
    cfg.CLIP_ADAPTERS.PP_PROKER_USE_MC = True
    cfg.CLIP_ADAPTERS.PP_PROKER_MC_SAMPLES = 64
    cfg.CLIP_ADAPTERS.PP_PROKER_RHO = 1.0
    cfg.CLIP_ADAPTERS.PP_PROKER_TAU = 1.0

    # True means final output is log posterior predictive probability.
    # This is preferred for CE/NLL-style evaluation.
    cfg.CLIP_ADAPTERS.PP_PROKER_RETURN_LOG_PROBS = True

    cfg.CLIP_ADAPTERS.PP_PROKER_VARIANCE_JITTER = 1.0e-6
    cfg.CLIP_ADAPTERS.PP_PROKER_MEAN_RESIDUAL_SCALE = 1.0



    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.CALIBRATION = CN()
    cfg.CALIBRATION.USE_FULL_VAL = False

    cfg.TASK = "B2N"








    _sync_active_mmrl_family(cfg)
    _as_legacy_mmrl(cfg)
    _as_legacy_mmrl_mix(cfg)
    _as_legacy_mmrlpp(cfg)
    _as_legacy_bayes_mmrl(cfg)
    _as_legacy_vcrm_mmrl(cfg)
    _as_legacy_clipadapter(cfg)
    return cfg


def _merge_if_exists(cfg, path: str | None):
    if path:
        cfg.merge_from_file(path)


def _apply_method_dataset_overrides(cfg):
    opts = resolve_method_dataset_overrides(
        method_name=cfg.METHOD.NAME,
        dataset_name=cfg.DATASET.NAME,
        protocol_name=cfg.PROTOCOL.NAME,
    )
    if opts:
        cfg.merge_from_list(opts)


def finalize_cfg(cfg):
    cfg.TASK = cfg.PROTOCOL.NAME
    _apply_method_dataset_overrides(cfg)

    # Convenience method:
    # METHOD.NAME=CLAP means CLAP's main method:
    # ZS-initialized cosine linear probing + class-adaptive l2 constraint.
    if str(cfg.METHOD.NAME).upper() == "CLAP":
        cfg.METHOD.FAMILY = "adapter"
        cfg.METHOD.EXEC_MODE = "cache"
        cfg.CLIP_ADAPTERS.INIT = "ZS"
        if str(cfg.CLIP_ADAPTERS.CONSTRAINT).lower() == "none":
            cfg.CLIP_ADAPTERS.CONSTRAINT = "l2"
        cfg.CLIP_ADAPTERS.ALLOW_CACHE = True

    _sync_active_mmrl_family(cfg)
    _as_legacy_mmrl(cfg)
    _as_legacy_mmrl_mix(cfg)
    _as_legacy_mmrlpp(cfg)
    _as_legacy_bayes_mmrl(cfg)
    _as_legacy_vcrm_mmrl(cfg)
    _as_legacy_clipadapter(cfg)

    cfg.freeze()
    return cfg


def setup_cfg(args):
    cfg = get_refactor_defaults()

    _merge_if_exists(cfg, args.dataset_config_file)
    _merge_if_exists(cfg, args.runtime_config_file)
    _merge_if_exists(cfg, args.protocol_config_file)
    _merge_if_exists(cfg, args.method_config_file)
    _merge_if_exists(cfg, args.exp_config)

    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed is not None and args.seed >= 0:
        cfg.SEED = args.seed
    if args.method:
        cfg.METHOD.NAME = args.method
    if args.protocol:
        cfg.PROTOCOL.NAME = args.protocol
        cfg.TASK = args.protocol
    if args.exec_mode:
        cfg.METHOD.EXEC_MODE = args.exec_mode
    if args.opts:
        cfg.merge_from_list(args.opts)

    return finalize_cfg(cfg)