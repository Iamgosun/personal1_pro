from __future__ import annotations

from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default

from core.overrides import resolve_method_dataset_overrides


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


def _as_legacy_mmrl(cfg):
    if not hasattr(cfg.TRAINER, "MMRL"):
        cfg.TRAINER.MMRL = CN()
    sec = cfg.TRAINER.MMRL
    src = cfg.MMRL
    sec.PREC = src.PREC
    sec.ALPHA = src.ALPHA
    sec.REG_WEIGHT = src.REG_WEIGHT
    sec.N_REP_TOKENS = src.N_REP_TOKENS
    sec.REP_LAYERS = src.REP_LAYERS
    sec.REP_DIM = src.REP_DIM


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
    sec.N_MC_TRAIN = src.N_MC_TRAIN
    sec.N_MC_TEST = src.N_MC_TEST
    sec.KL_WEIGHT = src.KL_WEIGHT
    sec.PRIOR_STD = src.PRIOR_STD
    sec.POSTERIOR_RHO_INIT = src.POSTERIOR_RHO_INIT
    sec.EVAL_USE_POSTERIOR_MEAN = src.EVAL_USE_POSTERIOR_MEAN


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
    cfg.BAYES_MMRL.N_MC_TRAIN = 3
    cfg.BAYES_MMRL.N_MC_TEST = 5
    cfg.BAYES_MMRL.KL_WEIGHT = 1e-4
    cfg.BAYES_MMRL.PRIOR_STD = 0.02
    # softplus(-3.9) ~= 0.02
    cfg.BAYES_MMRL.POSTERIOR_RHO_INIT = -3.9
    # False: use MC averaging at test time by default
    # True + N_MC_TEST=1: use posterior mean and enable eval cache
    cfg.BAYES_MMRL.EVAL_USE_POSTERIOR_MEAN = False

    cfg.CLIP_ADAPTERS = CN()
    cfg.CLIP_ADAPTERS.PREC = "fp32"
    cfg.CLIP_ADAPTERS.TYPE = "MP"
    cfg.CLIP_ADAPTERS.INIT = "ZS"
    cfg.CLIP_ADAPTERS.CONSTRAINT = "none"
    cfg.CLIP_ADAPTERS.ENHANCED_BASE = "none"
    cfg.CLIP_ADAPTERS.ALLOW_CACHE = True
    cfg.CLIP_ADAPTERS.N_SAMPLES = 3
    cfg.CLIP_ADAPTERS.KL_WEIGHT = 1e-4

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TASK = "B2N"

    _as_legacy_mmrl(cfg)
    _as_legacy_mmrlpp(cfg)
    _as_legacy_bayes_mmrl(cfg)
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
    _as_legacy_mmrl(cfg)
    _as_legacy_mmrlpp(cfg)
    _as_legacy_bayes_mmrl(cfg)
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