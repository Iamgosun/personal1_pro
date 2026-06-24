from __future__ import annotations

import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "12")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "4")

import torch

torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))

import argparse
import importlib
import tempfile
from pathlib import Path

import yaml

from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger

from core.config import setup_cfg
from core.hpo import hpo_enabled, run_hpo
from core.utils import import_optional_modules
from datasets.fair_val_protocol import install_kplusval_datasetbase_patch


_KPLUSVAL_KEYS = {
    "MERGE_VAL_TO_TRAIN",
    "DROP_VAL_AFTER_MERGE",
}


def _truthy(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _prepare_method_config_for_kplusval(args):
    """
    Read K+Val protocol flags directly from the method YAML.

    Why this function exists:
      - The user wants the protocol to be specified in method configs, e.g.
          DATASET:
            MERGE_VAL_TO_TRAIN: true
            DROP_VAL_AFTER_MERGE: true
      - YACS rejects unknown DATASET keys unless core/config.py declares them.
      - To avoid requiring a core/config.py patch, we consume these two protocol
        fields here, export them to environment variables for runtime code, and
        pass a sanitized temporary YAML to setup_cfg().

    Therefore run_plan.sh does not decide the protocol. The method config does.
    """
    method_config_file = getattr(args, "method_config_file", "")
    if not method_config_file:
        os.environ.pop("MMRL_MERGE_VAL_TO_TRAIN", None)
        os.environ.pop("MMRL_DROP_VAL_AFTER_MERGE", None)
        return None

    path = Path(method_config_file)
    if not path.exists():
        os.environ.pop("MMRL_MERGE_VAL_TO_TRAIN", None)
        os.environ.pop("MMRL_DROP_VAL_AFTER_MERGE", None)
        return None

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    dataset_cfg = cfg.get("DATASET", None)
    if not isinstance(dataset_cfg, dict):
        os.environ["MMRL_MERGE_VAL_TO_TRAIN"] = "0"
        os.environ["MMRL_DROP_VAL_AFTER_MERGE"] = "1"
        return None

    merge_val = _truthy(dataset_cfg.get("MERGE_VAL_TO_TRAIN", False), False)
    drop_val = _truthy(dataset_cfg.get("DROP_VAL_AFTER_MERGE", True), True)

    os.environ["MMRL_MERGE_VAL_TO_TRAIN"] = "1" if merge_val else "0"
    os.environ["MMRL_DROP_VAL_AFTER_MERGE"] = "1" if drop_val else "0"

    # Remove only protocol-private keys before YACS merge.
    changed = False
    for key in list(_KPLUSVAL_KEYS):
        if key in dataset_cfg:
            dataset_cfg.pop(key)
            changed = True

    if not changed:
        return None

    if len(dataset_cfg) == 0:
        cfg.pop("DATASET", None)

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="method_config_sanitized_",
        delete=False,
        encoding="utf-8",
    )
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)

    args.method_config_file = tmp.name

    print(
        "[KPlusValProtocol] method config flags: "
        f"MERGE_VAL_TO_TRAIN={merge_val}, DROP_VAL_AFTER_MERGE={drop_val}; "
        f"sanitized_method_config={tmp.name}"
    )

    return tmp.name


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    for key, val in sorted(vars(args).items()):
        print(f"{key}: {val}")
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def _import_runtime_modules():
    # Datasets are optional because users may not have every dataset module/file ready.
    import_optional_modules([
        "datasets.oxford_pets",
        "datasets.oxford_flowers",
        "datasets.fgvc_aircraft",
        "datasets.dtd",
        "datasets.eurosat",
        "datasets.stanford_cars",
        "datasets.food101",
        "datasets.sun397",
        "datasets.caltech101",
        "datasets.ucf101",
        "datasets.imagenet",
        "datasets.imagenetv2",
        "datasets.imagenet_sketch",
        "datasets.imagenet_a",
        "datasets.imagenet_r",
        "datasets.cifar_10",

        # Retina/fundus datasets.
        "datasets.retina_common",
        "datasets.fives",
        "datasets.fundus1000x39",
        "datasets.odir5k",
        "datasets.deepdrid",
        "datasets.deepdird",
        "datasets.mesidor",
        "datasets.mmac",
    ])

    # Trainer registration is NOT optional. Fail loudly so registry errors are debuggable.
    importlib.import_module("trainers.refactor_runner")


def main(args):
    _import_runtime_modules()

    sanitized_cfg_path = _prepare_method_config_for_kplusval(args)

    cfg = setup_cfg(args)

    # Must be installed before build_trainer(cfg), because trainer construction
    # creates the dataset and data loaders.
    install_kplusval_datasetbase_patch(cfg)

    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if hpo_enabled(cfg):
        run_hpo(args, cfg)
        return

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

    if sanitized_cfg_path:
        try:
            Path(sanitized_cfg_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dataset-config-file", type=str, default="")
    parser.add_argument("--method-config-file", type=str, default="")
    parser.add_argument("--protocol-config-file", type=str, default="")
    parser.add_argument("--runtime-config-file", type=str, default="")
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--protocol", type=str, default="")
    parser.add_argument("--exec-mode", type=str, default="")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--trainer", type=str, default="RefactorRunner")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    main(parser.parse_args())
