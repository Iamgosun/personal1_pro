from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

from core.config import setup_cfg
from core.hpo import (
    _evaluate_candidate_for_hpo,
    _select_best_row,
    _write_hpo_summary,
    _copy_best_model_if_requested,
    _test_best_model_if_requested,
    _normalise_json_value,
)
from run import _import_runtime_modules


CANDIDATES = [
    (1, "0.001", 0.001),
    (2, "0.01", 0.01),
    (3, "0.1", 0.1),
]


def read_header(run_log: Path, key: str) -> str | None:
    if not run_log.exists():
        return None
    prefix = key + ":"
    for line in run_log.read_text(errors="ignore").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def make_args(
    *,
    root: str,
    output_dir: Path,
    dataset: str,
    shot: int,
    seed: int,
    exec_mode: str,
    backbone: str,
    opts_extra: list[str] | None = None,
):
    opts = [
        "DATASET.NUM_SHOTS", str(shot),
        "DATASET.SUBSAMPLE_CLASSES", "all",
        "MODEL.BACKBONE.NAME", backbone,
    ]
    if opts_extra:
        opts.extend(opts_extra)

    return SimpleNamespace(
        root=root,
        output_dir=str(output_dir),
        dataset_config_file=f"configs/datasets/{dataset}.yaml",
        method_config_file="configs/methods/clip_adapters_bayes.yaml",
        protocol_config_file="configs/protocols/fs.yaml",
        runtime_config_file="configs/runtime/adapter_family.yaml",
        exp_config="",
        method="ClipAdapters",
        protocol="FS",
        exec_mode=exec_mode,
        seed=seed,
        trainer="RefactorRunner",
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=opts,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="output_refactor/ClipAdapters/BAYES_ADAPTER/FS/fewshot_train/sun397/shots_32/ViT-B-16")
    ap.add_argument("--dataset", default="sun397")
    ap.add_argument("--shot", type=int, default=32)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--root", default=None)
    ap.add_argument("--exec-mode", default=None)
    ap.add_argument("--backbone", default=None)
    args = ap.parse_args()

    _import_runtime_modules()

    base = Path(args.base)

    for seed in args.seeds:
        seed_dir = base / f"seed{seed}"
        run_log = seed_dir / "run.log"

        root = args.root or read_header(run_log, "DATA_ROOT")
        exec_mode = args.exec_mode or read_header(run_log, "EXEC_MODE")
        backbone = args.backbone or read_header(run_log, "BACKBONE")

        if not root or not exec_mode or not backbone:
            raise RuntimeError(
                f"seed{seed}: cannot infer DATA_ROOT/EXEC_MODE/BACKBONE from {run_log}. "
                "Pass --root, --exec-mode, --backbone explicitly."
            )

        base_args = make_args(
            root=root,
            output_dir=seed_dir,
            dataset=args.dataset,
            shot=args.shot,
            seed=seed,
            exec_mode=exec_mode,
            backbone=backbone,
        )
        base_cfg = setup_cfg(base_args)

        rows = []
        for index, value_str, value in CANDIDATES:
            tag = f"hpo{index:03d}__bayes_prior_std-{value_str}"
            cand_dir = seed_dir / "hpo_candidates" / f"candidate_{index:03d}_{tag}"
            model_dir = cand_dir / "refactor_model"

            if not model_dir.is_dir() or not list(model_dir.glob("model.pth.tar-*")):
                raise RuntimeError(f"missing checkpoint for seed{seed} candidate {index}: {model_dir}")

            cand_args = make_args(
                root=root,
                output_dir=cand_dir,
                dataset=args.dataset,
                shot=args.shot,
                seed=seed,
                exec_mode=exec_mode,
                backbone=backbone,
                opts_extra=[
                    "CLIP_ADAPTERS.BAYES_PRIOR_STD", value_str,
                    "HPO.ENABLED", "False",
                    "TEST.NO_TEST", "True",
                    "METHOD.TAG", tag,
                ],
            )
            cand_cfg = setup_cfg(cand_args)

            if cand_cfg.SEED >= 0:
                set_random_seed(cand_cfg.SEED)

            setup_logger(cand_cfg.OUTPUT_DIR)

            trainer = build_trainer(cand_cfg)
            trainer.load_model(str(cand_dir))

            metrics = _evaluate_candidate_for_hpo(
                trainer=trainer,
                split="val",
                require_val=True,
                n_bins=10,
            )

            row = {
                "candidate": index,
                "tag": tag,
                "split": metrics.get("_actual_split", "val"),
                "output_dir": str(cand_dir),
                "params": {"CLIP_ADAPTERS.BAYES_PRIOR_STD": value},
                "accuracy": float(metrics["accuracy"]),
                "ece": float(metrics["ece"]),
                "aece": float(metrics["aece"]),
                "CLIP_ADAPTERS.BAYES_PRIOR_STD": _normalise_json_value(value),
            }
            rows.append(row)

            print(
                f"[RECOVER] seed={seed} candidate={index} "
                f"ACC={row['accuracy']:.4f} ECE={row['ece']:.4f} AECE={row['aece']:.4f}"
            )

            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best = _select_best_row(rows, base_cfg)
        print(f"[RECOVER] seed={seed} best={best}")

        _write_hpo_summary(str(seed_dir), rows, best)
        _copy_best_model_if_requested(base_cfg, best)
        _test_best_model_if_requested(base_args, base_cfg, best)

    print("[RECOVER] done")


if __name__ == "__main__":
    main()
