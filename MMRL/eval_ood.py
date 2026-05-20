from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler

from dassl.data.datasets import DATASET_REGISTRY
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

from core.config import setup_cfg
from core.utils import import_optional_modules
from data.build import build_split_dataset
from evaluation.ood_detection import compute_ood_metrics, msp_from_logits


def _import_runtime_modules():
    import_optional_modules([
        "datasets.cifar10",
        "datasets.ood_image_datasets",
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
    ])

    importlib.import_module("trainers.refactor_runner")


def make_ood_cfg(base_cfg, ood_dataset_key: str):
    cfg = base_cfg.clone()
    cfg.defrost()

    dataset_cfg = Path("configs") / "datasets" / f"{ood_dataset_key}.yaml"
    if not dataset_cfg.exists():
        raise FileNotFoundError(
            f"OOD dataset config not found: {dataset_cfg}. "
            f"Create configs/datasets/{ood_dataset_key}.yaml first."
        )

    cfg.merge_from_file(str(dataset_cfg))

    # OOD evaluation only uses test split. Do not trigger few-shot train/val.
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.freeze()
    return cfg


def build_ood_data_source(base_cfg, ood_dataset_key: str):
    ood_cfg = make_ood_cfg(base_cfg, ood_dataset_key)
    registry_name = ood_cfg.DATASET.NAME

    dataset_cls = DATASET_REGISTRY.get(registry_name)
    dataset = dataset_cls(ood_cfg)

    if not hasattr(dataset, "test") or dataset.test is None:
        raise RuntimeError(
            f"OOD dataset {ood_dataset_key} "
            f"(registry={registry_name}) has no test split."
        )

    if len(dataset.test) == 0:
        raise RuntimeError(
            f"OOD dataset {ood_dataset_key} "
            f"(registry={registry_name}) has empty test split."
        )

    print(
        f"[OOD] {ood_dataset_key}: registry={registry_name}, "
        f"num_test={len(dataset.test)}"
    )

    return registry_name, dataset.test


def build_ood_loader(
    cfg,
    ood_dataset_key: str,
    batch_size: int,
    num_workers: int,
):
    registry_name, data_source = build_ood_data_source(
        base_cfg=cfg,
        ood_dataset_key=ood_dataset_key,
    )

    # Use the ID cfg transform/DatasetWrapper so OOD images go through the same
    # preprocessing path as CIFAR-10 test images.
    dataset = build_split_dataset(
        cfg,
        data_source=data_source,
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return registry_name, loader


@torch.no_grad()
def collect_logits(trainer, loader, split_name: str):
    eval_ctx = trainer.executor.build_eval_context(trainer, split_name)

    logits, labels = trainer.executor._collect_logits_and_labels(
        trainer=trainer,
        data_loader=loader,
        eval_ctx=eval_ctx,
        process_evaluator=False,
        collect_fusion_variants=False,
        keep_on_device=False,
    )

    return logits, labels


def mean_metrics(rows):
    keys = ["TNR95", "AUROC", "DetAcc", "AUPR_In", "AUPR_Out"]

    return {
        key: sum(row[key] for row in rows) / max(len(rows), 1)
        for key in keys
    }


def save_outputs(output_dir: Path, rows: List[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ood_results.json"
    csv_path = output_dir / "ood_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "dataset",
        "registry_dataset",
        "num_id",
        "num_ood",
        "TNR95",
        "AUROC",
        "DetAcc",
        "AUPR_In",
        "AUPR_Out",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved OOD JSON to {json_path}")
    print(f"Saved OOD CSV to {csv_path}")


def main(args):
    _import_runtime_modules()

    if not args.ood_dataset:
        raise ValueError(
            "No OOD datasets specified. "
            "Use --ood-dataset dtd --ood-dataset tinyimagenet ..."
        )

    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.set_model_mode("eval")

    print("Collecting ID logits from ID test loader...")
    id_logits, _ = collect_logits(
        trainer=trainer,
        loader=trainer.test_loader,
        split_name="test",
    )

    id_scores = msp_from_logits(id_logits)

    rows = []

    for ood_dataset_key in args.ood_dataset:
        print(f"Evaluating OOD dataset: {ood_dataset_key}")

        registry_name, ood_loader = build_ood_loader(
            cfg=cfg,
            ood_dataset_key=ood_dataset_key,
            batch_size=args.ood_batch_size,
            num_workers=args.ood_num_workers,
        )

        ood_logits, _ = collect_logits(
            trainer=trainer,
            loader=ood_loader,
            split_name=f"ood_{ood_dataset_key}",
        )

        ood_scores = msp_from_logits(ood_logits)
        metrics = compute_ood_metrics(id_scores, ood_scores)

        row = {
            "dataset": ood_dataset_key,
            "registry_dataset": registry_name,
            "num_id": int(id_scores.numel()),
            "num_ood": int(ood_scores.numel()),
            **metrics,
        }
        rows.append(row)

        print(
            f"{ood_dataset_key}: "
            f"TNR95={row['TNR95']:.3f}, "
            f"AUROC={row['AUROC']:.3f}, "
            f"DetAcc={row['DetAcc']:.3f}, "
            f"AUPR_In={row['AUPR_In']:.3f}, "
            f"AUPR_Out={row['AUPR_Out']:.3f}"
        )

    avg = {
        "dataset": "average",
        "registry_dataset": "average",
        "num_id": int(id_scores.numel()),
        "num_ood": sum(row["num_ood"] for row in rows),
        **mean_metrics(rows),
    }
    rows.append(avg)

    print(
        f"average: "
        f"TNR95={avg['TNR95']:.3f}, "
        f"AUROC={avg['AUROC']:.3f}, "
        f"DetAcc={avg['DetAcc']:.3f}, "
        f"AUPR_In={avg['AUPR_In']:.3f}, "
        f"AUPR_Out={avg['AUPR_Out']:.3f}"
    )

    save_outputs(Path(args.ood_output_dir), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Same surface as run.py
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
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--load-epoch", type=int, default=None)
    parser.add_argument("--no-train", action="store_true")

    # OOD args
    parser.add_argument("--ood-dataset", action="append", default=[])
    parser.add_argument("--ood-output-dir", type=str, required=True)
    parser.add_argument("--ood-batch-size", type=int, default=250)
    parser.add_argument("--ood-num-workers", type=int, default=4)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    main(parser.parse_args())
