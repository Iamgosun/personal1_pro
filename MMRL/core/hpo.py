from __future__ import annotations

import copy
import csv
import itertools
import json
import os
import os.path as osp
import shutil
from typing import Any

import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

from core.config import setup_cfg


def hpo_enabled(cfg) -> bool:
    return hasattr(cfg, "HPO") and bool(getattr(cfg.HPO, "ENABLED", False))


def _is_cfg_node_like(value: Any) -> bool:
    return hasattr(value, "items")


def _normalise_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalise_json_value(v) for v in value]
    return str(value)


def _value_to_opt_string(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _read_hpo_grid(cfg) -> dict[str, list[Any]]:
    if not hasattr(cfg.HPO, "GRID") or len(cfg.HPO.GRID) == 0:
        raise RuntimeError(
            "HPO.ENABLED=True but HPO.GRID is empty. "
            "Declare the search space in the method yaml."
        )

    grid: dict[str, list[Any]] = {}

    for key, spec in cfg.HPO.GRID.items():
        # Preferred schema:
        #   r_prior_std:
        #     PATH: BAYESRT_MMRL.R_PRIOR_STD
        #     VALUES: [...]
        if _is_cfg_node_like(spec) and hasattr(spec, "PATH") and hasattr(spec, "VALUES"):
            path = str(spec.PATH)
            values = list(spec.VALUES)
        else:
            # Shorthand schema:
            #   BAYESRT_MMRL.R_PRIOR_STD: [...]
            path = str(key)
            values = list(spec)

        if not path:
            raise RuntimeError(f"Invalid HPO grid item {key}: empty PATH")

        if len(values) == 0:
            raise RuntimeError(f"Invalid HPO grid item {key}: VALUES is empty")

        grid[path] = values

    return grid


def _iter_grid(grid: dict[str, list[Any]]):
    paths = list(grid.keys())
    values = [grid[p] for p in paths]

    for combo in itertools.product(*values):
        yield dict(zip(paths, combo))


def _flatten_params_to_opts(params: dict[str, Any]) -> list[str]:
    opts: list[str] = []
    for key, value in params.items():
        opts.extend([str(key), _value_to_opt_string(value)])
    return opts


def _safe_tag_piece(value: Any) -> str:
    text = str(value)
    for ch in ["/", "\\", ":", " ", "\t", "\n", ",", "[", "]", "(", ")"]:
        text = text.replace(ch, "_")
    return text


def _make_candidate_tag(index: int, params: dict[str, Any]) -> str:
    pieces = [f"hpo{index:03d}"]

    for path, value in params.items():
        name = path.split(".")[-1].lower()
        pieces.append(f"{name}-{_safe_tag_piece(value)}")

    return "__".join(pieces)


def _make_candidate_args(base_args, base_output_dir: str, index: int, tag: str, params: dict[str, Any]):
    cand_args = copy.deepcopy(base_args)

    base_opts = list(base_args.opts or [])
    hpo_opts = _flatten_params_to_opts(params)

    # Critical: avoid recursive HPO when candidate cfg is rebuilt.
    hpo_opts.extend(["HPO.ENABLED", "False"])

    # Make candidate configs/checkpoints distinguishable.
    hpo_opts.extend(["METHOD.TAG", tag])

    cand_args.opts = base_opts + hpo_opts
    cand_args.output_dir = osp.join(
        base_output_dir,
        "hpo_candidates",
        f"candidate_{index:03d}_{tag}",
    )

    return cand_args


def _score_candidate(trainer, split: str, require_val: bool) -> float:
    if split == "val" and getattr(trainer, "val_loader", None) is None:
        if require_val:
            raise RuntimeError(
                "HPO.SPLIT='val' but trainer.val_loader is None. "
                "Create a validation split first. Do not fall back to test for HPO."
            )
        split = "test"

    return float(trainer.test(split=split))


def _write_hpo_summary(base_output_dir: str, rows: list[dict[str, Any]], best_row: dict[str, Any] | None):
    os.makedirs(base_output_dir, exist_ok=True)

    json_path = osp.join(base_output_dir, "hpo_summary.json")
    csv_path = osp.join(base_output_dir, "hpo_summary.csv")
    best_path = osp.join(base_output_dir, "hpo_best_opts.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": best_row,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if best_row is not None:
        best_opts = []
        for key, value in best_row["params"].items():
            best_opts.extend([key, _value_to_opt_string(value)])

        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "opts": best_opts,
                    "best": best_row,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"[HPO] saved summary to {json_path}")
    print(f"[HPO] saved summary csv to {csv_path}")
    if best_row is not None:
        print(f"[HPO] saved best opts to {best_path}")


def _copy_best_model_if_requested(base_cfg, best_row: dict[str, Any] | None):
    if best_row is None:
        return

    if bool(getattr(base_cfg.HPO, "TRAIN_FINAL_WITH_BEST", False)):
        return

    if not bool(getattr(base_cfg.HPO, "COPY_BEST_MODEL", True)):
        return

    src = osp.join(best_row["output_dir"], "refactor_model")
    dst = osp.join(base_cfg.OUTPUT_DIR, "refactor_model")

    if not osp.isdir(src):
        print(f"[HPO] best model directory not found, skip copy: {src}")
        return

    os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[HPO] copied best model directory: {src} -> {dst}")


def _train_final_with_best_if_requested(base_args, base_cfg, best_row: dict[str, Any] | None):
    if best_row is None:
        return None

    if not bool(getattr(base_cfg.HPO, "TRAIN_FINAL_WITH_BEST", False)):
        return None

    final_args = copy.deepcopy(base_args)
    final_args.output_dir = base_cfg.OUTPUT_DIR

    base_opts = list(base_args.opts or [])
    final_opts = _flatten_params_to_opts(best_row["params"])
    final_opts.extend(["HPO.ENABLED", "False"])

    final_args.opts = base_opts + final_opts

    final_cfg = setup_cfg(final_args)

    if final_cfg.SEED >= 0:
        print(f"[HPO] Setting fixed seed for final best run: {final_cfg.SEED}")
        set_random_seed(final_cfg.SEED)

    setup_logger(final_cfg.OUTPUT_DIR)

    print("[HPO] Final training with best hyperparameters")
    print(f"[HPO] best params = {best_row['params']}")

    trainer = build_trainer(final_cfg)
    trainer.train()
    return trainer


def run_hpo(base_args, base_cfg):
    if getattr(base_args, "eval_only", False):
        raise RuntimeError("HPO is not supported with --eval-only.")

    if getattr(base_args, "no_train", False):
        raise RuntimeError("HPO is not supported with --no-train.")

    grid = _read_hpo_grid(base_cfg)

    split = str(getattr(base_cfg.HPO, "SPLIT", "val"))
    metric = str(getattr(base_cfg.HPO, "METRIC", "accuracy"))
    require_val = bool(getattr(base_cfg.HPO, "REQUIRE_VAL", True))

    base_output_dir = base_cfg.OUTPUT_DIR
    rows: list[dict[str, Any]] = []

    best_score = -float("inf")
    best_row: dict[str, Any] | None = None

    print("[HPO] enabled")
    print(f"[HPO] split={split}")
    print(f"[HPO] metric={metric}")
    print(f"[HPO] grid={grid}")

    for index, params in enumerate(_iter_grid(grid), start=1):
        tag = _make_candidate_tag(index, params)
        cand_args = _make_candidate_args(base_args, base_output_dir, index, tag, params)
        cand_cfg = setup_cfg(cand_args)

        print("=" * 80)
        print(f"[HPO] candidate {index}")
        print(f"[HPO] tag={tag}")
        print(f"[HPO] params={params}")
        print(f"[HPO] output_dir={cand_cfg.OUTPUT_DIR}")

        if cand_cfg.SEED >= 0:
            print(f"[HPO] Setting fixed seed: {cand_cfg.SEED}")
            set_random_seed(cand_cfg.SEED)

        setup_logger(cand_cfg.OUTPUT_DIR)

        trainer = build_trainer(cand_cfg)
        trainer.train()

        score = _score_candidate(trainer, split=split, require_val=require_val)

        row = {
            "candidate": index,
            "tag": tag,
            "score": score,
            "metric": metric,
            "split": split,
            "output_dir": cand_cfg.OUTPUT_DIR,
            "params": {k: _normalise_json_value(v) for k, v in params.items()},
        }

        for key, value in params.items():
            row[key] = _normalise_json_value(value)

        rows.append(row)

        print(f"[HPO] candidate {index} {metric}={score:.6f}")

        if score > best_score:
            best_score = score
            best_row = copy.deepcopy(row)
            print(f"[HPO] new best: candidate={index}, score={score:.6f}")

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("=" * 80)
    print(f"[HPO] best={best_row}")

    if bool(getattr(base_cfg.HPO, "SAVE_SUMMARY", True)):
        _write_hpo_summary(base_output_dir, rows, best_row)

    _copy_best_model_if_requested(base_cfg, best_row)
    final_trainer = _train_final_with_best_if_requested(base_args, base_cfg, best_row)

    return {
        "best": best_row,
        "rows": rows,
        "final_trainer": final_trainer,
    }