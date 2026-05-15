from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import os
import os.path as osp
import shutil
from typing import Any

import torch
import torch.nn.functional as F
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


def _make_candidate_args(
    base_args,
    base_output_dir: str,
    index: int,
    tag: str,
    params: dict[str, Any],
):
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


def _set_eval_seed(seed: int):
    if seed is None or int(seed) < 0:
        return

    eval_seed = int(seed) + 100000
    torch.manual_seed(eval_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed)


def _hpo_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.detach().cpu()
    labels = labels.detach().cpu().long()

    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def _hpo_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    correct = (preds == labels).float()

    n_bins = max(1, int(n_bins))
    edges = torch.linspace(0.0, 1.0, steps=n_bins + 1)
    total = max(int(labels.numel()), 1)

    ece = 0.0

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == 0:
            mask = (confs >= left) & (confs <= right)
        else:
            mask = (confs > left) & (confs <= right)

        count = int(mask.sum().item())
        if count == 0:
            continue

        bin_acc = float(correct[mask].mean().item() * 100.0)
        bin_conf = float(confs[mask].mean().item() * 100.0)
        ece += abs(bin_acc - bin_conf) * (count / total)

    return float(ece)


def _hpo_aece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Adaptive ECE with equal-count bins after sorting by confidence.

    Unit: percentage points, same as ECE.
    """
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()

    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    correct = (preds == labels).float()

    n = int(labels.numel())
    if n == 0:
        return float("nan")

    n_bins = max(1, min(int(n_bins), n))
    order = torch.argsort(confs, descending=False)

    confs = confs[order]
    correct = correct[order]

    chunks = torch.chunk(torch.arange(n), chunks=n_bins)

    aece = 0.0

    for idxs in chunks:
        if idxs.numel() == 0:
            continue

        bin_acc = float(correct[idxs].mean().item() * 100.0)
        bin_conf = float(confs[idxs].mean().item() * 100.0)
        frac = float(idxs.numel() / n)

        aece += abs(bin_acc - bin_conf) * frac

    return float(aece)


def _hpo_basic_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> dict[str, float]:
    return {
        "accuracy": _hpo_accuracy(logits, labels),
        "ece": _hpo_ece(logits, labels, n_bins=n_bins),
        "aece": _hpo_aece(logits, labels, n_bins=n_bins),
    }


def _evaluate_candidate_for_hpo(
    trainer,
    split: str,
    require_val: bool,
    n_bins: int = 10,
) -> dict[str, Any]:
    actual_split = str(split)

    if actual_split == "val":
        if getattr(trainer, "val_loader", None) is None:
            if require_val:
                raise RuntimeError(
                    "HPO.SPLIT='val' but trainer.val_loader is None. "
                    "Create a validation split first. Do not fall back to test for HPO."
                )

            actual_split = "test"
            data_loader = trainer.test_loader
        else:
            data_loader = trainer.val_loader

    elif actual_split == "test":
        data_loader = trainer.test_loader

    else:
        raise RuntimeError(f"Unsupported HPO.SPLIT={split!r}; use 'val' or 'test'.")

    trainer.set_model_mode("eval")

    _set_eval_seed(getattr(trainer.cfg, "SEED", -1))

    eval_ctx = trainer.executor.build_eval_context(trainer, actual_split)

    logits, labels = trainer.executor._collect_logits_and_labels(
        trainer=trainer,
        data_loader=data_loader,
        eval_ctx=eval_ctx,
        process_evaluator=False,
        collect_fusion_variants=False,
    )

    metrics = _hpo_basic_metrics(
        logits=logits,
        labels=labels,
        n_bins=n_bins,
    )
    metrics["_actual_split"] = actual_split

    return metrics


def _as_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _require_finite(row: dict[str, Any], key: str) -> float:
    value = _as_float(row, key)
    if not math.isfinite(value):
        raise RuntimeError(
            f"HPO selector requires finite metric {key!r}, "
            f"but got {row.get(key)!r} for candidate={row.get('candidate')}"
        )
    return value


def _stable_tag(row: dict[str, Any]) -> str:
    return str(row.get("tag", ""))


def _select_by_acc(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    valid = [r for r in rows if math.isfinite(_as_float(r, "accuracy"))]
    if not valid:
        raise RuntimeError("HPO selector 'acc' found no finite accuracy values.")

    selected = sorted(
        valid,
        key=lambda r: (
            -_require_finite(r, "accuracy"),
            _as_float(r, "ece", float("inf")),
            _as_float(r, "aece", float("inf")),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = "max ACC; tie-break lower ECE, lower AECE"
    selected["_selection_score"] = _require_finite(selected, "accuracy")
    return selected


def _select_by_ece(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    valid = [r for r in rows if math.isfinite(_as_float(r, "ece"))]
    if not valid:
        raise RuntimeError("HPO selector 'ece' found no finite ECE values.")

    selected = sorted(
        valid,
        key=lambda r: (
            _require_finite(r, "ece"),
            _as_float(r, "aece", float("inf")),
            -_as_float(r, "accuracy", -float("inf")),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = "min ECE; tie-break lower AECE, higher ACC"
    selected["_selection_score"] = -_require_finite(selected, "ece")
    return selected


def _select_by_aece(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    valid = [r for r in rows if math.isfinite(_as_float(r, "aece"))]
    if not valid:
        raise RuntimeError("HPO selector 'aece' found no finite AECE values.")

    selected = sorted(
        valid,
        key=lambda r: (
            _require_finite(r, "aece"),
            _as_float(r, "ece", float("inf")),
            -_as_float(r, "accuracy", -float("inf")),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = "min AECE; tie-break lower ECE, higher ACC"
    selected["_selection_score"] = -_require_finite(selected, "aece")
    return selected


def _select_by_acc_metric_window(
    rows: list[dict[str, Any]],
    cfg,
    calibration_metric: str,
) -> dict[str, Any]:
    valid = [r for r in rows if math.isfinite(_as_float(r, "accuracy"))]
    if not valid:
        raise RuntimeError(
            f"HPO selector 'acc_{calibration_metric}_window' "
            "found no finite accuracy values."
        )

    best_acc = max(_require_finite(r, "accuracy") for r in valid)
    tolerance = float(getattr(cfg.HPO, "ACC_TOLERANCE", 0.0))
    threshold = best_acc - tolerance

    eligible = [
        r for r in valid
        if _require_finite(r, "accuracy") >= threshold
    ]

    if not eligible:
        raise RuntimeError("HPO selector found no eligible candidates.")

    selected = sorted(
        eligible,
        key=lambda r: (
            _as_float(r, calibration_metric, float("inf")),
            -_require_finite(r, "accuracy"),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = (
        f"ACC >= best_ACC - {tolerance}, then min {calibration_metric.upper()}"
    )
    selected["_selection_score"] = (
        _require_finite(selected, "accuracy")
        - 1.0e-3 * _as_float(selected, calibration_metric, 0.0)
    )
    selected["_best_accuracy"] = best_acc
    selected["_accuracy_threshold"] = threshold
    selected["_num_eligible"] = len(eligible)
    selected["_num_finished"] = len(valid)

    return selected


def _select_by_acc_ece_window(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    return _select_by_acc_metric_window(rows, cfg, calibration_metric="ece")


def _select_by_acc_aece_window(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    return _select_by_acc_metric_window(rows, cfg, calibration_metric="aece")


def _select_by_weighted_acc_ece(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    acc_weight = float(getattr(cfg.HPO, "ACC_WEIGHT", 1.0))
    ece_weight = float(getattr(cfg.HPO, "ECE_WEIGHT", 1.0))

    valid = [
        r for r in rows
        if math.isfinite(_as_float(r, "accuracy"))
        and math.isfinite(_as_float(r, "ece"))
    ]
    if not valid:
        raise RuntimeError(
            "HPO selector 'weighted_acc_ece' requires finite ACC and ECE."
        )

    def score(row):
        return (
            acc_weight * _require_finite(row, "accuracy")
            - ece_weight * _require_finite(row, "ece")
        )

    selected = sorted(
        valid,
        key=lambda r: (
            -score(r),
            -_require_finite(r, "accuracy"),
            _require_finite(r, "ece"),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = (
        f"max {acc_weight} * ACC - {ece_weight} * ECE"
    )
    selected["_selection_score"] = score(selected)
    return selected


def _select_by_weighted_acc_aece(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    acc_weight = float(getattr(cfg.HPO, "ACC_WEIGHT", 1.0))
    aece_weight = float(getattr(cfg.HPO, "AECE_WEIGHT", 1.0))

    valid = [
        r for r in rows
        if math.isfinite(_as_float(r, "accuracy"))
        and math.isfinite(_as_float(r, "aece"))
    ]
    if not valid:
        raise RuntimeError(
            "HPO selector 'weighted_acc_aece' requires finite ACC and AECE."
        )

    def score(row):
        return (
            acc_weight * _require_finite(row, "accuracy")
            - aece_weight * _require_finite(row, "aece")
        )

    selected = sorted(
        valid,
        key=lambda r: (
            -score(r),
            -_require_finite(r, "accuracy"),
            _require_finite(r, "aece"),
            _stable_tag(r),
        ),
    )[0]

    selected = copy.deepcopy(selected)
    selected["_selection_rule"] = (
        f"max {acc_weight} * ACC - {aece_weight} * AECE"
    )
    selected["_selection_score"] = score(selected)
    return selected


_HPO_SELECTORS = {
    "acc": _select_by_acc,
    "accuracy": _select_by_acc,

    "ece": _select_by_ece,
    "aece": _select_by_aece,

    "acc_ece": _select_by_acc_ece_window,
    "acc_ece_window": _select_by_acc_ece_window,

    "acc_aece": _select_by_acc_aece_window,
    "acc_aece_window": _select_by_acc_aece_window,

    "weighted_acc_ece": _select_by_weighted_acc_ece,
    "weighted_acc_aece": _select_by_weighted_acc_aece,
}


def _select_best_row(rows: list[dict[str, Any]], cfg) -> dict[str, Any]:
    selector_name = str(getattr(cfg.HPO, "SELECTOR", "acc_ece_window"))

    if selector_name not in _HPO_SELECTORS:
        allowed = ", ".join(sorted(_HPO_SELECTORS))
        raise RuntimeError(
            f"Unknown HPO.SELECTOR={selector_name!r}. "
            f"Allowed selectors: {allowed}"
        )

    return _HPO_SELECTORS[selector_name](rows, cfg)


def _write_hpo_summary(
    base_output_dir: str,
    rows: list[dict[str, Any]],
    best_row: dict[str, Any] | None,
):
    os.makedirs(base_output_dir, exist_ok=True)

    json_path = osp.join(base_output_dir, "hpo_summary.json")
    csv_path = osp.join(base_output_dir, "hpo_summary.csv")
    best_path = osp.join(base_output_dir, "hpo_best_opts.json")

    rows_for_json = []
    best_candidate = None if best_row is None else best_row.get("candidate")

    for row in rows:
        item = dict(row)
        item["is_best"] = item.get("candidate") == best_candidate
        rows_for_json.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best": best_row,
                "rows": rows_for_json,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if rows:
        preferred = [
            "is_best",
            "candidate",
            "tag",
            "split",
            "accuracy",
            "ece",
            "aece",
            "_selection_rule",
            "_selection_score",
            "_best_accuracy",
            "_accuracy_threshold",
            "_num_eligible",
            "_num_finished",
            "output_dir",
        ]

        all_fields = sorted({k for row in rows_for_json for k in row.keys()})
        fieldnames = preferred + [k for k in all_fields if k not in preferred]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_for_json)

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


def _train_final_with_best_if_requested(
    base_args,
    base_cfg,
    best_row: dict[str, Any] | None,
):
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
    require_val = bool(getattr(base_cfg.HPO, "REQUIRE_VAL", True))
    n_bins = int(getattr(base_cfg.HPO, "N_BINS", 10))
    selector = str(getattr(base_cfg.HPO, "SELECTOR", "acc_ece_window"))

    base_output_dir = base_cfg.OUTPUT_DIR
    rows: list[dict[str, Any]] = []

    print("[HPO] enabled")
    print(f"[HPO] split={split}")
    print(f"[HPO] require_val={require_val}")
    print(f"[HPO] n_bins={n_bins}")
    print(f"[HPO] selector={selector}")
    print(f"[HPO] grid={grid}")

    for index, params in enumerate(_iter_grid(grid), start=1):
        tag = _make_candidate_tag(index, params)
        cand_args = _make_candidate_args(
            base_args=base_args,
            base_output_dir=base_output_dir,
            index=index,
            tag=tag,
            params=params,
        )
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

        metrics = _evaluate_candidate_for_hpo(
            trainer=trainer,
            split=split,
            require_val=require_val,
            n_bins=n_bins,
        )

        row = {
            "candidate": index,
            "tag": tag,
            "split": metrics.get("_actual_split", split),
            "output_dir": cand_cfg.OUTPUT_DIR,
            "params": {k: _normalise_json_value(v) for k, v in params.items()},
            "accuracy": float(metrics["accuracy"]),
            "ece": float(metrics["ece"]),
            "aece": float(metrics["aece"]),
        }

        for key, value in params.items():
            row[key] = _normalise_json_value(value)

        rows.append(row)

        print(
            "[HPO] candidate "
            f"{index} "
            f"ACC={row['accuracy']:.4f} "
            f"ECE={row['ece']:.4f} "
            f"AECE={row['aece']:.4f}"
        )

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_row = _select_best_row(rows, base_cfg)

    print("=" * 80)
    print(f"[HPO] selector={selector}")
    print(f"[HPO] selection_rule={best_row.get('_selection_rule')}")
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