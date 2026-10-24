"""E6: representation, posterior learning, objective gradients, and calibration.

Complete replacement of historical E6 (no imports from old E6 or v2).
Run from MMRL: python -m experiments.supplementary.e6 --help
Requires existing native AVI-B/DAVI supplementary checkpoints; read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from .core import ROOT, checkpoint_path, location, save_json, sha256
from . import e6_analysis as analysis

SCHEMA = 3
GROUPS = ("r_mean", "r_rho", "t_rho", "phi")


def _resolve(run_dir, request):
    if request == "latest":
        _, _, ep = checkpoint_path(run_dir, "latest")
        return "latest", None, ep
    ep = int(request)
    _, _, last = checkpoint_path(run_dir, "latest")
    if ep == last:
        return "latest", None, ep
    checkpoint_path(run_dir, "trajectory", ep)
    return "trajectory", ep, ep


def _paired_locations(root, ds, shot, seed, epoch):
    runs = {m: location(root, "native", ds, m, shot, seed) for m in ("branch", "davi")}
    choices = {m: _resolve(runs[m], epoch) for m in runs}
    if choices["branch"][2] != choices["davi"][2]:
        raise RuntimeError("AVI-B/DAVI checkpoint epochs differ")
    return runs, choices, choices["branch"][2]


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _flatten(record):
    out = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub, nested in value.items():
                if isinstance(nested, dict):
                    for field, v in nested.items():
                        if isinstance(v, (int, float, str)) or v is None:
                            out[f"{key}_{sub}_{field}"] = v
                elif isinstance(nested, (int, float, str)) or nested is None:
                    out[f"{key}_{sub}"] = nested
        elif isinstance(value, (int, float, str)) or value is None:
            out[key] = value
    return out


def _aggregate_seed_rows(rows, categories):
    buckets = {}
    for row in rows:
        key = tuple(row.get(c) for c in categories)
        buckets.setdefault(key, []).append(row)
    summary = []
    ids = {"shot", "seed", "epoch", "n_images", "input_sha256", "checkpoint_sha256",
           "branch_checkpoint_sha256", "davi_checkpoint_sha256", "mean_source_sha256",
           "mc_draws", "mc_repeats", "samples", "repeats", "draws", "native_mc_budget",
           "schema", "previous_epoch", "delta_epochs", "n_valid"}
    for key, parts in sorted(buckets.items()):
        line = dict(zip(categories, key))
        line["n_seeds"] = len(set(part["seed"] for part in parts))
        line["seeds"] = " ".join(str(k) for k in sorted(set(p["seed"] for p in parts)))
        for field in dict.fromkeys(k for p in parts for k in p):
            if field in categories or field in ids or field.startswith("dynamics_previous_epoch"):
                continue
            vals = [p[field] for p in parts if isinstance(p.get(field), (int, float))
                    and not isinstance(p[field], bool) and math.isfinite(float(p[field]))]
            if vals:
                line[field + "_mean"] = statistics.mean(vals)
                line[field + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        summary.append(line)
    return summary


def _cached(path, args, controls, calculate):
    controls = {**controls, "checkpoint_shas": args._checkpoint_shas}
    if path.exists() and not args.overwrite:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA or raw.get("controls") != controls:
            raise RuntimeError(f"Existing E6 result is incompatible: {path}; set FORCE_EVAL=1")
        print("[E6 REUSE]", path, flush=True)
        return raw
    computed = calculate()
    result = {"schema": SCHEMA, "controls": controls, **computed}
    save_json(path, result)
    print("[E6 WRITE]", path, flush=True)
    return result


def _features(args, folder, runs, choices, epoch):
    target = folder / "features.json"
    controls = {"images": args.images, "batch": args.batch, "input_seed": args.input_seed,
                "posterior_draws": args.posterior_draws, "sample_seed": args.sample_seed}
    def compute():
        a = analysis.extract_representation(runs["branch"], args.gpu, *choices["branch"][:2],
                                            args.images, args.batch, args.input_seed)
        d = analysis.extract_representation(runs["davi"], args.gpu, *choices["davi"][:2],
                                            args.images, args.batch, args.input_seed)
        metrics, f = analysis.compare_representation(a, d, args.posterior_draws, args.sample_seed)
        tensors = {"labels": a["labels"].numpy(), "rep_hidden_A": a["hidden"].numpy(),
                   "rep_hidden_D": d["hidden"].numpy()}
        tensors.update({"F_" + k: t.numpy() for k, t in f.items() if t is not None})
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "cross_projection.npz", **tensors)
        return {"metrics": metrics,
                "checkpoint": {"branch": a["checkpoint_sha256"], "davi": d["checkpoint_sha256"]},
                "note": "Mean projection attribution, fixed parameter state, not training causality"}
    # For a reused result require the corresponding raw feature arrays, not just JSON.
    if target.exists() and not args.overwrite and not (folder / "cross_projection.npz").is_file():
        raise FileNotFoundError(f"E6 feature details missing: {folder / 'cross_projection.npz'}")
    return _cached(target, args, controls, compute)


def _native(args, folder, runs, choices):
    for method in ("branch", "davi"):
        dest = folder / f"gradients_{method}.json"
        controls = {"grad_batch": args.grad_batch, "draws": args.grad_draws if method=="branch" else 1,
                    "sample_seed": args.sample_seed, "input_seed": args.input_seed,
                    "include_phi": not args.no_phi}
        def compute(method=method):
            return analysis.native_gradients(runs[method], args.gpu, *choices[method][:2],
                 args.grad_batch, args.input_seed, args.grad_draws, args.sample_seed,
                 include_phi=not args.no_phi)
        _cached(dest, args, controls, compute)


def _objective(args, folder, runs, choices):
    for method in ("branch", "davi"):
        dest = folder / f"objective_{method}.json"
        controls = {"grad_batch": args.grad_batch, "mc_ref_draws": args.mc_ref_draws,
                    "mc_ref_repeats": args.mc_ref_repeats, "sample_seed": args.sample_seed,
                    "input_seed": args.input_seed, "include_phi": not args.no_phi,
                    "davi_var_scale": args.davi_var_scale, "davi_var_clamp": args.davi_var_clamp}
        def compute(method=method):
            return analysis.same_checkpoint_objectives(runs[method], args.gpu, *choices[method][:2],
                 args.grad_batch, args.input_seed, args.mc_ref_draws, args.mc_ref_repeats,
                 args.sample_seed, include_phi=not args.no_phi,
                 davi_scale=args.davi_var_scale, davi_clamp=args.davi_var_clamp)
        _cached(dest, args, controls, compute)


def _prediction(args, folder, runs, choices):
    for method in ("branch", "davi"):
        other = "davi" if method == "branch" else "branch"
        _, ck, _ = checkpoint_path(runs[other], choices[other][0], choices[other][1])
        other_sha = sha256(ck)
        if args.mean_swap:
            # A lightweight native checkpoint contains the complete trainable R mean.
            other_state = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
            mean = other_state[analysis.GROUP_NAMES["r_mean"]].detach().float()
        else:
            mean = None
        dest = folder / f"prediction_{method}.json"
        controls = {"prediction_batch": args.prediction_batch, "samples": args.prediction_samples,
                    "chunk": args.prediction_chunk, "repeats": args.prediction_repeats,
                    "sample_seed": args.sample_seed, "sigmas": list(args.sigma_scales),
                    "mean_swap": args.mean_swap, "mean_source_sha256": other_sha if args.mean_swap else None}
        def compute(method=method, mean=mean):
            return analysis.prediction_interventions(runs[method], args.gpu, *choices[method][:2],
                 args.prediction_batch, args.prediction_samples, args.prediction_chunk,
                 args.prediction_repeats, args.sample_seed, sigmas=args.sigma_scales,
                 mean_override=mean, mean_source_sha=other_sha if args.mean_swap else None)
        _cached(dest, args, controls, compute)


def reports(root):
    root = Path(root)
    feature_rows = []
    native_rows = []
    objective_rows = []
    prediction_repeats = []
    for folder in sorted(root.glob("*/shots_*/seed_*/epoch_*")):
        for path in sorted(folder.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != SCHEMA:
                continue  # Never treat historical E6 v1 outputs as E6 reconstructed data.
            if path.name == "features.json":
                feature_rows.append(data["metrics"])
            elif path.name.startswith("gradients_"):
                native_rows.append(_flatten(data))
            elif path.name.startswith("objective_"):
                objective_rows.append(_flatten(data))
            elif path.name.startswith("prediction_"):
                for r in data["results"]:
                    prediction_repeats.append({**{k:data[k] for k in ("dataset","method","seed","shot","epoch")}, **r})
    for name, rows, grouping in (
        ("feature", feature_rows, ("dataset", "epoch")),
        ("gradient", native_rows, ("dataset", "method", "epoch")),
        ("objective", objective_rows, ("dataset", "method", "epoch"))):
        _write_csv(root / f"{name}_pairs.csv", rows)
        _write_csv(root / f"{name}_summary.csv", _aggregate_seed_rows(rows, grouping) if rows else [])
    _write_csv(root / "prediction_repeats.csv", prediction_repeats)
    # MC repeats share the same trained model and are NOT additional independent seeds.
    seed_groups = {}
    for r in prediction_repeats:
        key = (r["dataset"], r["method"], r["seed"], r["epoch"], r["condition"], r["gamma_r"], r["gamma_t"])
        seed_groups.setdefault(key, []).append(r)
    metrics = ("acc", "nll", "ece_15_equal_width", "aece_15_equal_count", "confidence_mean")
    seed_rows = []
    for key, members in sorted(seed_groups.items()):
        row = dict(zip(("dataset", "method", "seed", "epoch", "condition", "gamma_r", "gamma_t"), key))
        row.update({m: statistics.mean(r[m] for r in members) for m in metrics})
        seed_rows.append(row)
    _write_csv(root / "prediction_seed_means.csv", seed_rows)
    _write_csv(root / "prediction_summary.csv",
               _aggregate_seed_rows(seed_rows, ("dataset", "method", "epoch", "condition", "gamma_r", "gamma_t")) if seed_rows else [])
    baselines = {(r["dataset"], r["method"], r["seed"], r["epoch"]):r for r in seed_rows
                 if r["condition"] == "native"}
    changes = []
    for r in seed_rows:
        base = baselines.get((r["dataset"], r["method"], r["seed"], r["epoch"]))
        if base is not None:
            changes.append({**{k:r[k] for k in ("dataset", "method", "seed", "epoch", "condition", "gamma_r", "gamma_t")},
                            **{m+"_delta":r[m]-base[m] for m in metrics}})
    _write_csv(root / "prediction_delta_seed.csv", changes)
    print(f"[E6 REPORT] {len(feature_rows)} feature pairs, {len(native_rows)} native gradients, "
          f"{len(objective_rows)} objective comparisons, {len(seed_rows)} seed-level predictions", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output_supplementary_v7")
    parser.add_argument("--datasets", nargs="+", default=["caltech101", "eurosat", "oxford_pets"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--epochs", nargs="+", default=["5", "25", "latest"])
    parser.add_argument("--prediction-epoch", default="latest")
    parser.add_argument("--shot", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--mode", choices=("all", "representation", "native", "objective", "prediction", "report"), default="all")
    parser.add_argument("--images", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--posterior-draws", type=int, default=16)
    parser.add_argument("--grad-batch", type=int, default=32)
    parser.add_argument("--grad-draws", type=int, default=20)
    parser.add_argument("--mc-ref-draws", type=int, default=128)
    parser.add_argument("--mc-ref-repeats", type=int, default=2)
    parser.add_argument("--no-phi", action="store_true", help="Exclude shared representation gradients")
    parser.add_argument("--davi-var-scale", type=float, default=1.)
    parser.add_argument("--davi-var-clamp", type=float, default=50.)
    parser.add_argument("--prediction-batch", type=int, default=32)
    parser.add_argument("--prediction-samples", type=int, default=50)
    parser.add_argument("--prediction-chunk", type=int, default=10)
    parser.add_argument("--prediction-repeats", type=int, default=3)
    parser.add_argument("--sigma-scales", nargs="+", type=float, default=[0.5, 2.])
    parser.add_argument("--mean-swap", action="store_true", default=True)
    parser.add_argument("--no-mean-swap", action="store_false", dest="mean_swap")
    parser.add_argument("--sample-seed", type=int, default=99173)
    parser.add_argument("--input-seed", type=int, default=2027)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    sizes = (args.shot, args.images, args.batch, args.grad_batch, args.grad_draws,
             args.mc_ref_draws, args.mc_ref_repeats, args.prediction_batch,
             args.prediction_samples, args.prediction_chunk, args.prediction_repeats)
    if any(v<1 for v in sizes):
        parser.error("All shot/batch/draw/sample settings must be positive")
    if args.grad_draws < 10 and args.mode in ("all", "native"):
        parser.error("E6 grad-draws must be >= 10 (native AVI-B MC budget)")
    if args.mc_ref_repeats < 2 and args.mode in ("all", "objective"):
        parser.error("E6 MC reference requires >=2 independent repeats")
    if args.posterior_draws < 0 or any(not math.isfinite(s) or s<0 for s in args.sigma_scales):
        parser.error("Posterior draw count must be nonnegative; sigma scales finite and >= 0")
    root = Path(args.output)
    root = (root if root.is_absolute() else ROOT/root).resolve()
    dest = root / "summary" / "e6"
    if args.mode == "report":
        reports(dest)
        return
    stages = (("representation", "native", "objective", "prediction") if args.mode == "all" else (args.mode,))
    for ds in args.datasets:
        for seed in args.seeds:
            epochs = list(dict.fromkeys(([args.prediction_epoch] if args.mode == "prediction" else
                                         args.epochs + ([args.prediction_epoch] if "prediction" in stages else []))))
            for request in epochs:
                try:
                    runs, choices, ep = _paired_locations(root, ds, args.shot, seed, request)
                except (ValueError, FileNotFoundError) as exc:
                    print(f"[E6 SKIP missing paired checkpoints] {ds} seed={seed} epoch={request}: {exc}", flush=True)
                    continue
                folder = dest / ds / f"shots_{args.shot}" / f"seed_{seed}" / f"epoch_{ep}"
                args._checkpoint_shas = {
                    m: sha256(checkpoint_path(runs[m], choices[m][0], choices[m][1])[1])
                    for m in ("branch", "davi")
                }
                for stage in stages:
                    if stage == "prediction" and request != args.prediction_epoch:
                        continue
                    if stage != "prediction" and request not in args.epochs:
                        continue
                    if stage == "representation":
                        _features(args, folder, runs, choices, ep)
                    elif stage == "native":
                        _native(args, folder, runs, choices)
                    elif stage == "objective":
                        _objective(args, folder, runs, choices)
                    elif stage == "prediction":
                        _prediction(args, folder, runs, choices)
    reports(dest)
    print("[E6 DONE]", dest, flush=True)

if __name__ == "__main__":
    main()
