"""Three training objectives with the repo's *existing* DAVI and Branch-MC.

Only Original-MC is new. No core/method/trainer files need editing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "core" / "registry.py").is_file():
    raise RuntimeError(f"Expected this file under MMRL/experiments/objective_fidelity; resolved ROOT={ROOT}")
sys.path.insert(0, str(ROOT))

MODES = {"davi": "DetBayesRTMMRL", "branch_mc": "BayesRTMMRL", "original_mc": "FidelityOriginalMC"}
DAVI_CFG = ROOT / "configs/methods/det_bayesrt_mmrl_clamp.yaml"
AVI_CFG = ROOT / "configs/methods/bayesrt_mmrl.yaml"


def original_mixture_nll(logits_main, logits_rep, labels, alpha):
    """Mean_{s,b}[-log(alpha*p_main(y|s)+(1-alpha)*p_rep(y|s))].

    Mixture is INSIDE log and per MC sample; this is not CE(mean_probs).
    """
    import torch
    import torch.nn.functional as F

    logc = F.log_softmax(logits_main.float(), dim=-1)
    logr = F.log_softmax(logits_rep.float(), dim=-1)
    idx = labels.long()[None, :, None].expand(logc.shape[0], -1, 1)
    yc = logc.gather(2, idx).squeeze(-1)
    yr = logr.gather(2, idx).squeeze(-1)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"ALPHA must be in [0,1], got {alpha}")
    if alpha == 0.0:
        return -yr.mean()
    if alpha == 1.0:
        return -yc.mean()
    return -torch.logaddexp(yc + math.log(alpha), yr + math.log1p(-alpha)).mean()


def register_original_mc():
    """Register only the experiment's new objective, not another Branch-MC."""
    from core.registry import METHOD_REGISTRY
    from methods.bayesrt_mmrl.model import BayesRTMMRLMethod

    if "FidelityOriginalMC" in METHOD_REGISTRY.keys():
        return

    class FidelityOriginalMC(BayesRTMMRLMethod):
        method_name = "FidelityOriginalMC"
        # Intentionally inherits model_cls = BayesRTMMRLModel (unchanged AVI).

        def _build_train_outputs(self, label, img_ref, out):
            result = super()._build_train_outputs(label, img_ref, out)
            c = out["logits_main_stack"][..., : self.num_classes]
            r = out["logits_rep_stack"][..., : self.num_classes]
            alpha = float(self.cfg.BAYESRT_MMRL.ALPHA)
            original = original_mixture_nll(c, r, label, alpha)
            # Reuse original AVI branch CE, cosine, KL and all normalization.
            branch = alpha * result.losses["loss_main"] + (1.0 - alpha) * result.losses["loss_rep"]
            delta = original - branch
            result.losses["data_term"] = result.losses["data_term"] + delta
            result.losses["total"] = result.losses["total"] + delta
            result.losses["original_mc_nll"] = original.detach()
            return result

    METHOD_REGISTRY.register("FidelityOriginalMC")(FidelityOriginalMC)


def output_directory(output_root, profile, mode, dataset, shot, seed):
    base = Path(output_root)
    if not base.is_absolute():
        base = ROOT / base
    return base / profile / mode / dataset / f"shot_{shot}" / f"seed{seed}"


def build_run_args(options, mode, diagnostic=False):
    """Make a namespace matching repo.run.main/setup_cfg; keep original YAMLs."""
    from argparse import Namespace

    method_yaml = DAVI_CFG if mode == "davi" else AVI_CFG

    dest = output_directory(options.output_root, options.profile, mode,
                            options.dataset, options.shot, options.seed)
    overrides = [
        "DATASET.NUM_SHOTS", str(options.shot),
        "DATASET.SUBSAMPLE_CLASSES", "all",
        "HPO.ENABLED", "False",  # explicit: no HPO in this small comparison
    ]
    if options.epochs is not None:
        overrides += ["OPTIM.MAX_EPOCH", str(options.epochs)]
    if options.mc_train is not None:
        overrides += ["BAYESRT_MMRL.N_MC_TRAIN", str(options.mc_train)]
    data_root = Path(options.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    return Namespace(
        root=str(data_root), output_dir=str(dest),
        dataset_config_file=str(ROOT / "configs/datasets" / f"{options.dataset}.yaml"),
        method_config_file=str(method_yaml),
        protocol_config_file=str(ROOT / "configs/protocols/fs.yaml"),
        runtime_config_file=str(ROOT / "configs/runtime/mmrl_family.yaml"),
        exp_config="", method=MODES[mode], protocol="FS", exec_mode="online",
        seed=options.seed, trainer="RefactorRunner", eval_only=False,
        model_dir="", load_epoch=None, no_train=False, opts=overrides,
    )


def davi_diagnostic_params():
    import yaml
    doc = yaml.safe_load(DAVI_CFG.read_text(encoding="utf-8"))
    section = doc["BAYESRT_MMRL"]
    return float(section["DET_VAR_SCALE"]), float(section["DET_VAR_CLAMP"])


def main():
    p = argparse.ArgumentParser(description="Native DAVI / native Branch-MC / new Original-MC")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--dataset", required=True)
    p.add_argument("--shot", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--profile", choices=["native"], default="native", help="Only native is supported")
    p.add_argument("--epochs", type=int)
    p.add_argument("--mc-train", type=int)
    p.add_argument("--gpu", default="0")
    p.add_argument("--data-root", default="DATASETS")
    p.add_argument("--output-root", default="output_objective_fidelity_v3")
    o = p.parse_args()
    if o.shot < 1 or o.seed < 0 or (o.epochs is not None and o.epochs < 1) or (o.mc_train is not None and o.mc_train < 1):
        p.error("shot>=1, seed>=0, epochs>=1, and mc-train>=1 are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(o.gpu)
    os.chdir(ROOT)  # run.py and Dassl expect repo-relative paths

    register_original_mc()
    import run as repo_run
    from core.config import setup_cfg
    args = build_run_args(o, o.mode)
    for name in ("method_config_file", "dataset_config_file", "protocol_config_file", "runtime_config_file"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(getattr(args, name))

    folder = Path(args.output_dir)
    if folder.exists():
        raise FileExistsError(f"Output already exists (won't overwrite): {folder}")
    cfg = setup_cfg(args)
    if str(cfg.METHOD.NAME) != MODES[o.mode]:
        raise AssertionError(f"Unexpected METHOD.NAME: {cfg.METHOD.NAME}")

    eta, clamp = davi_diagnostic_params()
    # For native AVI/Original this is the DAVI surrogate used ONLY in diagnostics.
    if o.mode == "davi":
        if (float(cfg.BAYESRT_MMRL.DET_VAR_SCALE), float(cfg.BAYESRT_MMRL.DET_VAR_CLAMP)) != (eta, clamp):
            raise AssertionError("DAVI training and diagnostic variance settings differ")

    folder.mkdir(parents=True)
    (folder / "effective_config.yaml").write_text(cfg.dump(), encoding="utf-8")
    manifest = {
        "mode": o.mode, "profile": o.profile, "dataset": o.dataset,
        "shot": o.shot, "seed": o.seed, "method": MODES[o.mode],
        "output_dir": str(folder), "root": args.root,
        "dataset_config_file": args.dataset_config_file,
        "method_config_file": args.method_config_file,
        "protocol_config_file": args.protocol_config_file,
        "runtime_config_file": args.runtime_config_file,
        "opts": args.opts,
        "effective_config_sha256": hashlib.sha256(cfg.dump().encode()).hexdigest(),
        "det_diagnostic_scale": eta, "det_diagnostic_clamp": clamp,
        "r_prior_mode": str(cfg.BAYESRT_MMRL.R_PRIOR_MODE),
        "r_prior_std": float(cfg.BAYESRT_MMRL.R_PRIOR_STD),
        "t_prior_std": float(cfg.BAYESRT_MMRL.T_PRIOR_STD),
        "r_kl_weight": float(cfg.BAYESRT_MMRL.R_KL_WEIGHT),
        "t_kl_weight": float(cfg.BAYESRT_MMRL.T_KL_WEIGHT),
        "kl_warmup_epochs": int(cfg.BAYESRT_MMRL.KL_WARMUP_EPOCHS),
        "n_mc_train": int(cfg.BAYESRT_MMRL.N_MC_TRAIN),
        "max_epochs": int(cfg.OPTIM.MAX_EPOCH),
    }
    (folder / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ObjectiveFidelity] profile={o.profile}, mode={o.mode}, method={MODES[o.mode]}")
    print(f"[ObjectiveFidelity] R_prior_mode={manifest['r_prior_mode']}, DAVI diagnostic clamp={clamp}")
    start = time.perf_counter()
    repo_run.main(args)
    seconds = time.perf_counter() - start
    (folder / "elapsed_wall_seconds.txt").write_text(f"{seconds:.3f}\n", encoding="utf-8")
    print(f"[ObjectiveFidelity] finished in {seconds:.1f}s (includes setup+train+test; NOT matched-time)")


if __name__ == "__main__":
    main()
