"""Evaluate all three likelihood objectives on an existing trained checkpoint.

Loads each method as ITSELF: DAVI->DetBayesRTMMRL; Branch->BayesRTMMRL;
Original->FidelityOriginalMC. No cross-class checkpoint substitution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "core" / "registry.py").is_file():
    raise RuntimeError(f"Unexpected ROOT: {ROOT}. Place scripts in MMRL/experiments/objective_fidelity/.")
sys.path.insert(0, str(ROOT))

from train import MODES, build_run_args, original_mixture_nll, output_directory, register_original_mc


def checkpoint_path(folder):
    area = folder / "refactor_model"
    best = area / "model-best.pth.tar"
    if best.exists():
        return best, None
    checkpoints = list(area.glob("model.pth.tar-*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {area}")
    chosen = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    return chosen, int(chosen.name.rsplit("-", 1)[-1])


def verify_loaded_checkpoint(path, model):
    """Ensure key posterior parameters actually match the saved checkpoint."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    actual = model.state_dict()
    suffixes = (
        "image_encoder.bayes_proj_rep.posterior_mean",
        "image_encoder.bayes_proj_rep.posterior_rho",
        "text_posterior.posterior_rho",
    )
    for suffix in suffixes:
        keys = [k for k in state if k.endswith(suffix)]
        if len(keys) != 1:
            raise RuntimeError(f"Expected precisely one checkpoint tensor {suffix}, found {keys}")
        k = keys[0]
        if k not in actual or not torch.equal(state[k].float().cpu(), actual[k].float().cpu()):
            raise RuntimeError(f"Checkpoint tensor missing or not loaded: {k}")


def posterior_stats(model):
    import torch
    r, t = model.image_encoder.bayes_proj_rep, model.text_posterior
    if r is None or t is None:
        raise RuntimeError("Both R and T posterior distributions are required")
    with torch.no_grad():
        return {
            "r_mean": r.posterior_mean.float().detach().cpu().clone(),
            "r_mean_norm": float(r.posterior_mean.float().norm()),
            "r_trace": float(r.expanded_posterior_sigma().square().sum()),
            "t_trace": float(t.expanded_posterior_sigma().square().sum()),
        }


def common_moments(model, images, cfg):
    """Reuse DAVI's existing moment math on the actual native AVI model.

    DetBayesRTMMRLModel.forward_joint_moments only uses properties shared with
    BayesRTMMRLModel, plus det_moment_eps/logit_scale. Set the latter only for
    the diagnostic call. No parameters or model class are replaced.
    """
    if hasattr(model, "forward_joint_moments"):
        return model.forward_joint_moments(images)
    from methods.det_bayesrt_mmrl.modules import DetBayesRTMMRLModel
    model.det_moment_eps = float(getattr(cfg, "DET_MOMENT_EPS", 1e-6))
    model.logit_scale = float(getattr(cfg, "LOGIT_SCALE", 100.0))
    return DetBayesRTMMRLModel.forward_joint_moments(model, images)


def diagnose(trainer, samples, chunk, eta, clamp, limit_batches=None):
    import torch
    import torch.nn.functional as F
    from methods.det_bayesrt_mmrl.losses import det_jensen_ce

    method, model = trainer.method, trainer.method.model
    cfg = method.cfg.BAYESRT_MMRL
    model.eval()
    # Exactly the original training loader, including its transformations.
    loader = trainer.dm.train_loader_x
    alpha = float(cfg.ALPHA)
    totals = {k: 0.0 for k in ("original_nll", "branch_nll", "det_nll", "alignment")}
    examples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            images = batch["img"].to(method.device)
            labels = batch["label"].long().to(method.device)
            n = int(labels.numel())
            img_ref = method.image_encoder_clip(images.type(method.dtype))
            img_ref = F.normalize(img_ref, dim=-1)
            original_b = branch_b = 0.0
            for k in range(0, samples, chunk):
                s = min(chunk, samples - k)
                out = model.forward_joint(images, s, use_posterior_mean=False)
                c = out["logits_main_stack"][..., :method.num_classes]
                r = out["logits_rep_stack"][..., :method.num_classes]
                original_b += float(original_mixture_nll(c, r, labels, alpha)) * s / samples
                logc = F.log_softmax(c.float(), dim=-1)
                logr = F.log_softmax(r.float(), dim=-1)
                idx = labels[None, :, None].expand(s, -1, 1)
                branch_b += float((
                    -alpha * logc.gather(2, idx).squeeze(-1)
                    - (1.0 - alpha) * logr.gather(2, idx).squeeze(-1)
                ).mean()) * s / samples

            mom = common_moments(model, images, cfg)
            det_b = (alpha * det_jensen_ce(mom["mu_main"], mom["var_main"], labels,
                                            var_scale=eta, var_clamp=clamp)
                     + (1.0 - alpha) * det_jensen_ce(mom["mu_rep"], mom["var_rep"], labels,
                                                    var_scale=eta, var_clamp=clamp))
            text_ref = method.text_features_clip[:method.num_classes].to(method.device)
            align = float(cfg.REG_WEIGHT) * (
                1.0 - F.cosine_similarity(mom["image_features_main"], img_ref, dim=-1).mean()
                + 1.0 - F.cosine_similarity(mom["text_features"][:method.num_classes],
                                            text_ref, dim=-1).mean()
            )
            for key, val in (("original_nll", original_b), ("branch_nll", branch_b),
                             ("det_nll", float(det_b)), ("alignment", float(align))):
                totals[key] += n * val
            examples += n

    if examples == 0:
        raise RuntimeError("No diagnostic batches available")
    v = {k: value / examples for k, value in totals.items()}
    if v["branch_nll"] + 1e-5 < v["original_nll"]:
        raise RuntimeError("Jensen check failed: branch_nll should not be below original_nll")

    # This is the existing code's KL weighting/normalization at beta=1.
    # Not a new ELBO normalization. For native profile these KL configs differ.
    kl = model.kl_terms()
    rpost, tpost = model.image_encoder.bayes_proj_rep, model.text_posterior
    num_batches = max(1, len(loader))
    kl_r = float(cfg.R_KL_WEIGHT) * float(kl["r_proj"]) / max(1, rpost.prior_mean.numel()) / num_batches
    kl_t = float(cfg.T_KL_WEIGHT) * float(kl["t_proj"]) / max(1, tpost.prior_mean.numel()) / num_batches
    v["kl_r"] = kl_r
    v["kl_t"] = kl_t
    for src, dst in (("original_nll", "original_reference_loss"),
                     ("branch_nll", "branch_reference_loss"),
                     ("det_nll", "deterministic_reference_loss")):
        v[dst] = v[src] + v["alignment"] + kl_r + kl_t
    v["mix_gap"] = v["branch_nll"] - v["original_nll"]
    v["det_gap"] = v["det_nll"] - v["branch_nll"]
    v["total_gap"] = v["det_nll"] - v["original_nll"]
    v["num_examples"] = examples
    v["mc_reference_samples"] = samples
    v["det_var_scale"] = eta
    v["det_var_clamp"] = clamp
    v["kl_beta_reference"] = 1.0
    return v


def main():
    p = argparse.ArgumentParser(description="Checkpoint-based objective-fidelity diagnostics")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--dataset", required=True)
    p.add_argument("--shot", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--profile", choices=["native", "controlled"], default="native")
    p.add_argument("--gpu", default="0")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--limit-batches", type=int, default=None, help="Only for smoke testing")
    p.add_argument("--output-root", default="output_objective_fidelity_v3")
    o = p.parse_args()
    if o.samples < 1 or o.chunk < 1 or (o.limit_batches is not None and o.limit_batches < 1):
        p.error("samples, chunk and limit-batches must be >=1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(o.gpu)
    os.chdir(ROOT)

    folder = output_directory(o.output_root, o.profile, o.mode, o.dataset, o.shot, o.seed)
    mf_path = folder / "experiment_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(f"Training manifest missing: {mf_path}")
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    for k in ("mode", "profile", "dataset", "shot", "seed"):
        if mf[k] != getattr(o, k):
            raise ValueError(f"Manifest mismatch: {k}={mf[k]!r} vs CLI {getattr(o, k)!r}")
    register_original_mc()
    import run as repo_run
    from core.config import setup_cfg
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    from datasets.fair_val_protocol import install_kplusval_datasetbase_patch
    from argparse import Namespace

    repo_run._import_runtime_modules()
    # Restore exact training config sources, arguments, and training output dir.
    original_ns = Namespace(**{
        "root": mf["root"], "output_dir": mf["output_dir"],
        "dataset_config_file": mf["dataset_config_file"],
        "method_config_file": mf["method_config_file"],
        "protocol_config_file": mf["protocol_config_file"],
        "runtime_config_file": mf["runtime_config_file"],
        "exp_config": "", "method": mf["method"], "protocol": "FS", "exec_mode": "online",
        "seed": mf["seed"], "opts": mf["opts"], "trainer": "RefactorRunner",
    })
    cfg = setup_cfg(original_ns)
    sha = hashlib.sha256(cfg.dump().encode()).hexdigest()
    if sha != mf["effective_config_sha256"]:
        raise RuntimeError("Training YAML/configuration changed since checkpoint creation; refuse silent mismatch")
    if str(cfg.METHOD.NAME) != MODES[o.mode]:
        raise RuntimeError("Resolved method mismatch")
    install_kplusval_datasetbase_patch(cfg)
    set_random_seed(o.seed)
    trainer = build_trainer(cfg)
    chosen, epoch = checkpoint_path(folder)
    trainer.load_model(str(folder), epoch=epoch)
    verify_loaded_checkpoint(chosen, trainer.method.model)
    posterior = posterior_stats(trainer.method.model)
    if posterior["r_mean_norm"] < 1e-10:
        print("[WARN] visual posterior mean is near zero: moment approximation may be degenerate")
    eta, clamp = mf["det_diagnostic_scale"], mf["det_diagnostic_clamp"]
    metrics = diagnose(trainer, o.samples, o.chunk, eta, clamp, o.limit_batches)
    metrics.update({k: mf[k] for k in ("mode", "profile", "dataset", "shot", "seed")})
    metrics.update({"r_trace": posterior["r_trace"], "t_trace": posterior["t_trace"],
                    "r_mean_norm": posterior["r_mean_norm"], "checkpoint_path": str(chosen),
                    "r_prior_mode": mf["r_prior_mode"],
                    "diagnostic_partial": o.limit_batches is not None})
    name = "fidelity_smoke.json" if o.limit_batches is not None else "fidelity.json"
    (folder / name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if o.limit_batches is None:
        import torch
        torch.save({"r_mean": posterior["r_mean"]}, folder / "posterior_diagnostic.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
