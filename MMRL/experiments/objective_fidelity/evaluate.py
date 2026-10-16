"""V5: fixed-train AND held-out-test reference diagnostics for all three methods.

All three objective NLLs are evaluated on the SAME samples/inputs at each
checkpoint; only the inference-time diagnostics vary. Do not confuse test
reference NLL with the native trainer's static_logit predictive test NLL.
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

from train import MODES, original_mixture_nll, output_directory, register_original_mc


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
            "r_sigma": r.expanded_posterior_sigma().float().detach().cpu().clone(),
            "t_mean": t.posterior_mean.float().detach().cpu().clone(),
            "t_sigma": t.expanded_posterior_sigma().float().detach().cpu().clone(),
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


def deterministic_reference_loaders(trainer, batch_size):
    """Two *fixed* diagnostic input streams: train_X and held-out test.

    Copy Dassl DatasetWrapper rather than changing the training loader in place.
    Both splits use EXACTLY the test dataset's deterministic transform; use a
    sequential sampler and disable the DatasetWrapper train-time K transforms.
    Fail if required implementation attributes are unavailable.
    """
    import copy
    from torch.utils.data import DataLoader, SequentialSampler

    dm = trainer.dm
    train_source = dm.train_loader_x.dataset
    test_source = dm.test_loader.dataset
    for label, dataset in (("train", train_source), ("test", test_source)):
        if not hasattr(dataset, "transform") or not hasattr(dataset, "data_source"):
            raise RuntimeError(f"{label} loader does not use expected Dassl DatasetWrapper: {type(dataset)}")
    test_transform = test_source.transform
    if test_transform is None:
        raise RuntimeError("Test DatasetWrapper has no deterministic transform; refuse ambiguous reference evaluation")
    # These sanity checks protect against a mistaken random training transform.
    tfm_name = repr(test_transform).lower()
    for token in ("randomresizedcrop", "randomhorizontalflip", "randaugment", "randomcrop", "randomrotation", "randomaffine", "colorjitter"):
        if token in tfm_name:
            raise RuntimeError(f"Test transform unexpectedly stochastic: {test_transform!r}")
    loaders = {}
    for split, dataset in (("train_fixed", train_source), ("test", test_source)):
        fixed = copy.copy(dataset)
        fixed.transform = test_transform
        fixed.is_train = False
        if hasattr(fixed, "k_tfm"):
            fixed.k_tfm = 1
        loaders[split] = DataLoader(
            fixed, batch_size=batch_size, sampler=SequentialSampler(fixed),
            num_workers=0, pin_memory=False, drop_last=False,
        )
    return loaders, repr(test_transform)


def diagnose(trainer, loader, samples, chunk, eta, clamp, split_name,
             reference_transform, limit_batches=None):
    """Evaluate all three objectives on precisely the same frozen checkpoint/input.

    The MC expectation averages per-sample losses, NEVER -log(mean(probs)).
    Labels, inputs and MC logit draws are shared across Original and Branch;
    deterministic moments use the same images. The named split is explicit.
    """
    import torch
    import torch.nn.functional as F
    from methods.det_bayesrt_mmrl.losses import det_jensen_ce

    method, model = trainer.method, trainer.method.model
    cfg = method.cfg.BAYESRT_MMRL
    model.eval()
    alpha = float(cfg.ALPHA)
    totals = {k: 0.0 for k in (
        "original_nll", "branch_nll", "det_nll", "alignment", "main_nll", "rep_nll",
        "main_accuracy", "rep_accuracy", "static_prob_accuracy", "static_logit_accuracy",
    )}
    examples = batches = 0
    input_digest = hashlib.sha256()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            images = batch["img"].to(method.device)
            labels = batch["label"].long().to(method.device)
            n = int(labels.numel())
            # Digest every observed input and label (before any stochastic draw).
            input_digest.update(images.detach().cpu().contiguous().numpy().tobytes())
            input_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
            img_ref = method.image_encoder_clip(images.type(method.dtype))
            img_ref = F.normalize(img_ref, dim=-1)
            original_b = branch_b = main_b = rep_b = 0.0
            main_prob_sum = rep_prob_sum = main_logit_sum = rep_logit_sum = None
            for k in range(0, samples, chunk):
                s = min(chunk, samples - k)
                out = model.forward_joint(images, s, use_posterior_mean=False)
                c = out["logits_main_stack"][..., :method.num_classes]
                r = out["logits_rep_stack"][..., :method.num_classes]
                original_b += float(original_mixture_nll(c, r, labels, alpha)) * s / samples
                logc = F.log_softmax(c.float(), dim=-1)
                logr = F.log_softmax(r.float(), dim=-1)
                idx = labels[None, :, None].expand(s, -1, 1)
                c_ce = -logc.gather(2, idx).squeeze(-1).mean()
                r_ce = -logr.gather(2, idx).squeeze(-1).mean()
                main_b += float(c_ce) * s / samples
                rep_b += float(r_ce) * s / samples
                branch_b += float(alpha * c_ce + (1.0 - alpha) * r_ce) * s / samples
                pc = logc.exp().sum(dim=0)
                pr = logr.exp().sum(dim=0)
                zc = c.float().sum(dim=0)
                zr = r.float().sum(dim=0)
                main_prob_sum = pc if main_prob_sum is None else main_prob_sum + pc
                rep_prob_sum = pr if rep_prob_sum is None else rep_prob_sum + pr
                main_logit_sum = zc if main_logit_sum is None else main_logit_sum + zc
                rep_logit_sum = zr if rep_logit_sum is None else rep_logit_sum + zr

            p_main = main_prob_sum / samples
            p_rep = rep_prob_sum / samples
            static_prob = alpha * p_main + (1.0 - alpha) * p_rep
            static_logit = alpha * main_logit_sum + (1.0 - alpha) * rep_logit_sum
            acc = {
                "main_accuracy": float((p_main.argmax(dim=-1) == labels).sum()),
                "rep_accuracy": float((p_rep.argmax(dim=-1) == labels).sum()),
                "static_prob_accuracy": float((static_prob.argmax(dim=-1) == labels).sum()),
                "static_logit_accuracy": float((static_logit.argmax(dim=-1) == labels).sum()),
            }

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
                             ("main_nll", main_b), ("rep_nll", rep_b),
                             ("det_nll", float(det_b)), ("alignment", float(align))):
                totals[key] += n * val
            for key, val in acc.items():
                totals[key] += val
            examples += n
            batches += 1

    if examples == 0:
        raise RuntimeError(f"No {split_name} diagnostic examples available")
    v = {k: value / examples for k, value in totals.items()}
    for key in ("main_accuracy", "rep_accuracy", "static_prob_accuracy", "static_logit_accuracy"):
        v[key] *= 100.0
    if abs(v["branch_nll"] - (alpha * v["main_nll"] + (1.0 - alpha) * v["rep_nll"])) > 5e-5:
        raise RuntimeError(f"{split_name}: Branch NLL decomposition sanity check failed")
    if v["branch_nll"] + 1e-5 < v["original_nll"]:
        raise RuntimeError(f"{split_name}: Jensen inequality unexpectedly violated")

    # KL is identical in all three reference objectives at this checkpoint.
    # Keep ORIGINAL per-training-batch normalization even for the test split.
    kl = model.kl_terms()
    rpost, tpost = model.image_encoder.bayes_proj_rep, model.text_posterior
    train_batches = max(1, len(trainer.dm.train_loader_x))
    kl_r = float(cfg.R_KL_WEIGHT) * float(kl["r_proj"]) / max(1, rpost.prior_mean.numel()) / train_batches
    kl_t = float(cfg.T_KL_WEIGHT) * float(kl["t_proj"]) / max(1, tpost.prior_mean.numel()) / train_batches
    v["kl_r"] = kl_r
    v["kl_t"] = kl_t
    for src, dst in (("original_nll", "original_reference_loss"),
                     ("branch_nll", "branch_reference_loss"),
                     ("det_nll", "deterministic_reference_loss")):
        v[dst] = v[src] + v["alignment"] + kl_r + kl_t
    v["mix_gap"] = v["branch_nll"] - v["original_nll"]
    v["det_gap"] = v["det_nll"] - v["branch_nll"]
    v["total_gap"] = v["det_nll"] - v["original_nll"]
    if abs(v["mix_gap"] + v["det_gap"] - v["total_gap"]) > 1e-5:
        raise RuntimeError(f"{split_name}: Gap addition identity failed")
    v["num_examples"] = examples
    v["num_batches"] = batches
    v["input_sha256"] = input_digest.hexdigest()
    v["reference_transform"] = reference_transform
    v["mc_reference_samples"] = samples
    v["det_var_scale"] = eta
    v["det_var_clamp"] = clamp
    v["kl_beta_reference"] = 1.0
    return v


def representation_snapshot(trainer, limit_images=128):
    """Effective prompt tokens and paired deterministic validation features.

    We record an image-tensor hash for EVERY row. summarize.py refuses to
    compute cross-method feature similarity if inputs differ byte-for-byte.
    Does not use test labels or alter evaluation transforms.
    """
    import torch
    import hashlib

    model = trainer.method.model
    model.eval()
    result = {}
    with torch.no_grad():
        text_tokens, visual_tokens = model.representation_learner()
        result["visual_prompts"] = torch.stack([v.detach().float().cpu() for v in visual_tokens])
        result["text_prompts"] = torch.stack([t.detach().float().cpu() for t in text_tokens])
        if limit_images == 0:
            return result

        loader = getattr(trainer.dm, "val_loader", None)
        if loader is None:
            raise RuntimeError("Validation loader unavailable for paired feature comparison; rerun --features 0")
        hashes, labels, main, rep = [], [], [], []
        remaining = limit_images
        for batch in loader:
            if remaining <= 0:
                break
            images = batch["img"][:remaining].to(trainer.method.device)
            y = batch["label"][:remaining].long().cpu()
            if images.shape[0] == 0:
                continue
            # Byte hashes make input pairing explicit, not an assumption about seeds.
            for x in images.detach().float().cpu():
                hashes.append(hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest())
            m = common_moments(model, images, trainer.method.cfg.BAYESRT_MMRL)
            main.append(m["image_features_main"].detach().float().cpu())
            rep.append(m["image_features_rep"].detach().float().cpu())
            labels.append(y)
            remaining -= images.shape[0]
        if not hashes:
            raise RuntimeError("Validation loader produced no image for feature comparison")
        result["image_hashes"] = hashes
        result["image_labels"] = torch.cat(labels)
        result["image_main_features"] = torch.cat(main)
        result["image_rep_features"] = torch.cat(rep)
        # Each class's text representation is shared across validation images.
        # Compute its current posterior-mean feature without sampling.
        result["text_features"] = m["text_features"].detach().float().cpu()
    return result


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
    p.add_argument("--features", type=int, default=128, help="Number of deterministic validation inputs for feature similarity; 0 disables")
    p.add_argument("--reference-batch-size", type=int, default=32, help="Diagnostic batch size; independent of training batch size")
    p.add_argument("--output-root", default="output_objective_fidelity_v3")
    o = p.parse_args()
    if o.samples < 1 or o.chunk < 1 or o.reference_batch_size < 1 or o.features < 0 or (o.limit_batches is not None and o.limit_batches < 1):
        p.error("samples and chunk must be >=1; features>=0; limit-batches>=1")
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
    import run as repo_run
    from core.config import setup_cfg
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    from datasets.fair_val_protocol import install_kplusval_datasetbase_patch
    from argparse import Namespace

    register_original_mc()  # needed also during checkpoint reconstruction
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
    loaders, reference_transform = deterministic_reference_loaders(trainer, o.reference_batch_size)
    metrics = {"diagnostic_schema": 5, "reference_input_protocol": "fixed_train_and_heldout_test",
               "reference_batch_size": o.reference_batch_size,
               "diagnostic_splits": ("train_fixed", "test")}
    for split_name, loader in loaders.items():
        result = diagnose(trainer, loader, o.samples, o.chunk, eta, clamp, split_name,
                          reference_transform, o.limit_batches)
        metrics.update({f"{split_name}_{key}": value for key, value in result.items()})
    metrics.update({k: mf[k] for k in ("mode", "profile", "dataset", "shot", "seed")})
    metrics.update({"r_trace": posterior["r_trace"], "t_trace": posterior["t_trace"],
                    "r_mean_norm": posterior["r_mean_norm"], "checkpoint_path": str(chosen),
                    "r_prior_mode": mf["r_prior_mode"],
                    "diagnostic_partial": o.limit_batches is not None})
    name = "fidelity_smoke.json" if o.limit_batches is not None else "fidelity.json"
    if o.limit_batches is None:
        import torch
        representation = representation_snapshot(trainer, limit_images=o.features)
        torch.save({**{k: posterior[k] for k in ("r_mean", "r_sigma", "t_mean", "t_sigma")},
                    **representation}, folder / "posterior_diagnostic.pt")
        metrics["feature_diagnostic_images"] = len(representation.get("image_hashes", []))
        # Paired input hashes are checked by summarize.py.
    (folder / name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
