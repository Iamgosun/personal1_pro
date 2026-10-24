"""E6 analysis primitives: read-only adapters to native DAVI and AVI-B.

No dependencies on historical E6 implementations or any trained artifacts.
No model architecture, training objective, optimizer or checkpoint changes.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cli import select_e3_fixed_batch
from .core import cached_draw, cached_encode, native_moments, restore, reference_loader
from methods.det_bayesrt_mmrl.losses import det_jensen_ce

GROUP_NAMES = {
    "r_mean": "image_encoder.bayes_proj_rep.posterior_mean",
    "r_rho": "image_encoder.bayes_proj_rep.posterior_rho",
    "t_rho": "text_posterior.posterior_rho",
}


def norm(v):
    return float(v.detach().double().norm())


def rms(v):
    return float(v.detach().double().square().mean().sqrt())


def cos(a, b, eps=1e-12):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape}, {b.shape}")
    if norm(a) < eps or norm(b) < eps:
        return None
    return float(F.cosine_similarity(a[None], b[None])[0])


def relchange(new, old):
    if old.shape != new.shape:
        raise ValueError("Parameter shapes differ")
    denominator = norm(old)
    return float((new.detach().double() - old.detach().double()).norm() / denominator) if denominator > 1e-12 else None


def row_cos(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Expected paired 2-D tensors")
    valid = (a.norm(dim=-1) > 1e-8) & (b.norm(dim=-1) > 1e-8)
    if not bool(valid.all()):
        return None
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def normalize_project(hidden, matrix):
    if hidden.shape[-1] != matrix.shape[0]:
        raise ValueError("Projection dimensions disagree")
    z = hidden.float() @ matrix.float()
    lengths = z.norm(dim=-1)
    if bool((lengths <= 1e-8).any()):
        return None
    return F.normalize(z, dim=-1)


def cross_project(h_a, h_d, m_a, m_d):
    """A=AVI-B, D=DAVI. Degenerate mean projection is undefined (not zero)."""
    return {"AA": normalize_project(h_a, m_a),
            "AD": normalize_project(h_a, m_d),
            "DA": normalize_project(h_d, m_a),
            "DD": normalize_project(h_d, m_d)}


def directional_attribution(features):
    """Per-image symmetric signed contributions to DD-AA in the common CLIP space.

    By construction Head + Hidden == DD-AA. Contributions are not causal shares
    and can be negative or greater than 100 percent.
    """
    if any(features[k] is None for k in ("AA", "AD", "DA", "DD")):
        return {"head_pct": None, "hidden_pct": None, "n_valid": 0}
    aa, ad, da, dd = (features[k].double() for k in ("AA", "AD", "DA", "DD"))
    head = ((ad - aa) + (dd - da)) / 2
    hidden = ((da - aa) + (dd - ad)) / 2
    change = dd - aa
    if not torch.allclose(head + hidden, change, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Symmetric attribution identity violated")
    denom = change.square().sum(-1)
    valid = denom > 1e-12
    if not bool(valid.any()):
        return {"head_pct": None, "hidden_pct": None, "n_valid": 0}
    hp = (head[valid] * change[valid]).sum(-1) / denom[valid]
    vp = (hidden[valid] * change[valid]).sum(-1) / denom[valid]
    return {"head_pct": 100 * float(hp.mean()), "hidden_pct": 100 * float(vp.mean()),
            "n_valid": int(valid.sum())}


def posterior_snapshot(model):
    snap = {}
    for tag, q in (("r", model.image_encoder.bayes_proj_rep), ("t", model.text_posterior)):
        if q is None:
            raise RuntimeError(f"Missing {tag} posterior")
        sigma = q.posterior_sigma().detach().float().cpu()
        full_sigma = q.expanded_posterior_sigma().detach().float().cpu()
        snap[f"{tag}_sigma"] = sigma
        snap[f"{tag}_sigma_expanded"] = full_sigma
        snap[f"{tag}_sigma_trace"] = float(full_sigma.square().sum())
        snap[f"{tag}_sigma_mean"] = float(sigma.mean())
        snap[f"{tag}_sigma_q10"] = float(torch.quantile(sigma.flatten(), .10))
        snap[f"{tag}_sigma_q50"] = float(torch.quantile(sigma.flatten(), .50))
        snap[f"{tag}_sigma_q90"] = float(torch.quantile(sigma.flatten(), .90))
    m = model.image_encoder.bayes_proj_rep.posterior_mean.detach().float().cpu()
    snap["r_mean"] = m
    snap["r_mean_fro"] = norm(m)
    snap["r_mean_rms"] = rms(m)
    snap["r_mean_abs_mean"] = float(m.abs().mean())
    snap["r_mean_prior_distance"] = norm(m - model.image_encoder.bayes_proj_rep.prior_mean.detach().float().cpu())
    return snap


def _previous_state(run_dir, current_epoch):
    run_dir = Path(run_dir)
    candidates = list((run_dir / "trajectory" / "refactor_model").glob("model.pth.tar-*"))
    candidates += list((run_dir / "refactor_model").glob("model.pth.tar-*"))
    valid = []
    for p in candidates:
        ep = p.name.rsplit("-", 1)[-1]
        if ep.isdecimal() and int(ep) < current_epoch:
            valid.append((int(ep), p))
    if not valid:
        return None, None
    ep, path = max(valid, key=lambda z: z[0])
    # These files are locally created trusted PyTorch training checkpoints.
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return ep, state


def parameter_dynamics(run_dir, epoch, model):
    cur = posterior_snapshot(model)
    result = {k: v for k, v in cur.items() if isinstance(v, float)}
    prev_ep, prev = _previous_state(run_dir, epoch)
    result["previous_epoch"] = prev_ep
    result["delta_epochs"] = (epoch - prev_ep) if prev_ep is not None else None
    for name, key in GROUP_NAMES.items():
        q = model.text_posterior if name == "t_rho" else model.image_encoder.bayes_proj_rep
        now = q.posterior_mean if name == "r_mean" else q.posterior_rho
        now = now.detach().float().cpu()
        old = prev[key].float() if prev is not None else None
        result[f"{name}_delta_norm"] = norm(now - old) if old is not None else None
        result[f"{name}_delta_relative"] = relchange(now, old) if old is not None else None
        if name == "r_mean":
            result["r_mean_previous_cos"] = cos(now, old) if old is not None else None
        else:
            sigma_now = F.softplus(now) + float(q.min_sigma)
            sigma_old = F.softplus(old) + float(q.min_sigma) if old is not None else None
            result[f"{name}_sigma_delta_relative"] = relchange(sigma_now, sigma_old) if old is not None else None
    result["note"] = "Adjacent saved-checkpoint interval, NOT a single optimizer step"
    return result


def _unload(trainer):
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_representation(run_dir, gpu, selection, epoch, images, batch_size, input_seed):
    trainer, meta, ck = restore(run_dir, gpu=gpu, selection=selection, epoch=epoch)
    if meta.get("study") != "native" or meta.get("status") != "complete":
        raise RuntimeError("E6 requires completed native training")
    model = trainer.method.model
    model.eval()
    batch, info = select_e3_fixed_batch(trainer, images, strategy="stratified", selection_seed=input_seed)
    h, native = [], []
    with torch.no_grad():
        for start in range(0, len(batch["label"]), batch_size):
            x = batch["img"][start:start+batch_size].to(trainer.method.device)
            _, tokens_v = model.representation_learner()
            _, r_hidden = model.image_encoder.encode_hidden(x.type(model.dtype), tokens_v)
            h.append(r_hidden.detach().float().cpu())
            native.append(F.normalize((r_hidden.float() @ model.image_encoder.bayes_proj_rep.posterior_mean.float()), dim=-1).cpu())
    snap = posterior_snapshot(model)
    classes = tuple(trainer.dm.dataset.classnames)
    class_digest = hashlib.sha256(json.dumps(classes, ensure_ascii=False).encode()).hexdigest()
    out = {"dataset": meta["dataset"], "seed": meta["seed"], "shot": meta["shot"],
           "method": meta["method"], "epoch": ck["epoch"],
           "checkpoint_sha256": ck["checkpoint_sha256"],
           "input_sha256": info["input_sha256"], "indices_sha256": info["selected_indices_sha256"],
           "classnames_sha256": class_digest, "labels": batch["label"].long().cpu(), "hidden": torch.cat(h), "native": torch.cat(native),
           "posterior": snap, "dynamics": parameter_dynamics(run_dir, int(ck["epoch"]), model)}
    check = normalize_project(out["hidden"], snap["r_mean"])
    if check is not None and not torch.allclose(check, out["native"], atol=1e-4, rtol=1e-4):
        raise RuntimeError("Native R projection vs diagnostic parity mismatch")
    _unload(trainer)
    return out


def compare_representation(a, d, posterior_draws=0, sample_seed=99173):
    for key in ("dataset", "seed", "shot", "epoch", "input_sha256", "indices_sha256", "classnames_sha256"):
        if a[key] != d[key]:
            raise RuntimeError(f"Unpaired AVI-B/DAVI: {key}")
    if not torch.equal(a["labels"], d["labels"]):
        raise RuntimeError("Paired image labels differ")
    h_a, h_d = a["hidden"], d["hidden"]
    p_a, p_d = a["posterior"], d["posterior"]
    f = cross_project(h_a, h_d, p_a["r_mean"], p_d["r_mean"])
    result = {"dataset": a["dataset"], "seed": a["seed"], "shot": a["shot"], "epoch": a["epoch"],
              "n_images": len(a["labels"]), "input_sha256": a["input_sha256"],
              "branch_checkpoint_sha256": a["checkpoint_sha256"],
              "davi_checkpoint_sha256": d["checkpoint_sha256"],
              "r_hidden_cos": row_cos(h_a, h_d),
              "r_mean_cos": cos(p_a["r_mean"], p_d["r_mean"]),
              "r_mean_rel_l2": relchange(p_d["r_mean"], p_a["r_mean"]),
              "r_mean_delta_fro": norm(p_d["r_mean"] - p_a["r_mean"]),
              "native_rep_cos": row_cos(f["AA"], f["DD"]) if f["AA"] is not None and f["DD"] is not None else None,
              "head_change_cos": None, "hidden_change_cos": None}
    if all(f[k] is not None for k in ("AA", "AD", "DA", "DD")):
        result["head_change_cos"] = (row_cos(f["AA"], f["AD"]) + row_cos(f["DA"], f["DD"])) / 2
        result["hidden_change_cos"] = (row_cos(f["AA"], f["DA"]) + row_cos(f["AD"], f["DD"])) / 2
    result.update(directional_attribution(f))
    for tag, snap in (("A", p_a), ("D", p_d)):
        for key in ("r_mean_fro", "r_mean_rms", "r_mean_prior_distance", "r_sigma_trace", "t_sigma_trace",
                    "r_sigma_q10", "r_sigma_q50", "r_sigma_q90", "t_sigma_q10", "t_sigma_q50", "t_sigma_q90"):
            result[f"{key}_{tag}"] = snap[key]
    for tag, src in (("A", a), ("D", d)):
        for key, val in src["dynamics"].items():
            if isinstance(val, (int, float)) or val is None:
                result[f"dynamics_{key}_{tag}"] = val
    if posterior_draws:
        result.update(covariance_projection_diagnostic(a, d, posterior_draws, sample_seed))
    return result, f


def covariance_projection_diagnostic(a, d, draws, seed):
    """Same standard-normal noise, fixed input and mean; swap only R posterior sigma."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    sa = a["posterior"]["r_sigma_expanded"]
    sd = d["posterior"]["r_sigma_expanded"]
    output = {}
    for tag, src in (("A", a), ("D", d)):
        h = src["hidden"]; mu = src["posterior"]["r_mean"]
        if normalize_project(h, mu) is None:
            output[f"cov_swap_{tag}_cos"] = None
            continue
        sims, deviations = [], {"A": [], "D": []}
        reference = normalize_project(h, mu)
        for _ in range(draws):
            eps = torch.randn(mu.shape, generator=rng)
            aa = normalize_project(h, mu + sa * eps)
            dd = normalize_project(h, mu + sd * eps)
            if aa is None or dd is None:
                raise RuntimeError("Random R projected sample norm is zero")
            sims.append(row_cos(aa, dd))
            deviations["A"].append(float((aa - reference).square().sum(-1).mean()))
            deviations["D"].append(float((dd - reference).square().sum(-1).mean()))
        output[f"cov_swap_{tag}_cos"] = float(np.mean(sims))
        output[f"cov_swap_{tag}_deviation_A"] = float(np.mean(deviations["A"]))
        output[f"cov_swap_{tag}_deviation_D"] = float(np.mean(deviations["D"]))
    return output


def groups_for(model, include_phi=True):
    p = dict(model.named_parameters())
    groups = {name: [p[key]] for name, key in GROUP_NAMES.items()}
    for name, ps in groups.items():
        if not all(w.requires_grad for w in ps):
            raise RuntimeError(f"Missing trainable group {name}")
    if include_phi:
        shared = [q for n, q in model.named_parameters() if n.startswith("representation_learner.") and q.requires_grad]
        if not shared:
            raise RuntimeError("Missing trainable shared representation parameters")
        groups["phi"] = shared
    return groups


def _gradient_flat(loss, params, retain_graph=False):
    gg = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    z = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().reshape(-1)
                   for g, p in zip(gg, params)])
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("Non-finite gradient")
    return z


def grad_groups(loss, groups, retain_graph=False):
    params = [p for sub in groups.values() for p in sub]
    # One autograd traversal per draw instead of one traversal per parameter group.
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    out, pos = {}, 0
    for name, gs in groups.items():
        sub = grads[pos:pos+len(gs)]
        pos += len(gs)
        vec = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().cpu().flatten()
                         for g, p in zip(sub, gs)])
        if not bool(torch.isfinite(vec).all()):
            raise RuntimeError(f"Non-finite gradient: {name}")
        out[name] = vec
    return out


def draw_statistics(samples):
    """RMS/dispersion of an MC gradient estimator; deterministic has no draw variance."""
    if not samples:
        raise ValueError("Empty list of sample gradients")
    g = torch.stack(samples).double()
    mean = g.mean(0)
    lengths = g.norm(dim=-1)
    valid = lengths > 1e-12
    out = {"draws": int(g.shape[0]), "single_rms": float(g.square().mean().sqrt()),
           "mean_rms": rms(mean), "mean_norm": norm(mean),
           "pairwise_cos": None, "cancellation_retention": None, "relative_variance": None}
    if len(samples) >= 2:
        var = float((g - mean).square().sum(-1).mean() * len(samples)/(len(samples)-1))
        out["relative_variance"] = var / float(mean.square().sum()) if norm(mean) > 1e-12 else None
        denom = float(lengths.mean())
        out["cancellation_retention"] = norm(mean) / denom if denom > 1e-12 else None
        if int(valid.sum()) >= 2:
            unit = g[valid] / lengths[valid, None]
            matrix = unit @ unit.T
            m = matrix.shape[0]
            out["pairwise_cos"] = float((matrix.sum() - m) / (m * (m-1)))
    return out


def _davi_objective(model, cfg, x, y, n_classes):
    mom = native_moments(model, x, cfg)
    alpha = float(cfg.ALPHA)
    scale = float(getattr(cfg, "DET_VAR_SCALE", 1.0))
    clamp = float(getattr(cfg, "DET_VAR_CLAMP", 50.0))
    total = alpha * det_jensen_ce(mom["mu_main"][..., :n_classes], mom["var_main"][..., :n_classes], y,
                                  var_scale=scale, var_clamp=clamp)
    total = total + (1-alpha) * det_jensen_ce(mom["mu_rep"][..., :n_classes],
                        mom["var_rep"][..., :n_classes], y, var_scale=scale, var_clamp=clamp)
    pure = alpha * F.cross_entropy(mom["mu_main"][..., :n_classes].float(), y)
    pure = pure + (1-alpha) * F.cross_entropy(mom["mu_rep"][..., :n_classes].float(), y)
    cf = {name: float((mom[name].detach() >= clamp).float().mean()) if clamp>0 else 0.
          for name in ("var_main", "var_rep")}
    return total, pure, cf


def _mc_objective(cache, y, alpha, n_samples):
    m, r = cached_draw(cache, n_samples)
    ce_m = F.cross_entropy(m.float().flatten(0,1), y.repeat(n_samples))
    ce_r = F.cross_entropy(r.float().flatten(0,1), y.repeat(n_samples))
    return alpha * ce_m + (1-alpha) * ce_r


def _kl_gradients(model, method, groups, train_batches):
    raw = model.kl_terms()
    beta = float(getattr(method, "kl_beta", 1.0))
    n = max(1, train_batches)
    cfg = method.cfg.BAYESRT_MMRL
    r = beta * float(cfg.R_KL_WEIGHT) * raw["r_proj"] / model.image_encoder.bayes_proj_rep.prior_mean.numel() / n
    t = beta * float(cfg.T_KL_WEIGHT) * raw["t_proj"] / model.text_posterior.prior_mean.numel() / n
    grads = grad_groups(r + t, groups)
    return grads, {"kl_r": float(r.detach()), "kl_t": float(t.detach()), "kl_beta": beta, "kl_normalizer_batches": n}


def _align_gradients(model, method, x, groups):
    cache = cached_encode(model, x, method.num_classes)
    with torch.no_grad():
        original = F.normalize(method.image_encoder_clip(x.type(method.dtype)), dim=-1)
    _, text = model.text_sample_features(cache["eot"], 1, use_mean=True)
    text_ref = method.text_features_clip[:method.num_classes].to(text.device)
    loss = float(method.cfg.BAYESRT_MMRL.REG_WEIGHT) * (
        (1 - F.cosine_similarity(cache["main"], original, dim=-1).mean()) +
        (1 - F.cosine_similarity(text[:method.num_classes], text_ref, dim=-1).mean()))
    return grad_groups(loss, groups)


def _fixed_grad_batch(trainer, batch_size, input_seed):
    batch, info = select_e3_fixed_batch(trainer, batch_size, strategy="stratified", selection_seed=input_seed)
    return batch["img"].to(trainer.method.device), batch["label"].long().to(trainer.method.device), info


def _setup(run_dir, gpu, selection, epoch, batch_size, input_seed, include_phi):
    trainer, meta, ck = restore(run_dir, gpu=gpu, selection=selection, epoch=epoch)
    if meta.get("study") != "native" or meta.get("status") != "complete":
        raise RuntimeError("E6 requires a completed native run")
    method, model = trainer.method, trainer.method.model
    model.eval()
    # Match actual native per-batch scaling and KL warmup at the checkpoint's epoch.
    n_batches = len(trainer.dm.train_loader_x)
    warmup = int(getattr(method, "kl_warmup_epochs", 0))
    epoch0 = max(0, int(ck["epoch"]) - 1)
    total_epochs = int(trainer.max_epoch)
    if warmup <= 0:
        beta = 1.0
    elif epoch0 < warmup:
        beta = 0.0
    else:
        beta = min(1., float(epoch0-warmup)/max(1, total_epochs-warmup-1))
    method.set_kl_beta(beta)
    method.set_kl_normalizer(n_batches)
    x, y, info = _fixed_grad_batch(trainer, batch_size, input_seed)
    return trainer, meta, ck, x, y, info, groups_for(model, include_phi), n_batches


def native_gradients(run_dir, gpu, selection, epoch, batch_size, input_seed, draws, seed, include_phi):
    trainer, meta, ck, x, y, info, groups, n_batches = _setup(
        run_dir, gpu, selection, epoch, batch_size, input_seed, include_phi)
    method, model = trainer.method, trainer.method.model
    grads_kl, kl_meta = _kl_gradients(model, method, groups, n_batches)
    grads_align = _align_gradients(model, method, x, groups)
    alpha = float(method.cfg.BAYESRT_MMRL.ALPHA)
    sample = {key: [] for key in groups}
    devices = [torch.cuda.current_device()] if x.is_cuda else []
    with torch.enable_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if meta["method"] == "branch":
            if draws < int(method.n_mc_train):
                raise ValueError("E6 --grad-draws must be >= native N_MC_TRAIN")
            cache = cached_encode(model, x, method.num_classes)
            for j in range(draws):
                cls = _mc_objective(cache, y, alpha, 1)
                gg = grad_groups(cls, groups, retain_graph=True)
                for name in groups:
                    sample[name].append(gg[name])
            del cache
        elif meta["method"] == "davi":
            cfg = method.cfg.BAYESRT_MMRL
            cls, _, cf = _davi_objective(model, cfg, x, y, method.num_classes)
            gg = grad_groups(cls, groups)
            for name in groups:
                sample[name].append(gg[name])
        else:
            raise RuntimeError("E6 only supports AVI-B and DAVI")
    result = {"schema": 3, "dataset": meta["dataset"], "method": meta["method"],
              "seed": meta["seed"], "shot": meta["shot"], "epoch": ck["epoch"],
              "checkpoint_sha256": ck["checkpoint_sha256"], "input_sha256": info["input_sha256"],
              "draws": len(next(iter(sample.values()))), "native_mc_budget": int(method.n_mc_train) if meta["method"] == "branch" else None,
              "kl": kl_meta, "clamp": cf if meta["method"] == "davi" else None,
              "dynamics": parameter_dynamics(run_dir, int(ck["epoch"]), model), "groups": {}}
    for name, vectors in sample.items():
        stats = draw_statistics(vectors)
        cls = torch.stack(vectors).mean(0)
        align = grads_align[name]; kl = grads_kl[name]
        total = cls + align + kl
        stats.update({"cls_grad_rms": rms(cls), "alignment_grad_rms": rms(align),
                      "kl_grad_rms": rms(kl), "total_grad_rms": rms(total),
                      "cls_align_cos": cos(cls, align), "cls_kl_cos": cos(cls, kl),
                      "cls_total_cos": cos(cls, total)})
        if meta["method"] == "branch":
            native = torch.stack(vectors[:int(method.n_mc_train)]).mean(0)
            stats["native_budget_cls_grad_rms"] = rms(native)
            stats["native_budget_total_grad_rms"] = rms(native + align + kl)
        result["groups"][name] = stats
    _unload(trainer)
    return result


def same_checkpoint_objectives(run_dir, gpu, selection, epoch, batch_size, input_seed, mc_draws, mc_repeats,
                               seed, include_phi, davi_scale=1., davi_clamp=50.):
    """Direct loss-operator comparison at identical theta and batch, classification only."""
    trainer, meta, ck, x, y, info, groups, _ = _setup(run_dir, gpu, selection, epoch, batch_size, input_seed, include_phi)
    method, model = trainer.method, trainer.method.model
    cfg = method.cfg.BAYESRT_MMRL
    if meta["method"] == "branch":
        with torch.no_grad():
            cache = cached_encode(model, x, method.num_classes)
            z = cache["rep"].float() @ model.image_encoder.bayes_proj_rep.posterior_mean.float()
            projected_norm_min = float(z.norm(dim=-1).min())
        if projected_norm_min < 1e-5:
            _unload(trainer)
            return {"schema": 3, "dataset": meta["dataset"], "method": meta["method"], "seed": meta["seed"],
                    "epoch": ck["epoch"], "status": "not_defined", "reason": "near_zero_R_projection_mean",
                    "min_projection_norm": projected_norm_min}
    # Run counterfactual DAVI with DAVI's actual fixed 1/50 controls, not AVI-B YAML fallbacks.
    if meta["method"] == "davi":
        davi_scale = float(cfg.DET_VAR_SCALE); davi_clamp = float(cfg.DET_VAR_CLAMP)
    proxy = type("DaviControls", (), {"DET_VAR_SCALE": davi_scale, "DET_VAR_CLAMP": davi_clamp,
                                         "ALPHA": float(cfg.ALPHA)})()
    # Differentiable native DAVI moments. AVI-B uses the native_moments adapter for a
    # diagnostic counterfactual, not a modified model or checkpoint.
    with torch.enable_grad():
        det, pure, cf = _davi_objective(model, proxy, x, y, method.num_classes)
        gd = grad_groups(det, groups, retain_graph=True)
        gp = grad_groups(pure, groups)
    devices = [torch.cuda.current_device()] if x.is_cuda else []
    mc = []
    with torch.enable_grad(), torch.random.fork_rng(devices=devices):
        for repeat in range(mc_repeats):
            torch.manual_seed(seed + 100003*repeat)
            acc = {name: None for name in groups}
            cache = cached_encode(model, x, method.num_classes)
            for j in range(mc_draws):
                loss = _mc_objective(cache, y, float(cfg.ALPHA), 1)
                current = grad_groups(loss, groups, retain_graph=True)
                for name in groups:
                    acc[name] = current[name] if acc[name] is None else acc[name] + current[name]
            mc.append({name: acc[name]/mc_draws for name in groups})
            del cache
    result = {"schema": 3, "dataset": meta["dataset"], "method": meta["method"], "seed": meta["seed"],
              "shot": meta["shot"], "epoch": ck["epoch"], "checkpoint_sha256": ck["checkpoint_sha256"],
              "input_sha256": info["input_sha256"], "status": "ok", "mc_draws": mc_draws,
              "mc_repeats": mc_repeats, "davi_scale": davi_scale, "davi_clamp": davi_clamp,
              "clamp": cf, "groups": {}, "note": "Same checkpoint; classification gradients only; local not causal."}
    for name in groups:
        gm = torch.stack([t[name] for t in mc]).mean(0)
        gdg = gd[name]; gpure = gp[name]
        result["groups"][name] = {
            "mc_grad_rms": rms(gm), "davi_grad_rms": rms(gdg),
            "mc_davi_cos": cos(gm, gdg), "davi_to_mc_relative_l2": relchange(gdg, gm),
            "mc_repeat_cos": cos(mc[0][name], mc[1][name]) if mc_repeats >= 2 else None,
            "mc_repeat_relative_l2": relchange(mc[0][name], mc[1][name]) if mc_repeats >= 2 else None,
            "mean_only_grad_rms": rms(gpure),
            "variance_correction_grad_rms": rms(gdg - gpure),
            "variance_correction_cos": cos(gdg - gpure, gdg),
        }
    _unload(trainer)
    return result


def calibration_metrics(probs, labels, bins=15):
    probs = probs.float().cpu(); labels = labels.long().cpu()
    if probs.ndim != 2 or labels.shape != (len(probs),):
        raise ValueError("Probabilities and labels must align")
    confidence, pred = probs.max(-1)
    correct = (pred == labels).float()
    def expected_gap(blocks):
        total = 0.
        for ids in blocks:
            if ids.numel():
                total += float((len(ids)/len(labels)) * (correct[ids].mean() - confidence[ids].mean()).abs())
        return total
    width_blocks = []
    for i in range(bins):
        left, right = i/bins, (i+1)/bins
        ids = torch.where((confidence >= left) & ((confidence < right) if i < bins-1 else (confidence <= right)))[0]
        width_blocks.append(ids)
    count_blocks = list(torch.tensor_split(confidence.argsort(), bins))
    picked = probs.gather(1, labels[:, None]).clamp_min(1e-12)
    incorrect = correct < .5
    return {"acc": float(correct.mean()), "nll": float(-picked.log().mean()),
            "ece_15_equal_width": expected_gap(width_blocks), "aece_15_equal_count": expected_gap(count_blocks),
            "confidence_mean": float(confidence.mean()),
            "confidence_correct": float(confidence[~incorrect].mean()) if bool((~incorrect).any()) else None,
            "confidence_wrong": float(confidence[incorrect].mean()) if bool(incorrect.any()) else None,
            "n_test": len(labels)}


def _rho_for_scaled_sigma(sigma, min_sigma, gamma):
    target = sigma.float() * float(gamma)
    value = (target - float(min_sigma)).clamp_min(1e-12)
    return value + torch.log(-torch.expm1(-value))


def prediction_interventions(run_dir, gpu, selection, epoch, batch_size, samples, chunk, repeats,
                             seed, sigmas=(.5, 1., 2.), mean_override=None, mean_source_sha=None):
    trainer, meta, ck = restore(run_dir, gpu=gpu, selection=selection, epoch=epoch)
    method, model = trainer.method, trainer.method.model
    model.eval()
    cfg = method.cfg.BAYESRT_MMRL
    if str(cfg.EVAL_FUSION_VARIANT).lower() != "static_logit" or bool(cfg.EVAL_USE_POSTERIOR_MEAN):
        raise RuntimeError("Prediction intervention requires native sampled static_logit")
    q_r, q_t = model.image_encoder.bayes_proj_rep, model.text_posterior
    old_rho_r = q_r.posterior_rho.detach().clone()
    old_rho_t = q_t.posterior_rho.detach().clone()
    old_mu = q_r.posterior_mean.detach().clone()
    base_sigmas = (q_r.posterior_sigma().detach().clone(), q_t.posterior_sigma().detach().clone())
    conditions = [("native", 1., 1.)]
    for gamma in sigmas:
        if gamma != 1.:
            conditions += [("r_sigma_scaled", gamma, 1.), ("t_sigma_scaled", 1., gamma)]
    if mean_override is not None:
        conditions += [("r_mean_swapped", 1., 1.)]
    loader = reference_loader(trainer, "test", batch_size=batch_size)
    devices = [torch.cuda.current_device()] if str(method.device).startswith("cuda") else []
    records = []
    try:
        for condition, gamma_r, gamma_t in conditions:
            with torch.no_grad():
                q_r.posterior_rho.copy_(_rho_for_scaled_sigma(base_sigmas[0], q_r.min_sigma, gamma_r).to(q_r.posterior_rho.device))
                q_t.posterior_rho.copy_(_rho_for_scaled_sigma(base_sigmas[1], q_t.min_sigma, gamma_t).to(q_t.posterior_rho.device))
                q_r.posterior_mean.copy_((mean_override if condition=="r_mean_swapped" else old_mu).to(q_r.posterior_mean.device))
            for repeat in range(repeats):
                probs, labels = [], []
                with torch.no_grad(), torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + 100003*repeat)
                    for batch in loader:
                        x = batch["img"].to(method.device)
                        y = batch["label"].long().cpu()
                        cache = cached_encode(model, x, method.num_classes)
                        logits_sum = None
                        for first in range(0, samples, chunk):
                            n = min(chunk, samples-first)
                            main, rep = cached_draw(cache, n)
                            summed = float(cfg.ALPHA) * main.float().sum(0) + (1-float(cfg.ALPHA))*rep.float().sum(0)
                            logits_sum = summed if logits_sum is None else logits_sum + summed
                        probs.append(F.softmax(logits_sum / samples, -1).cpu())
                        labels.append(y)
                met = calibration_metrics(torch.cat(probs), torch.cat(labels))
                records.append({"condition": condition, "gamma_r": gamma_r, "gamma_t": gamma_t,
                                "repeat": repeat, **met})
    finally:
        with torch.no_grad():
            q_r.posterior_rho.copy_(old_rho_r)
            q_t.posterior_rho.copy_(old_rho_t)
            q_r.posterior_mean.copy_(old_mu)
        _unload(trainer)
    return {"schema": 3, "dataset": meta["dataset"], "method": meta["method"], "shot": meta["shot"],
            "seed": meta["seed"], "epoch": ck["epoch"], "checkpoint_sha256": ck["checkpoint_sha256"],
            "mean_source_sha256": mean_source_sha, "samples": samples, "repeats": repeats,
            "split": "test", "fusion": "static_logit", "results": records,
            "note": "Read-only, fixed-model interventions; inference on 3 seeds, not MC repeats."}
