"""Cache deterministic deep-prompt CLIP encodings; resample ONLY posterior heads.

The reference estimator still uses the original q(P_r),q(P_t), nonlinear
normalization and paired draws. Model weights, prompt states and model classes
are never modified. Source: BayesRTMMRLModel.forward_joint/forward_joint_moments.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_once(model, images, num_classes):
    text_tokens, visual_tokens = model.representation_learner()
    eot = model.text_encoder.forward_hidden(model.prompt_embeddings,
        model.tokenized_prompts, text_tokens)
    cls, rep = model.image_encoder.encode_hidden(images.type(model.dtype), visual_tokens)
    image_main=F.normalize(cls @ model.image_encoder.visual.proj,dim=-1)
    return {"model":model,"eot":eot,"cls":cls,"rep":rep,
            "main":image_main,"num_classes":num_classes}


@torch.no_grad()
def cached_joint_logits(cache,samples):
    model=cache["model"]
    text,_=model.text_sample_features(cache["eot"],samples,use_mean=False)
    image_rep=model.image_encoder.rep_samples(cache["rep"],samples,use_mean=False)
    image_rep=F.normalize(image_rep,dim=-1)
    main=cache["main"]
    text=text.type(main.dtype)
    main_logits=100.0*torch.einsum('bd,scd->sbc',main,text)
    rep_logits=100.0*torch.einsum('sbd,scd->sbc',image_rep,text)
    classes=cache["num_classes"]
    return main_logits[...,:classes],rep_logits[...,:classes]


@torch.no_grad()
def moments_from_cache(cache, moment_eps=1e-6, tau=100.0):
    from methods.det_bayesrt_mmrl.modules import (
        _factorized_project_moments,_normalize_moments,
        _norm_directional_var,_rep_logit_var)
    model=cache["model"]
    eot,rep=cache["eot"],cache["rep"]
    if model.text_posterior is not None:
        tm,tv=_factorized_project_moments(model.text_posterior,eot.float())
    else:
        tm=eot.float()@model.text_projection.to(eot.device).float()
        tv=torch.zeros_like(tm)
    if model.image_encoder.bayes_proj_rep is not None:
        rm,rv=_factorized_project_moments(model.image_encoder.bayes_proj_rep,rep.float())
    else:
        rm=rep.float()@model.image_encoder.visual.proj_rep.to(rep.device).float()
        rv=torch.zeros_like(rm)
    t,rt,tv=_normalize_moments(tm,tv,eps=moment_eps)
    r,rr,rv=_normalize_moments(rm,rv,eps=moment_eps)
    main=cache["main"].float(); t=t.float();r=r.float()
    mu_m=tau*(main@t.T)
    var_m=tau*tau*_norm_directional_var(main,t,tv.float(),rt.float(),eps=moment_eps)
    mu_r=tau*(r@t.T)
    var_r=_rep_logit_var(r,rv.float(),rr.float(),t,tv.float(),rt.float(),tau=tau,eps=moment_eps)
    classes=cache["num_classes"]
    # SNR: norm E[a] / sqrt(trace Cov[a]); meaningful for delta-method regime.
    def snr(mean,var):
        return (mean.float().norm(dim=-1) /
                var.float().clamp_min(0).sum(dim=-1).sqrt().clamp_min(1e-12))
    return {"mu_main":mu_m[:,:classes],"var_main":var_m[:,:classes],
        "mu_rep":mu_r[:,:classes],"var_rep":var_r[:,:classes],
        "visual_pre_snr":float(snr(rm,rv).mean()),
        "text_pre_snr":float(snr(tm,tv).mean()),
        "visual_pre_mean_norm_min":float(rm.float().norm(dim=-1).min()),
        "text_pre_mean_norm_min":float(tm.float().norm(dim=-1).min())}


@torch.no_grad()
def validate_native_parity(model,images,cache,mom,common_moments,cfg,
                           n_draws=4,rtol=3e-3,atol=3e-3):
    """Compare cached sampler and moments to native paths on same RNG state.

    For different CUDA kernels half precision may be slightly non-bitwise.
    Call before the expensive MC reference evaluation; fail closed.
    """
    device=images.device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(999801)
        native=model.forward_joint(images,n_draws,use_posterior_mean=False)
        native_m=native['logits_main_stack'][...,:cache['num_classes']]
        native_r=native['logits_rep_stack'][...,:cache['num_classes']]
        torch.manual_seed(999801)
        got_m,got_r=cached_joint_logits(cache,n_draws)
        for label,x,y in (('main logits',got_m,native_m),('rep logits',got_r,native_r)):
            if not torch.allclose(x.float(),y.float(),rtol=rtol,atol=atol):
                raise RuntimeError(f'Cached vs native {label} disagree; max_abs={(x.float()-y.float()).abs().max().item():.5g}')
    ref=common_moments(model,images,cfg)
    for key in ('mu_main','var_main','mu_rep','var_rep'):
        target=ref[key][...,:cache['num_classes']]
        got=mom[key]
        if not torch.allclose(got.float(),target.float(),rtol=rtol,atol=atol):
            raise RuntimeError(f'Cached vs native {key} disagree; max_abs={(got-target).abs().max().item():.5g}')
    return {'checked_draws':n_draws,'rtol':rtol,'atol':atol,'max_abs_main':float((got_m.float()-native_m.float()).abs().max()),
            'max_abs_rep':float((got_r.float()-native_r.float()).abs().max())}
