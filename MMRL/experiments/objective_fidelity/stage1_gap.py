"""Streaming, per-image Stage-1 objective fidelity diagnostics (native DAVI).

All six signed terms telescope *per image* on the same frozen model and
posterior draws. Only the first two components have a guaranteed nonnegative
sign. Empirical J uses a finite-sample log-MGF and is not an exact q bound.
"""
from __future__ import annotations

import math
from typing import Iterable
import torch
import torch.nn.functional as F

SIX_TERMS = ("mix_gap", "softmax_jensen_gap", "gaussian_error",
             "propagation_error", "scale_error", "clamp_error")
LOSS_KEYS = ("original_nll", "branch_nll", "empirical_jensen_nll",
             "gaussian_moment_nll", "propagated_unscaled_nll",
             "propagated_scaled_nll", "davi_nll")


def surrogate_per_image(mu: torch.Tensor, var: torch.Tensor,
                        labels: torch.Tensor, eta: float = 1.0,
                        clamp: float = 0.0) -> torch.Tensor:
    """Same mathematical expression as the native det_jensen_ce, at FP64."""
    if eta < 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be finite and >= 0")
    if mu.ndim != 2 or mu.shape != var.shape or labels.numel() != mu.shape[0]:
        raise ValueError("mu and var must have matching [B,C] shape")
    v = var.double().clamp_min(0)
    if clamp is not None and float(clamp) > 0:
        v = v.clamp_max(float(clamp))
    m = mu.double()
    return torch.logsumexp(m + 0.5 * eta * v, -1) - m.gather(1, labels.long().view(-1, 1)).squeeze(1)


def surrogate(mu, var, labels, eta=1., clamp=0.):
    return surrogate_per_image(mu, var, labels, eta, clamp).mean()


class GapAccumulator:
    """Constant-in-S memory streaming sampler with exact per-image statistics."""
    def __init__(self, labels, mu_main, var_main, mu_rep, var_rep,
                 alpha=.7, eta=1., clamp=50.):
        self.labels = labels.long().view(-1)
        self.b = self.labels.numel()
        if not self.b or not 0 <= alpha <= 1 or eta < 0:
            raise ValueError("invalid labels/alpha/eta")
        self.alpha, self.eta, self.clamp = float(alpha), float(eta), float(clamp)
        self.analytical = tuple(t.double() for t in (mu_main, var_main, mu_rep, var_rep))
        if any(t.shape != mu_main.shape for t in self.analytical):
            raise ValueError("analytical moment shapes disagree")
        if mu_main.shape[0] != self.b:
            raise ValueError("batch/label mismatch")
        device = self.labels.device
        self.n = 0
        self.sum_o = torch.zeros(self.b, device=device, dtype=torch.float64)
        self.sum_m = torch.zeros_like(self.sum_o)
        self.sum_r = torch.zeros_like(self.sum_o)
        self.mean_m = self.mean_r = self.m2_m = self.m2_r = None
        self.logexp_m = self.logexp_r = self.logexp2_m = self.logexp2_r = None

    def add(self, m, r):
        if m.ndim != 3 or m.shape != r.shape or m.shape[1:] != self.analytical[0].shape:
            raise ValueError("logits must have matching [S,B,C] shape")
        if not m.shape[0]:
            return
        m, r = m.double(), r.double()
        if not torch.isfinite(m).all() or not torch.isfinite(r).all():
            raise ValueError("nonfinite sampled logits")
        s = m.shape[0]
        index = self.labels[None, :, None].expand(s, -1, 1)
        a = F.log_softmax(m, -1).gather(2, index).squeeze(-1)
        b = F.log_softmax(r, -1).gather(2, index).squeeze(-1)
        self.sum_m -= a.sum(0)
        self.sum_r -= b.sum(0)
        if self.alpha == 0:
            self.sum_o -= b.sum(0)
        elif self.alpha == 1:
            self.sum_o -= a.sum(0)
        else:
            self.sum_o -= torch.logaddexp(a+math.log(self.alpha), b+math.log1p(-self.alpha)).sum(0)
        self._merge_moments(m, r)
        for label, chunk in (("logexp_m",m),("logexp_r",r),
                             ("logexp2_m",2*m),("logexp2_r",2*r)):
            z = torch.logsumexp(chunk, dim=0)
            old = getattr(self, label)
            setattr(self, label, z if old is None else torch.logaddexp(old,z))

    def _merge_moments(self, m, r):
        n0, s = self.n, m.shape[0]
        for label, z in (("m",m),("r",r)):
            z_mean = z.mean(0)
            z_m2 = (z-z_mean).square().sum(0)
            old_mu = getattr(self, "mean_"+label)
            if n0 == 0:
                setattr(self, "mean_"+label, z_mean)
                setattr(self, "m2_"+label, z_m2)
            else:
                delta=z_mean-old_mu
                setattr(self, "mean_"+label, old_mu+delta*(s/(n0+s)))
                setattr(self, "m2_"+label,
                        getattr(self,"m2_"+label)+z_m2+delta.square()*(n0*s/(n0+s)))
        self.n += s

    def snapshot(self):
        if self.n < 2:
            raise ValueError("need at least 2 posterior draws")
        alpha, n, labels = self.alpha, self.n, self.labels
        mm, mr = self.mean_m, self.mean_r
        vm, vr = (self.m2_m/n).clamp_min(0), (self.m2_r/n).clamp_min(0)
        pm, pv, rm, rv = self.analytical
        def combine(x,y):return alpha*x+(1-alpha)*y
        logmgf_m=self.logexp_m-math.log(n)
        logmgf_r=self.logexp_r-math.log(n)
        def jen(mu,lg):return torch.logsumexp(lg,-1)-mu.gather(1,labels[:,None]).squeeze(1)
        O=self.sum_o/n; B=combine(self.sum_m/n,self.sum_r/n)
        J=combine(jen(mm,logmgf_m),jen(mr,logmgf_r))
        G=combine(surrogate_per_image(mm,vm,labels),surrogate_per_image(mr,vr,labels))
        P=combine(surrogate_per_image(pm,pv,labels),surrogate_per_image(rm,rv,labels))
        S=combine(surrogate_per_image(pm,pv,labels,self.eta),surrogate_per_image(rm,rv,labels,self.eta))
        D=combine(surrogate_per_image(pm,pv,labels,self.eta,self.clamp),surrogate_per_image(rm,rv,labels,self.eta,self.clamp))
        g=dict(zip(LOSS_KEYS,(O,B,J,G,P,S,D)))
        g.update(zip(SIX_TERMS,(B-O,J-B,G-J,P-G,S-P,D-S)))
        g["det_gap"]=D-B
        g["total_gap"]=D-O
        g["abs_total_gap_per_image"]=(D-O).abs()
        g["abs_det_gap_per_image"]=(D-B).abs()
        g["underestimate_det_per_image"]=(D<B).double()
        g["underestimate_per_image"]=(D<O).double()
        g["empirical_logit_variance"]=combine(vm.mean(-1),vr.mean(-1))
        g["propagated_logit_variance"]=combine(pv.mean(-1),rv.mean(-1))
        g["moment_mu_mae"]=combine((pm-mm).abs().mean(-1),(rm-mr).abs().mean(-1))
        g["moment_var_mae"]=combine((pv-vm).abs().mean(-1),(rv-vr).abs().mean(-1))
        g["moment_var_relative_error"]=combine((pv-vm).abs().mean(-1)/(vm.abs().mean(-1)+1e-6),
                                                  (rv-vr).abs().mean(-1)/(vr.abs().mean(-1)+1e-6))
        if self.clamp > 0:
            g["clip_fraction"]=combine((pv>self.clamp).double().mean(-1),(rv>self.clamp).double().mean(-1))
        else:
            g["clip_fraction"]=torch.zeros_like(O)
        ess_m=torch.exp(2*self.logexp_m-self.logexp2_m)/n
        ess_r=torch.exp(2*self.logexp_r-self.logexp2_r)/n
        # Per-image, class-averaged ESS; very low min ESS means the MGF is unreliable.
        g["mgf_ess_ratio_mean"]=.5*(ess_m.mean(-1)+ess_r.mean(-1))
        g["mgf_ess_ratio_min"]=torch.minimum(ess_m.amin(-1),ess_r.amin(-1))
        if max(abs(float((sum(g[k] for k in SIX_TERMS)-g["total_gap"]).abs().max())), 0) > 1e-8:
            raise ArithmeticError("per-image telescope failed")
        if min(float(g["mix_gap"].min()),float(g["softmax_jensen_gap"].min())) < -1e-8:
            raise ArithmeticError("empirical Jensen negative")
        if self.eta >=0 and float(g["clamp_error"].max()) > 1e-8:
            raise ArithmeticError("clamp increased surrogate")
        metrics = {k:float(v.mean()) for k,v in g.items()}
        metrics["mgf_ess_ratio_min"]=float(g["mgf_ess_ratio_min"].min())
        metrics["mc_samples"]=self.n
        metrics["num_examples"]=self.b
        # Private per-image vectors are not serialized: aggregation uses them for
        # E|gap| and exact dataset-wide quantiles, avoiding batch cancellation.
        metrics["_signed_vec"]=g["total_gap"].detach().cpu().tolist()
        metrics["_abs_vec"]=g["abs_total_gap_per_image"].detach().cpu().tolist()
        metrics["_abs_det_vec"]=g["abs_det_gap_per_image"].detach().cpu().tolist()
        return metrics


@torch.no_grad()
def decompose_batch(chunks: Iterable[tuple[torch.Tensor,torch.Tensor]],
                    labels, mu_main, var_main, mu_rep, var_rep, alpha=.7, eta=1.,clamp=50.):
    acc=GapAccumulator(labels,mu_main,var_main,mu_rep,var_rep,alpha,eta,clamp)
    for a,b in chunks:acc.add(a,b)
    return acc.snapshot()


def weighted_aggregate(items: Iterable[tuple[int,dict]]):
    """Dataset-level per-example stats; P90 computed globally over all examples."""
    data=list(items)
    if not data:raise ValueError("no evaluated batches")
    counts=[n for n,_ in data]
    if any(n<=0 or len(row["_abs_vec"])!=n or len(row["_abs_det_vec"])!=n for n,row in data):
        raise ValueError("inconsistent per-image count")
    total=sum(counts)
    ss={row["mc_samples"] for _,row in data}
    if len(ss)!=1:raise ValueError("inconsistent MC sample counts")
    all_abs=[v for _,r in data for v in r["_abs_vec"]]
    all_abs_det=[v for _,r in data for v in r["_abs_det_vec"]]
    all_signed=[v for _,r in data for v in r["_signed_vec"]]
    metric_names=set.intersection(*(set(r)-{"_abs_vec","_abs_det_vec","_signed_vec","mc_samples","num_examples"} for _,r in data))
    out={k:sum(n*r[k] for n,r in data)/total for k in metric_names if k!="mgf_ess_ratio_min"}
    out["mgf_ess_ratio_min"]=min(r["mgf_ess_ratio_min"] for _,r in data)
    out["mae_total_gap"]=sum(all_abs)/total
    out["mae_det_gap"]=sum(all_abs_det)/total
    out["p90_abs_total_gap"]=float(torch.quantile(torch.tensor(all_abs,dtype=torch.float64),.9))
    out["p90_abs_det_gap"]=float(torch.quantile(torch.tensor(all_abs_det,dtype=torch.float64),.9))
    out["underestimate_fraction"]=sum(z<0 for z in all_signed)/total
    out["abs_total_gap"]=abs(out["total_gap"]) # Legacy, labeled |mean gap| NOT MAE.
    out["num_examples"]=total
    out["mc_samples"]=ss.pop()
    if abs(sum(out[k] for k in SIX_TERMS)-out["total_gap"])>1e-8:
        raise ArithmeticError("dataset telescope failed")
    if out["mae_det_gap"]+1e-10 < abs(out["det_gap"]):
        raise ArithmeticError("Per-image |D-B| is smaller than |mean(D-B)|")
    if out["mae_total_gap"]+1e-10 < out["abs_total_gap"]:
        raise ArithmeticError("Jensen inequality between mean absolute and absolute mean failed")
    return out
