"""E7 | Directional influence of DAVI's variance correction on test confidence.

Run: python -m experiments.supplementary.e7 --help
Uses production checkpoint restore, deterministic test preprocessing and native
sampled static_logit (mean logits then softmax); no training or checkpoint writes.

Contrast objectives at the SAME theta: MeanOnly, DAVI correction, DAVI CE,
AVI-Branch expected CE. Only visual posterior mean M_r is perturbed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .core import (ROOT, checkpoint_path, location, native_moments, cached_encode,
                   cached_draw, reference_loader, restore, save_json, validate_cache_parity)
from . import e6_analysis as e6
from methods.det_bayesrt_mmrl.losses import det_jensen_ce

SCHEMA = 1
BASELINE_COMMIT = '1473048691ab433be8488c411babd060972d9d31'
DIRECTIONS = ("variance_correction", "mean_only", "davi_full", "avi_mc")


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _epoch_selection(run, request):
    if request == 'latest':
        _, _, actual = checkpoint_path(run, 'latest')
        return 'latest', None, actual
    target = int(request)
    _, _, last = checkpoint_path(run, 'latest')
    if target == last:
        return 'latest', None, last
    checkpoint_path(run, 'trajectory', target)
    return 'trajectory', target, target


def _test_subset(trainer, images, batch_size, seed):
    """One fixed, uniformly selected held-out subset; identical indices for all methods."""
    loader = reference_loader(trainer, 'test', batch_size)
    total = len(loader.dataset)
    count = total if images == 0 else min(images, total)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randperm(total, generator=generator)[:count].sort().values.tolist()
    signature = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    subset = Subset(loader.dataset, ids)
    return (DataLoader(subset, batch_size=batch_size, shuffle=False,
                       num_workers=0, drop_last=False), ids, signature)


def _native_logits_prob(cache, alpha, samples, chunk, seed, devices, grad):
    """Exactly the E6 native sampled static_logit semantics, with local CRN.

    Rebuild posterior samples for each +/- parameter perturbation under the
    SAME seed. Encoding is cached because it is independent of M_r.
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        logits_sum = None
        for start in range(0, samples, chunk):
            n = min(chunk, samples - start)
            main, rep = cached_draw(cache, n)
            subtotal = alpha * main.float().sum(0) + (1.0 - alpha) * rep.float().sum(0)
            logits_sum = subtotal if logits_sum is None else logits_sum + subtotal
        return F.softmax(logits_sum / samples, dim=-1)


def _objective_directions(model, cfg, n_classes, x, y, param, mc_draws, sample_seed):
    """Compute same-checkpoint classification-only gradients on train_fixed.

    g_R = d(L_DAVI - L_mean) / d(M_r). No KL, alignment or optimizer step.
    AVI MC classification gradient is estimated from independent posterior draws.
    """
    mom = native_moments(model, x, cfg)
    alpha = float(cfg.ALPHA)
    scale = float(getattr(cfg, 'DET_VAR_SCALE', 1.0))
    clamp = float(getattr(cfg, 'DET_VAR_CLAMP', 50.0))
    # Branch checkpoint has no DAVI loss settings: use actual DAVI controls.
    if not math.isfinite(scale) or scale <= 0 or clamp != 50.0:
        raise RuntimeError('E7 expected native DAVI variance controls 1.0/50.0')
    mu_m = mom['mu_main'][..., :n_classes]
    mu_r = mom['mu_rep'][..., :n_classes]
    var_m = mom['var_main'][..., :n_classes]
    var_r = mom['var_rep'][..., :n_classes]
    mean_loss = alpha * F.cross_entropy(mu_m.float(), y) + (1-alpha) * F.cross_entropy(mu_r.float(), y)
    det_loss = alpha * det_jensen_ce(mu_m, var_m, y, var_scale=1., var_clamp=50.)
    det_loss = det_loss + (1-alpha) * det_jensen_ce(mu_r, var_r, y, var_scale=1., var_clamp=50.)
    g_det = torch.autograd.grad(det_loss, param, retain_graph=True, allow_unused=False)[0].detach().float()
    g_mean = torch.autograd.grad(mean_loss, param, retain_graph=False, allow_unused=False)[0].detach().float()
    g_corr = g_det - g_mean
    clamp_diag = {'main':float((var_m.detach() >= 50.).float().mean()),
                  'rep':float((var_r.detach() >= 50.).float().mean())}
    del mom, mu_m, mu_r, var_m, var_r, mean_loss, det_loss
    # The train-fixed MC reference gradient is a same-theta counterfactual,
    # NOT an attempt to reproduce the optimizer's noisy single minibatch step.
    devices = [torch.cuda.current_device()] if x.is_cuda else []
    with torch.enable_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(sample_seed)
        with torch.no_grad():
            cache = cached_encode(model, x, n_classes)
        acc = torch.zeros_like(param, dtype=torch.float32)
        for j in range(mc_draws):
            main, rep = cached_draw(cache, 1)
            loss = (alpha * F.cross_entropy(main[0].float(), y)
                    + (1-alpha) * F.cross_entropy(rep[0].float(), y))
            grad = torch.autograd.grad(loss, param, retain_graph=False, allow_unused=False)[0]
            acc.add_(grad.detach().float())
    g_avi = acc / mc_draws
    results = {'variance_correction':g_corr, 'mean_only':g_mean,
               'davi_full':g_det, 'avi_mc':g_avi}
    diag = {'variance_clamp_fraction':clamp_diag,
            'mean_only_rms':float(g_mean.double().square().mean().sqrt()),
            'variance_correction_rms':float(g_corr.double().square().mean().sqrt()),
            'davi_full_rms':float(g_det.double().square().mean().sqrt()),
            'avi_mc_rms':float(g_avi.double().square().mean().sqrt()),
            'mc_draws':int(mc_draws),
            'davi_correction_formula':'g_R = grad_Mr(L_DAVI) - grad_Mr(L_MeanOnly)',
            'avi_mc_formula':'mean_s grad_Mr(alpha*CE(main_s)+(1-alpha)*CE(rep_s))'}
    for k, v in results.items():
        if not bool(torch.isfinite(v).all()):
            raise RuntimeError(f'Non-finite {k} gradient')
        diag[k+'_norm'] = float(v.double().norm())
    return results, diag


def _metrics(p, y):
    """Returns E6-compatible ECE/ACC values as fractions (not percentages)."""
    if p.numel() == 0:
        return {}
    return e6.calibration_metrics(p.detach().float().cpu(), y.detach().long().cpu())


def _group(mask, data, original, plus, minus, true_y, threshold):
    ix = torch.nonzero(mask).flatten()
    if not len(ix):
        return dict(n=0)
    orig = original[ix]
    m = minus[ix]
    p = plus[ix]
    yy = true_y[ix]
    yp = orig.argmax(-1)
    cp = orig.gather(1, yp[:,None]).squeeze(1)
    cminus = m.gather(1, yp[:,None]).squeeze(1)
    observed = data['observed_minus'][ix]
    predicted = data['predicted_minus'][ix]
    A = data['A_raw'][ix]
    good = predicted.abs() > threshold
    true_sign = (predicted[good] * observed[good]) > 0 if good.any() else torch.empty(0, dtype=torch.bool)
    return {
        'n': int(len(ix)), 'n_tested_sign':int(good.sum()),
        'fraction_A_positive':float((A>0).float().mean()),
        'sign_agreement':float(true_sign.float().mean()) if true_sign.numel() else None,
        'mean_A_raw':float(A.mean()),
        'mean_predicted_delta':float(predicted.mean()),
        'mean_observed_delta_fixed_class':float(observed.mean()),
        'mean_observed_delta_top1':float((m.max(-1).values-cp).mean()),
        'mean_plus_delta_fixed_class':float(data['observed_plus'][ix].mean()),
        'mean_abs_linear_error':float((predicted-observed).abs().mean()),
        'mean_baseline_confidence':float(cp.mean()),
        'mean_minus_confidence':float(m.max(-1).values.mean()),
        'accuracy_base':float((yp==yy).float().mean()),
        'accuracy_minus':float((m.argmax(-1)==yy).float().mean()),
        'top1_flip_rate':float((m.argmax(-1)!=yp).float().mean()),
    }


def _one_run(args, dataset, seed, method_name, epoch_request):
    run = location(args.output, 'native', dataset, method_name, args.shot, seed)
    selection, epoch, ep_actual = _epoch_selection(run, epoch_request)
    target = (Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output)
    folder = target / 'e7' / dataset / f'shots_{args.shot}' / f'seed_{seed}' / f'epoch_{ep_actual}' / method_name
    json_path = folder / 'result.json'
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f'E7 refuses to reuse unverified stale output: {json_path}; FORCE_EVAL=1')

    trainer, meta, ck = restore(run, gpu=args.gpu, selection=selection, epoch=epoch)
    if meta['study'] != 'native' or meta['status'] != 'complete' or meta['method'] != method_name:
        raise RuntimeError('E7 only supports completed native AVI-B/DAVI checkpoints')
    method, model = trainer.method, trainer.method.model
    model.eval()
    cfg = method.cfg.BAYESRT_MMRL
    if str(cfg.EVAL_FUSION_VARIANT).lower() != 'static_logit' or bool(cfg.EVAL_USE_POSTERIOR_MEAN):
        raise RuntimeError('E7 requires E6-style sampled static_logit, not static_prob/mean-only')
    param = dict(model.named_parameters())[e6.GROUP_NAMES['r_mean']]
    if not param.requires_grad:
        raise RuntimeError('Visual posterior mean must be trainable')
    x_train, y_train, train_meta = e6._fixed_grad_batch(trainer, args.grad_batch, args.input_seed)
    # Important: native DAVI controls are always 1.0/50.0 for both checkpoints.
    proxy = type('E7DaviControls', (), {'ALPHA':float(cfg.ALPHA),
                                       'DET_VAR_SCALE':1.0,'DET_VAR_CLAMP':50.0})()
    dirs_raw, grad_info = _objective_directions(model, proxy, method.num_classes, x_train, y_train,
                                                 param, args.mc_draws, args.mc_seed)
    with torch.no_grad():
        pnorm = float(param.detach().float().norm())
    if pnorm <= 1e-12:
        raise RuntimeError('M_r norm is zero, relative perturbation undefined')
    device_list = [torch.cuda.current_device()] if param.is_cuda else []
    loader, ids, ids_sha = _test_subset(trainer, args.eval_images, args.eval_batch, args.eval_seed)

    normals = {}
    for k, g in dirs_raw.items():
        n = float(g.double().norm())
        if n <= 1e-15:
            normals[k] = None
        else:
            normals[k] = g / n
    delta = pnorm * args.relative_step
    if delta <= 0 or not math.isfinite(delta):
        raise ValueError('Invalid relative perturbation')
    folder.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    # Store compact per-sample data on CPU. Fixed-class confidence always
    # refers to the baseline predicted class, even if a perturbation flips it.
    aggregate = {key:{field:[] for field in ('base','plus','minus','y','pred','A','linear','observed','observed_plus')}
                 for key in DIRECTIONS}
    n_total = 0
    parity_max = None
    try:
        for bi, batch in enumerate(loader):
            x = batch['img'].to(method.device)
            labels = batch['label'].long().to(method.device)
            n = labels.numel()
            with torch.no_grad():
                cache = cached_encode(model, x, method.num_classes)
            # Verify sampled cached logits reproduce production native forward once.
            if args.parity and bi == 0:
                with torch.no_grad():
                    parity_max = validate_cache_parity(model, x, method.num_classes, cache, n=2)
            repeat_seed = args.eval_mc_seed + 100003 * bi
            probs = _native_logits_prob(cache, float(cfg.ALPHA), args.samples, args.chunk,
                                        repeat_seed, device_list, grad=True)
            top = probs.detach().argmax(-1)
            cp = probs.gather(1, top[:,None]).squeeze(1)
            # Per-example exact VJP of *native sampled static_logit* confidence.
            # No 'softmax(posterior_mean)' proxy is used.
            a = {k:[] for k in DIRECTIONS}
            for ii in range(n):
                grad = torch.autograd.grad(cp[ii], param,
                                           retain_graph=(ii != n-1), allow_unused=False)[0].detach().float()
                for k, g in dirs_raw.items():
                    a[k].append(float(torch.sum(grad.double() * g.double()).item()))
            original = param.detach().clone()
            with torch.no_grad():
                p0 = probs.detach().cpu().float()
                yp = top.detach().cpu()
                yy = labels.detach().cpu()
                for k in DIRECTIONS:
                    d = normals[k]
                    if d is None:
                        continue
                    # Common random numbers, strictly symmetric +/- steps.
                    param.copy_(original - delta*d.to(dtype=param.dtype))
                    pm = _native_logits_prob(cache, float(cfg.ALPHA), args.samples, args.chunk,
                                             repeat_seed, device_list, grad=False).detach().cpu().float()
                    param.copy_(original + delta*d.to(dtype=param.dtype))
                    pp = _native_logits_prob(cache, float(cfg.ALPHA), args.samples, args.chunk,
                                             repeat_seed, device_list, grad=False).detach().cpu().float()
                    param.copy_(original)
                    scale = grad_info[k+'_norm']
                    A = torch.tensor(a[k], dtype=torch.float64)
                    predicted = -delta * A / scale
                    ix = torch.arange(n)
                    cm = pm[ix,yp].double()
                    cp_plus = pp[ix,yp].double()
                    base = p0[ix,yp].double()
                    for ii in range(n):
                        sample_rows.append({
                            'dataset':dataset,'shot':args.shot,'seed':seed,'method':method_name,
                            'epoch':ep_actual,'direction':k,'subset_position':n_total+ii,
                            'dataset_index':ids[n_total+ii], 'true_class':int(yy[ii]),
                            'predicted_class_base':int(yp[ii]),
                            'correct_base':int(yp[ii]==yy[ii]),
                            'confidence_base':float(base[ii]),
                            'A_raw_grad_conf_dot_grad_objective':float(A[ii]),
                            'predicted_delta_conf_descent':float(predicted[ii]),
                            'observed_delta_conf_descent_fixed_class':float(cm[ii]-base[ii]),
                            'observed_delta_conf_ascent_fixed_class':float(cp_plus[ii]-base[ii]),
                            'central_delta_conf_descent_fixed_class':float((cm[ii]-cp_plus[ii])/2),
                            'confidence_minus_top1':float(pm[ii].max()),
                            'predicted_class_minus':int(pm[ii].argmax()),
                            'flip_under_minus':int(yp[ii]!=pm[ii].argmax()),
                        })
                    agg = aggregate[k]
                    for field,val in [('base',p0),('plus',pp),('minus',pm),('y',yy),('pred',yp),
                                      ('A',A),('linear',predicted),('observed',cm-base),
                                      ('observed_plus',cp_plus-base)]:
                        agg[field].append(val)
            n_total += n
            del probs, cp, cache, x, labels
        summaries=[]
        for key in DIRECTIONS:
            if normals[key] is None:
                summaries.append({'direction':key,'status':'zero_gradient','gradient_norm':0.0})
                continue
            agg = {field:torch.cat(pieces, 0) for field,pieces in aggregate[key].items()}
            orig, pp, pm = agg['base'], agg['plus'], agg['minus']
            y, pred = agg['y'], agg['pred']
            conf = orig.max(-1).values
            common = {'observed_minus':agg['observed'], 'observed_plus':agg['observed_plus'],
                      'predicted_minus':agg['linear'], 'A_raw':agg['A']}
            masks = {
                'all': torch.ones(len(conf), dtype=torch.bool),
                'correct':pred.eq(y),
                'wrong':pred.ne(y),
                'high_conf_correct':pred.eq(y)&(conf>=args.conf_threshold),
                'high_conf_wrong':pred.ne(y)&(conf>=args.conf_threshold),
            }
            global_base = _metrics(orig,y)
            global_minus = _metrics(pm,y)
            global_plus = _metrics(pp,y)
            # Metric changes are on SAME examples; not retrained/checkpoint replaced.
            result = {'direction':key,'status':'ok','n_eval':len(y),
                      'step_l2':delta,'relative_step':args.relative_step,
                      'gradient_norm':grad_info[key+'_norm'],
                      'avg_conf_base':float(conf.mean()),
                      'avg_conf_descent':float(pm.max(-1).values.mean()),
                      'avg_conf_ascent':float(pp.max(-1).values.mean()),
                      'avg_accuracy_base':float((pred==y).float().mean()),
                      'avg_accuracy_descent':float((pm.argmax(-1)==y).float().mean()),
                      'ece_base':float(global_base['ece_15_equal_width']),
                      'ece_descent':float(global_minus['ece_15_equal_width']),
                      'ece_ascent':float(global_plus['ece_15_equal_width']),
                      'confidence_accuracy_gap_base':float(conf.mean()-(pred==y).float().mean()),
                      'confidence_accuracy_gap_descent':float(pm.max(-1).values.mean()-(pm.argmax(-1)==y).float().mean()),
                      'groups':{group:_group(mask,common,orig,pp,pm,y,args.sign_threshold)
                                for group,mask in masks.items()},
                      'mean_central_fd_delta_descent':float((pm[torch.arange(len(y)),pred]-pp[torch.arange(len(y)),pred]).mean()/2),
                      'mean_first_order_delta_descent':float(agg['linear'].mean()),
                      'mean_forward_delta_descent_fixed_class':float(agg['observed'].mean()),
                      'mean_abs_first_order_error':float((agg['linear']-agg['observed']).abs().mean()),
                      'mean_abs_central_fd_error':float((agg['linear']-(agg['observed']-agg['observed_plus'])/2).abs().mean())}
            summaries.append(result)
        metadata = {'schema':SCHEMA,'dataset':dataset,'seed':seed,'shot':args.shot,
                    'method':method_name,'epoch':ep_actual,'source_base_commit':BASELINE_COMMIT,
                    'split_grad':'train_fixed',
                    'split_confidence':'test','checkpoint_sha256':ck['checkpoint_sha256'],
                    'gradient_input_sha256':train_meta['input_sha256'],
                    'test_subset_indices_sha256':ids_sha,'selected_test_indices':ids,
                    'fusion':'native_sampled_static_logit','n_eval':n_total,
                    'mean_parameter_name':e6.GROUP_NAMES['r_mean'],
                    'davi_var_scale':1.,'davi_var_clamp':50.,
                    'mc_reference_grad_draws':args.mc_draws,'pred_samples':args.samples,
                    'pred_chunk':args.chunk,'gradient_seed':args.mc_seed,
                    'prediction_seed':args.eval_mc_seed,'relative_step':args.relative_step,
                    'sign_threshold':args.sign_threshold,'conf_threshold':args.conf_threshold,
                    'cache_parity_max_abs':parity_max,'gradient_diagnostics':grad_info,
                    'directions':summaries,
                    'interpretation':'One-step, same-checkpoint, normalized-gradient perturbations; not retraining or proof of ECE causality. Test data is not used to calculate update directions.'}
        save_json(json_path, metadata)
        _write_csv(folder/'samples.csv',sample_rows)
        print(f'[E7] {dataset} seed={seed} {method_name} epoch={ep_actual} '
              f'n_test={n_total} direction_R={next(z for z in summaries if z["direction"]=="variance_correction")["status"]}',flush=True)
        return metadata
    finally:
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def report(output):
    root = Path(output)
    root = root if root.is_absolute() else ROOT/root
    folder = root/'e7'
    per_seed=[]
    for p in sorted(folder.glob('*/shots_*/seed_*/epoch_*/*/result.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('schema')!=SCHEMA:
            continue
        for item in d['directions']:
            if item['status']!='ok':
                continue
            base = {k:d[k] for k in ('dataset','seed','shot','method','epoch')}
            base.update({k:item[k] for k in item if k!='groups'})
            for group,stats in item['groups'].items():
                per_seed.append({**base,'group':group,**stats})
    _write_csv(folder/'direction_seed.csv',per_seed)
    grouped={}
    for row in per_seed:
        key=tuple(row[k] for k in ('dataset','shot','method','epoch','direction','group'))
        grouped.setdefault(key,[]).append(row)
    summary=[]
    for key,rows in sorted(grouped.items()):
        record=dict(zip(('dataset','shot','method','epoch','direction','group'),key))
        record['n_seeds']=len(set(r['seed'] for r in rows))
        record['n_samples_total']=sum(r['n'] for r in rows)
        # Each independent training seed contributes equally; do not treat samples or MC as seeds.
        for field in rows[0]:
            if field in record or field in ('seed','n','n_eval','n_tested_sign','step_l2'):
                continue
            vals=[r.get(field) for r in rows if type(r.get(field)) in (int,float)
                  and math.isfinite(r[field])]
            if vals:
                record[field+'_mean']=statistics.mean(vals)
                record[field+'_sd']=statistics.stdev(vals) if len(vals)>1 else None
        summary.append(record)
    _write_csv(folder/'direction_summary.csv',summary)
    print(f'[E7 REPORT] {len(per_seed)} seed-group rows; {len(summary)} summarized rows',flush=True)
    # Papers benefit from predicted-vs-actual scatter rather than unrelated bars.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[E7] matplotlib not installed; CSV/JSON still available', flush=True)
        return
    for dataset in sorted(set(r['dataset'] for r in per_seed)):
        fig, ax = plt.subplots(figsize=(6.2,5.0))
        pts=[]
        for p in sorted((folder/dataset).glob('shots_*/seed_*/epoch_*/*/samples.csv')):
            with p.open(encoding='utf-8',newline='') as f:
                pts += [r for r in csv.DictReader(f) if r['direction']=='variance_correction']
        for method,marker in (('branch','o'),('davi','x')):
            subset=[r for r in pts if r['method']==method]
            if not subset:
                continue
            xx=[float(r['predicted_delta_conf_descent']) for r in subset]
            yy=[float(r['observed_delta_conf_descent_fixed_class']) for r in subset]
            ax.scatter(xx, yy, s=12, alpha=.30, marker=marker,
                       label=('AVI-B checkpoint' if method=='branch' else 'DAVI checkpoint'))
        allvals=[float(r[f]) for r in pts for f in ('predicted_delta_conf_descent',
                                                   'observed_delta_conf_descent_fixed_class')]
        if allvals:
            lo,hi=min(allvals),max(allvals)
            ax.plot([lo,hi],[lo,hi],ls='--',color='black',lw=1,label='First-order agreement')
        ax.set(xlabel='Predicted confidence change (descent -g_R)',
               ylabel='Observed fixed-class confidence change',
               title=f'E7: {dataset} | normalized-gradient local intervention')
        ax.legend(fontsize=8)
        fig.tight_layout()
        dest=folder/'figures'/f'E7_directional_parity_{dataset}.png'
        dest.parent.mkdir(exist_ok=True,parents=True)
        fig.savefig(dest,dpi=220)
        fig.savefig(dest.with_suffix('.pdf'))
        plt.close(fig)


def self_test():
    """CPU algebraic test for g_R and the signed confidence directional derivative."""
    theta=torch.tensor([2.0,0.8,-0.2],dtype=torch.float64,requires_grad=True)
    v=torch.tensor([0.3,0.7,0.5],dtype=torch.float64)
    s=theta.softmax(-1)
    D=-theta[0]+torch.logsumexp(theta+v/2,-1)
    mean=-theta[0]+torch.logsumexp(theta,-1)
    R=D-mean
    assert torch.allclose(torch.autograd.grad(R,theta,retain_graph=True)[0],
                          (theta+v/2).softmax(-1)-s,atol=1e-12)
    g=torch.autograd.grad(R,theta)[0].detach()
    c=theta.softmax(-1)[0]
    dot=(torch.autograd.grad(c,theta)[0].detach()*g).sum()
    eps=1e-5
    with torch.no_grad():
        finite=((theta-eps*g).softmax(-1)[0]-(theta+eps*g).softmax(-1)[0])/(2*eps)
    assert torch.isclose(finite,-dot,rtol=1e-5,atol=1e-8)
    print('[E7 SELFTEST PASS] Multi-class R gradient and signed confidence shift')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='output_supplementary_v7')
    p.add_argument('--datasets',nargs='+',default=['caltech101','eurosat','oxford_pets'])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--methods',nargs='+',choices=['branch','davi'],default=['branch','davi'])
    p.add_argument('--epochs',nargs='+',default=['latest'])
    p.add_argument('--shot',type=int,default=16)
    p.add_argument('--gpu',default='0')
    p.add_argument('--grad-batch',type=int,default=32)
    p.add_argument('--input-seed',type=int,default=2027)
    p.add_argument('--mc-draws',type=int,default=32)
    p.add_argument('--mc-seed',type=int,default=99173)
    p.add_argument('--eval-images',type=int,default=128,help='0 = entire test set')
    p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--eval-seed',type=int,default=2027)
    p.add_argument('--samples',type=int,default=50,help='Posterior samples for native static_logit')
    p.add_argument('--chunk',type=int,default=10)
    p.add_argument('--eval-mc-seed',type=int,default=81721)
    p.add_argument('--relative-step',type=float,default=0.002,
                   help='L2 step = relative-step * norm(M_r) for unit gradient directions')
    p.add_argument('--conf-threshold',type=float,default=.9)
    p.add_argument('--sign-threshold',type=float,default=1e-6)
    p.add_argument('--no-parity',action='store_true')
    p.add_argument('--report',action='store_true',help='Generate reports/figures only')
    p.add_argument('--self-test',action='store_true',help='CPU unit algebra test without checkpoints')
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if args.self_test:
        self_test();return
    if args.report:
        report(args.output);return
    for key in ('grad_batch','mc_draws','eval_batch','samples','chunk','shot'):
        if getattr(args,key)<=0:
            p.error(f'--{key.replace("_","-")} must be positive')
    if args.eval_images<0 or not math.isfinite(args.relative_step) or args.relative_step<=0:
        p.error('eval-images >= 0 and relative-step > 0 required')
    if not 0<args.conf_threshold<=1:
        p.error('conf-threshold must lie in (0,1]')
    args.parity=not args.no_parity
    # Deliberately do not import/init training runner or save model in E7.
    for dataset in args.datasets:
        for seed in args.seeds:
            for method in args.methods:
                for epoch in args.epochs:
                    _one_run(args,dataset,seed,method,epoch)
    report(args.output)


if __name__=='__main__':
    main()
