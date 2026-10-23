"""DAVI vs AVI checkpoint-paired posterior / deep-feature diagnostics.

Additive, read-only supplement for MMRL/experiments/supplementary, R15-family.
No new model, training objective or checkpoint writes. Run from MMRL root:
  python -m experiments.supplementary.paired_checkpoint_diagnostics \
      --output output_supplementary_v7 --datasets caltech101 eurosat oxford_pets \
      --seeds 1 2 3 --shot 16 --images 128 --batch 16 --gpu 0

Outputs: OUT/summary/ppt_posterior_feature_pairs.csv, *_summary.csv, *.json.
Fails closed on mismatched seed/label/input/class indices and missing checkpoints.
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

import numpy as np

import torch
import torch.nn.functional as F

from .cli import select_e3_fixed_batch
from .core import ROOT, location, restore, save_json


def _cos(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f'Unaligned shapes: {a.shape} != {b.shape}')
    if float(a.norm()) < 1e-12 or float(b.norm()) < 1e-12:
        return float('nan')
    return float(F.cosine_similarity(a[None], b[None], dim=1)[0])


def _norm_relative(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        raise ValueError(f'Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}')
    return float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-12))


def gaussian_comparison(mu_d, sigma_d, mu_b, sigma_b):
    """Diagnostic for diagonal q_d and q_branch in aligned parameter coordinates.

    Gaussian KL is asymmetric and evaluated PER MATRIX ELEMENT. It compares
    trained native distributions, not distance from an unknowable true posterior.
    """
    mu_d, mu_b = mu_d.float().cpu(), mu_b.float().cpu()
    sigma_d, sigma_b = sigma_d.float().cpu(), sigma_b.float().cpu()
    if mu_d.shape != mu_b.shape or sigma_d.shape != mu_d.shape or sigma_b.shape != mu_d.shape:
        raise ValueError('Posterior tensor shapes must be identical')
    sd, sb = sigma_d.clamp_min(1e-12), sigma_b.clamp_min(1e-12)
    delta = mu_d-mu_b
    kl_db = (sb.log()-sd.log()+(sd.square()+delta.square())/(2*sb.square())-.5).mean()
    kl_bd = (sd.log()-sb.log()+(sb.square()+delta.square())/(2*sd.square())-.5).mean()
    return dict(mean_abs_difference=float(delta.abs().mean()),
                mean_relative_distance=_norm_relative(mu_d,mu_b),
                mean_cosine=_cos(mu_d,mu_b),
                davi_variance_trace=float(sd.square().sum()),
                branch_variance_trace=float(sb.square().sum()),
                variance_trace_ratio=float(sd.square().sum()/sb.square().sum().clamp_min(1e-12)),
                kl_davi_to_branch_per_dim=float(kl_db),
                kl_branch_to_davi_per_dim=float(kl_bd))


def _rowwise_cos(x, y):
    if x.shape != y.shape:
        raise ValueError(f'Cannot compare features with shapes {x.shape} / {y.shape}')
    return F.cosine_similarity(x.float(),y.float(),dim=-1).cpu()


def _extract(run_dir, gpu, images, batch_size, input_seed):
    """Restore and read one native trained model, then release GPU memory."""
    trainer, meta, ck = restore(run_dir, gpu=gpu, selection='latest')
    if meta['study']!='native' or meta.get('status')!='complete':
        raise RuntimeError('Expected complete native checkpoint: '+str(run_dir))
    model=trainer.method.model
    model.eval()
    chosen, inp = select_e3_fixed_batch(trainer, images,
                         strategy='stratified',selection_seed=input_seed)
    classes = tuple(trainer.dm.dataset.classnames)
    r = model.image_encoder.bayes_proj_rep
    t = model.text_posterior
    post={}
    for key, q in [('r',r), ('t',t)]:
        if q is None:
            raise RuntimeError('Missing posterior: '+key)
        post[key+'_mean']=q.posterior_mean.detach().float().cpu().clone()
        post[key+'_sigma']=q.expanded_posterior_sigma().detach().float().cpu().clone()
    with torch.no_grad():
        token_t, token_v = model.representation_learner()
        token_t_cpu=[x.detach().float().cpu().clone() for x in token_t]
        token_v_cpu=[x.detach().float().cpu().clone() for x in token_v]
        shared = model.representation_learner.compound_rep_tokens.detach().float().cpu().clone()
        all_main,all_rep=[] ,[]
        # The class labels and text prompts are fixed; cache once per model.
        all_text=None
        for i in range(0,len(chosen['label']),batch_size):
            x=chosen['img'][i:i+batch_size].to(trainer.method.device)
            from .core import cached_encode
            cache=cached_encode(model,x,trainer.method.num_classes)
            main=cache['main'].float()
            rep=F.normalize(model.image_encoder.rep_samples(cache['rep'],1,use_mean=True)[0].float(),dim=-1)
            text_samples,text_mean=model.text_sample_features(cache['eot'],1,use_mean=True)
            # Same class-index order for every minibatch.
            if all_text is None:
                all_text=text_mean.detach().float().cpu().clone()
            else:
                if not torch.allclose(all_text, text_mean.detach().float().cpu(),atol=2e-5,rtol=2e-5):
                    raise RuntimeError('Text prototypes vary across fixed image minibatches')
            all_main.append(main.detach().cpu().clone())
            all_rep.append(rep.detach().cpu().clone())
        all_main=torch.cat(all_main)
        all_rep=torch.cat(all_rep)
        labels=chosen['label'].long().cpu()
        # Diagnostic scores based on posterior projection means. NOT native formal MC inference.
        z_main=100*all_main @ all_text.t()
        z_rep=100*all_rep @ all_text.t()
        alpha=float(trainer.method.cfg.BAYESRT_MMRL.ALPHA)
        z_fusion=alpha*z_main+(1-alpha)*z_rep
    result=dict(method=meta['method'],dataset=meta['dataset'],shot=int(meta['shot']),
                seed=int(meta['seed']),checkpoint_sha256=ck['checkpoint_sha256'],
                checkpoint_epoch=int(ck['epoch']), input_sha256=inp['input_sha256'],
                indices_sha256=inp['selected_indices_sha256'], n_images=len(labels),
                n_classes=len(classes), classnames_sha256=hashlib.sha256(
                    json.dumps(classes,ensure_ascii=False).encode()).hexdigest(),
                labels=labels,posterior=post,tokens_t=token_t_cpu,tokens_v=token_v_cpu,
                shared=shared,main=all_main,rep=all_rep,text=all_text,
                fusion=z_fusion)
    del trainer, model, r, t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def compare(d, b):
    """Alignment is mandatory; use matched frozen CLIP output coordinates only."""
    for k in ['dataset','shot','seed','input_sha256','indices_sha256',
              'n_classes','classnames_sha256','n_images','checkpoint_epoch']:
        if d[k]!=b[k]:
            raise RuntimeError(f'Cannot pair due to {k}: {d[k]} != {b[k]}')
    if not torch.equal(d['labels'],b['labels']):
        raise RuntimeError('Class labels are not identical')
    p_r=gaussian_comparison(d['posterior']['r_mean'],d['posterior']['r_sigma'],
                            b['posterior']['r_mean'],b['posterior']['r_sigma'])
    p_t=gaussian_comparison(d['posterior']['t_mean'],d['posterior']['t_sigma'],
                            b['posterior']['t_mean'],b['posterior']['t_sigma'])
    if len(d['tokens_v'])!=len(b['tokens_v']) or len(d['tokens_t'])!=len(b['tokens_t']):
        raise RuntimeError('Prompt layers misaligned')
    prompt_v=[_cos(x,y) for x,y in zip(d['tokens_v'],b['tokens_v'])]
    prompt_t=[_cos(x,y) for x,y in zip(d['tokens_t'],b['tokens_t'])]
    feats={}
    for key in ('main','rep','text'):
        sim=_rowwise_cos(d[key],b[key])
        feats[key+'_cos_mean']=float(sim.mean())
        feats[key+'_cos_median']=float(sim.median())
        feats[key+'_cos_sd']=float(sim.std(unbiased=True)) if len(sim)>1 else 0.0
        feats[key+'_relative_l2']=_norm_relative(d[key],b[key])
    for label,source in (('davi',d),('branch',b)):
        pred=source['fusion'].argmax(-1)
        feats[label+'_mean_projected_fusion_acc']=float((pred==d['labels']).float().mean())
    feats['mean_projected_fusion_disagreement']=float((d['fusion'].argmax(-1)!=b['fusion'].argmax(-1)).float().mean())
    feats['mean_projected_fusion_logit_mae']=float((d['fusion']-b['fusion']).abs().mean())
    detail=dict(schema=1,dataset=d['dataset'],shot=d['shot'],seed=d['seed'],
         epoch=d['checkpoint_epoch'],input_sha256=d['input_sha256'],
         n_images=d['n_images'],checkpoint_sha256={'davi':d['checkpoint_sha256'],
          'branch':b['checkpoint_sha256']},
         posterior={'visual':p_r,'text':p_t},
         prompt={'shared_token_cos':_cos(d['shared'],b['shared']),
                 'visual_per_layer_cos':prompt_v,'text_per_layer_cos':prompt_t},
         features=feats,
         interpretation=('Two native models: different original initialization/prior; '
          'point-estimate normalized projections; not original MC-fused predictions, '
          'not a Jensen-only causal test.'))
    row=dict(dataset=d['dataset'],shot=d['shot'],seed=d['seed'],epoch=d['checkpoint_epoch'],
       n_images=d['n_images'],input_sha256=d['input_sha256'],
       davi_checkpoint_sha256=d['checkpoint_sha256'],
       branch_checkpoint_sha256=b['checkpoint_sha256'])
    for tag,stats in [('r',p_r),('t',p_t)]:
        row.update({tag+'_'+k:v for k,v in stats.items()})
    row.update(dict(shared_token_cos=detail['prompt']['shared_token_cos'],
                    visual_prompt_cos_mean=statistics.mean(prompt_v),
                    text_prompt_cos_mean=statistics.mean(prompt_t)))
    row.update(feats)
    return row,detail


def write_feature_matrices(path, d, b):
    """Save true aligned across-model cosine matrices for later offline plotting.

    For each modality, row i is DAVI sample/class i, column j is Branch
    sample/class j. These are NOT intra-class covariance or gradient matrices.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays = {}
    for name in ('main','rep','text'):
        x = F.normalize(d[name].float(),dim=-1)
        y = F.normalize(b[name].float(),dim=-1)
        if x.ndim!=2 or y.ndim!=2 or x.shape!=y.shape:
            raise RuntimeError('Feature shapes not aligned: '+name)
        arrays[name] = (x @ y.T).detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path,dataset=np.asarray(d['dataset']),shot=d['shot'],
        seed=d['seed'],epoch=d['checkpoint_epoch'],n_images=d['n_images'],
        labels=d['labels'].cpu().numpy(),**arrays)


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        return
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(rows):
    sets=sorted(set(r['dataset'] for r in rows))
    fields=[k for k,v in rows[0].items() if type(v) in (int,float) and
            k not in ('shot','seed','epoch','n_images')]
    out=[]
    for dataset in sets:
        group=[r for r in rows if r['dataset']==dataset]
        line=dict(dataset=dataset,shot=group[0]['shot'],n_seeds=len(group),
                  seeds=' '.join(str(r['seed']) for r in group))
        for k in fields:
            vals=[r[k] for r in group]
            line[k+'_mean']=statistics.mean(vals)
            line[k+'_sample_sd']=statistics.stdev(vals) if len(vals)>1 else ''
        out.append(line)
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='output_supplementary_v7')
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--shot',type=int,default=16)
    p.add_argument('--images',type=int,default=128)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--input-seed',type=int,default=2027)
    p.add_argument('--gpu',default='0')
    p.add_argument('--save-matrices',action='store_true',
       help='Save aligned DAVI-vs-Branch Main/Rep/text cosine matrices as NPZ')
    a=p.parse_args(argv)
    if a.images<1 or a.batch<1 or a.shot<1:
        p.error('--images, --batch, and --shot must be positive')
    root=Path(a.output)
    if not root.is_absolute():root=ROOT/root
    out=root/'summary'
    rows=[]
    for dataset in a.datasets:
        for seed in a.seeds:
            b_dir=location(root,'native',dataset,'branch',a.shot,seed)
            d_dir=location(root,'native',dataset,'davi',a.shot,seed)
            print(f'[PAIR] {dataset} seed={seed} (branch then davi)',flush=True)
            branch=_extract(b_dir,a.gpu,a.images,a.batch,a.input_seed)
            davi=_extract(d_dir,a.gpu,a.images,a.batch,a.input_seed)
            row,detail=compare(davi,branch)
            if a.save_matrices:
                write_feature_matrices(out/'ppt_feature_matrices'/f'{dataset}_shot{a.shot}_seed{seed}.npz',davi,branch)
            save_json(out/'ppt_posterior_feature_details'/f'{dataset}_shot{a.shot}_seed{seed}.json',detail)
            rows.append(row)
            print(f'  rep cos={row["rep_cos_mean"]:.5f} '
                  f'visual posterior mean rel-L2={row["r_mean_relative_distance"]:.5f}',flush=True)
    write_csv(out/'ppt_posterior_feature_pairs.csv',rows)
    write_csv(out/'ppt_posterior_feature_summary.csv',summarize(rows))
    print(f'[DONE] {len(rows)} matched pairs in {out}',flush=True)

if __name__=='__main__':
    main()
