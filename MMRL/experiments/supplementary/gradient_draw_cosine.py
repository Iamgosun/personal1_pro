"""Read-only, actual JxJ MC posterior-draw gradient cosine matrices.

Uses the native AVI-Branch classification gradient at the same checkpoint and
train_fixed minibatch. Model weights, augmentations and batch are fixed; only
posterior samples vary. Does NOT compare DAVI vs Branch gradients.

Run: python -m experiments.supplementary.gradient_draw_cosine \
  --output output_supplementary_v7 --datasets caltech101 eurosat oxford_pets \
  --seeds 1 2 3 --shot 16 --gpu 0 --draws 10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .cli import select_e3_fixed_batch
from .core import ROOT, location, restore, cached_encode, cached_draw
from .diagnostics import sampled_ce

GROUPS = {
  'phi': lambda n,p: n.startswith('representation_learner.'),
  'r_mean': lambda n,p: n=='image_encoder.bayes_proj_rep.posterior_mean',
  'r_rho': lambda n,p: n=='image_encoder.bayes_proj_rep.posterior_rho',
  't_rho': lambda n,p: n=='text_posterior.posterior_rho',
}


def evaluate_one(run_dir, gpu, *, draws, batch_size, input_seed, sample_seed):
    trainer,meta,ck=restore(run_dir,gpu=gpu,selection='latest')
    if meta.get('method')!='branch' or meta.get('study')!='native' or meta.get('status')!='complete':
        raise RuntimeError('Expected completed native AVI-Branch run')
    model=trainer.method.model
    model.eval()
    batch,inp=select_e3_fixed_batch(trainer,batch_size,strategy='stratified',selection_seed=input_seed)
    x=batch['img'].to(trainer.method.device)
    y=batch['label'].long().to(x.device)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups={k:[p for n,p in named if predicate(n,p)] for k,predicate in GROUPS.items()}
    if not all(groups.values()):raise RuntimeError('Missing required native group gradients')
    params=[p for ps in groups.values() for p in ps]
    alpha=float(trainer.method.cfg.BAYESRT_MMRL.ALPHA)
    by_group={name:[] for name in groups}
    devices=[torch.cuda.current_device()] if x.is_cuda else []
    with torch.enable_grad(),torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(sample_seed))
        for j in range(draws):
            cache=cached_encode(model,x,trainer.method.num_classes)
            main,rep=cached_draw(cache,1)
            ce=sampled_ce(main,rep,y,alpha)[1].mean()
            gs=torch.autograd.grad(ce,params,allow_unused=True)
            idx=0
            for name,ps in groups.items():
                this=gs[idx:idx+len(ps)];idx+=len(ps)
                v=torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p).float()).reshape(-1)
                        for p,g in zip(ps,this)]).cpu()
                if not torch.isfinite(v).all():raise RuntimeError(f'Nonfinite gradient: {name} draw{j}')
                by_group[name].append(v)
    arrays={}
    for name, vectors in by_group.items():
        m=torch.stack(vectors).float()
        lens=m.norm(dim=1)
        safe=m/lens.clamp_min(1e-12).unsqueeze(1)
        cos=(safe@safe.T).cpu().numpy()
        # Undefined directions are NaN; never imply zero cosine means orthogonality.
        bad=(lens<=1e-12).cpu().numpy()
        cos[bad,:]=np.nan;cos[:,bad]=np.nan
        np.fill_diagonal(cos,np.where(bad,np.nan,1.))
        arrays[name]=cos.astype(np.float32)
    return arrays,dict(dataset=meta['dataset'],shot=meta['shot'],seed=meta['seed'],
         epoch=ck['epoch'],checkpoint_sha256=ck['checkpoint_sha256'],
         input_sha256=inp['input_sha256'],draws=draws,batch_size=batch_size,
         input_seed=input_seed,sample_seed=sample_seed,method='branch',
         gradient_target='branch_mc_classification_only')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='output_supplementary_v7')
    p.add_argument('--datasets',nargs='+',default=['caltech101','eurosat','oxford_pets'])
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--shot',type=int,default=16)
    p.add_argument('--gpu',default='0')
    p.add_argument('--draws',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--input-seed',type=int,default=2027)
    p.add_argument('--sample-seed',type=int,default=99173)
    a=p.parse_args()
    if a.draws<2 or a.batch_size<2:p.error('--draws and --batch-size must be >=2')
    root=Path(a.output)
    if not root.is_absolute():root=ROOT/root
    root=root.resolve()
    dest=root/'summary'/'ppt_gradient_matrices'
    dest.mkdir(parents=True,exist_ok=True)
    for dataset in a.datasets:
        for seed in a.seeds:
            src=location(root,'native',dataset,'branch',a.shot,seed)
            if not src.is_dir():raise FileNotFoundError(src)
            arr,meta=evaluate_one(src,a.gpu,draws=a.draws,batch_size=a.batch_size,
                    input_seed=a.input_seed,sample_seed=a.sample_seed)
            stem=f'{dataset}_shot{a.shot}_seed{seed}_epoch{meta["epoch"]}_J{a.draws}'
            np.savez_compressed(dest/(stem+'.npz'),**arr)
            (dest/(stem+'.json')).write_text(json.dumps(meta,indent=2),encoding='utf-8')
            print('[MATRIX]',dest/(stem+'.npz'),flush=True)
    print('[DONE] Read-only gradient matrices saved to',dest)

if __name__=='__main__':main()
