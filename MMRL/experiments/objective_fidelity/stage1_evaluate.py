"""Phase 1: native-only, checkpoint-read-only objective fidelity evaluator.

COST: deterministic CLIP hidden states once per input batch; only posterior
projection matrices are resampled. Each MC repeat uses independent random
streams and each prefix (S=...) is nested within a repeat. All results saved
under distinct S/R filenames; V5 fidelity.json and checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from argparse import Namespace
from pathlib import Path

from train import MODES, ROOT, output_directory, register_original_mc
from evaluate import (checkpoint_path,verify_loaded_checkpoint,
                      deterministic_reference_loaders,common_moments)
from stage1_gap import GapAccumulator, weighted_aggregate, surrogate
from stage1_protocol import get_grid,mean_se,deterministic_seed
from stage1_cached import encode_once,cached_joint_logits,moments_from_cache,validate_native_parity


def restore_native_checkpoint(a):
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.chdir(ROOT)
    import run as repo_run
    from core.config import setup_cfg
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    from datasets.fair_val_protocol import install_kplusval_datasetbase_patch

    folder=output_directory(a.output_root,'native',a.mode,a.dataset,a.shot,a.seed)
    mf_path=folder/'experiment_manifest.json'
    if not mf_path.is_file():
        raise FileNotFoundError(f'Existing native checkpoint manifest required: {mf_path}')
    mf=json.loads(mf_path.read_text(encoding='utf-8'))
    for key,expected in [('mode',a.mode),('profile','native'),('dataset',a.dataset),
                         ('shot',a.shot),('seed',a.seed)]:
        if mf.get(key)!=expected:
            raise RuntimeError(f'{mf_path}: expected {key}={expected!r}, got {mf.get(key)!r}')
    register_original_mc()
    repo_run._import_runtime_modules()
    ns=Namespace(**{'root':mf['root'],'output_dir':mf['output_dir'],
        'dataset_config_file':mf['dataset_config_file'],
        'method_config_file':mf['method_config_file'],
        'protocol_config_file':mf['protocol_config_file'],
        'runtime_config_file':mf['runtime_config_file'],
        'exp_config':'','method':mf['method'],'protocol':'FS',
        'exec_mode':'online','seed':mf['seed'],'opts':mf['opts'],'trainer':'RefactorRunner'})
    cfg=setup_cfg(ns)
    if hashlib.sha256(cfg.dump().encode()).hexdigest()!=mf['effective_config_sha256']:
        raise RuntimeError('Effective training config hash changed; refusing checkpoint load')
    if str(cfg.METHOD.NAME)!=MODES[a.mode]:
        raise RuntimeError('Resolved model class differs from native checkpoint')
    install_kplusval_datasetbase_patch(cfg)
    set_random_seed(a.seed)
    trainer=build_trainer(cfg)
    checkpoint,epoch=checkpoint_path(folder)
    trainer.load_model(str(folder),epoch=epoch)
    verify_loaded_checkpoint(checkpoint,trainer.method.model)
    trainer.method.model.eval()
    return trainer,folder,mf,checkpoint


def main():
    p=argparse.ArgumentParser(description='DAVI native Stage-1 objective fidelity (checkpoint-only)')
    p.add_argument('--mode',required=True,choices=tuple(MODES))
    p.add_argument('--dataset',required=True)
    p.add_argument('--shot',type=int,default=16)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--gpu',default='0')
    p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--mc-repeats',type=int,default=1,help='Optional independent MC repeats (default 1; no CI with R=1)')
    p.add_argument('--sample-grid',default='auto',help='auto or S1,S2,..,samples (nested within each repeat)')
    p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--reference-batch-size',type=int,default=32)
    p.add_argument('--output-root',default='output_objective_fidelity_v3')
    p.add_argument('--limit-batches',type=int,default=None,help='smoke only')
    p.add_argument('--overwrite',action='store_true',help='explicitly overwrite SAME S/R result')
    p.add_argument('--skip-cache-parity',action='store_true',help='only for debugging; invalidates validation')
    a=p.parse_args()
    if a.samples<2 or a.mc_repeats<1 or a.chunk<1 or a.reference_batch_size<1 or a.shot<1:
        p.error('samples>=2, repeats>=1, chunk>=1, batch>=1, shot>=1')
    if a.limit_batches is not None and a.limit_batches<1:
        p.error('limit-batches >=1')
    grid=get_grid(a.sample_grid,a.samples)
    import torch
    from methods.det_bayesrt_mmrl.losses import det_jensen_ce
    trainer,folder,mf,checkpoint=restore_native_checkpoint(a)
    model=trainer.method.model
    cfg=trainer.method.cfg.BAYESRT_MMRL
    alpha=float(cfg.ALPHA);eta=float(mf['det_diagnostic_scale']);clamp=float(mf['det_diagnostic_clamp'])
    if a.mode=='davi' and (eta,clamp)!=(float(cfg.DET_VAR_SCALE),float(cfg.DET_VAR_CLAMP)):
        raise RuntimeError('DAVI training vs diagnostic variance settings disagree')
    target_dir=folder/'phase1'
    target_dir.mkdir(exist_ok=True)
    suffix='_smoke' if a.limit_batches else ''
    target=target_dir/f'stage1_S{a.samples}_R{a.mc_repeats}{suffix}.json'
    if target.exists() and not a.overwrite:
        raise FileExistsError(f'{target} already exists; use --overwrite to explicitly replace')
    loaders,transform=deterministic_reference_loaders(trainer,a.reference_batch_size)
    result={
        'diagnostic_schema':'stage1-native-3','mode':a.mode,'profile':'native',
        'dataset':a.dataset,'shot':a.shot,'seed':a.seed,'mc_samples':a.samples,
        'mc_repeats':a.mc_repeats,'sample_grid':grid,'alpha':alpha,
        'eta':eta,'clamp':clamp,'checkpoint_path':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'effective_config_sha256':mf['effective_config_sha256'],
        'reference_input_protocol':'fixed_train_and_heldout_test',
        'reference_transform':transform,'diagnostic_partial':a.limit_batches is not None,
        'cache_validated':not a.skip_cache_parity,
        'interpretation':'Primary metrics are O/B/D, Mix=B-O, Det=D-B, Total=D-O; '
            'empirical J is a finite-S log-mgf estimate. MC CI is across repeats; '
            'train_fixed uses deterministic test-style transforms; held-out test is not training ELBO.',
        'splits':{}}
    oldpath=folder/'fidelity.json'
    old=json.loads(oldpath.read_text(encoding='utf-8')) if oldpath.exists() else None
    with torch.no_grad():
        for split,loader in loaders.items():
            batches={s:[[] for _ in range(a.mc_repeats)] for s in grid}
            input_digest=hashlib.sha256()
            parity=None
            auxiliary=[]
            seen=0
            for b_idx,batch in enumerate(loader):
                if a.limit_batches is not None and b_idx>=a.limit_batches:break
                images=batch['img'].to(trainer.method.device)
                labels=batch['label'].long().to(trainer.method.device)
                input_digest.update(images.detach().cpu().contiguous().numpy().tobytes())
                input_digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
                num_classes=trainer.method.num_classes
                cache=encode_once(model,images,num_classes)
                mom=moments_from_cache(cache,
                    moment_eps=float(getattr(model,'det_moment_eps',getattr(cfg,'DET_MOMENT_EPS',1e-6))),
                    tau=float(getattr(model,'logit_scale',getattr(cfg,'LOGIT_SCALE',100.))))
                if b_idx==0 and not a.skip_cache_parity:
                    parity=validate_native_parity(model,images,cache,mom,common_moments,cfg)
                mu_m,vm=mom['mu_main'],mom['var_main']
                mu_r,vr=mom['mu_rep'],mom['var_rep']
                prod=alpha*det_jensen_ce(mu_m,vm,labels,var_scale=eta,var_clamp=clamp) + \
                     (1-alpha)*det_jensen_ce(mu_r,vr,labels,var_scale=eta,var_clamp=clamp)
                local=alpha*surrogate(mu_m,vm,labels,eta=eta,clamp=clamp)+ \
                      (1-alpha)*surrogate(mu_r,vr,labels,eta=eta,clamp=clamp)
                perr=abs(float(prod)-float(local))
                if perr>max(1e-4,5e-5*abs(float(prod))):
                    raise RuntimeError(f'Production DAVI loss parity failed ({split} batch {b_idx}): {perr}')
                auxiliary.append((labels.numel(),{
                    'visual_pre_snr':mom['visual_pre_snr'],
                    'text_pre_snr':mom['text_pre_snr'],
                    'visual_pre_mean_norm_min':mom['visual_pre_mean_norm_min'],
                    'text_pre_mean_norm_min':mom['text_pre_mean_norm_min'],
                    'production_loss_parity_abs':perr}))
                for rep_idx in range(a.mc_repeats):
                    # Fresh independent seed for each checkpoint/seed/input/repeat.
                    # This does not affect model weights or held-out transform.
                    torch.manual_seed(deterministic_seed(a.dataset,a.mode,a.seed,split,b_idx,rep_idx))
                    acc=GapAccumulator(labels,mu_m,vm,mu_r,vr,alpha,eta,clamp)
                    n=0
                    for goal in grid:
                        while n<goal:
                            draw=min(a.chunk,goal-n)
                            lm,lr=cached_joint_logits(cache,draw)
                            acc.add(lm,lr)
                            n+=draw
                        values=acc.snapshot()
                        batches[goal][rep_idx].append((labels.numel(),values))
                seen+=int(labels.numel())
                if b_idx==0 or (b_idx+1)%10==0:
                    print(f'[{split}] {a.mode} batch={b_idx+1} examples={seen} '
                          f'R={a.mc_repeats} S={a.samples}',flush=True)
            if not seen:raise RuntimeError(f'empty {split}')
            sha=input_digest.hexdigest()
            if old and a.limit_batches is None:
                old_sha=old.get(split+'_input_sha256')
                if old_sha and old_sha!=sha:
                    raise RuntimeError(f'{split}: deterministic input hash disagrees with existing V5')
            out={'input_sha256':sha,'num_examples':seen,
                 'cache_parity':parity,
                 'visual_pre_snr':sum(n*x['visual_pre_snr'] for n,x in auxiliary)/seen,
                 'text_pre_snr':sum(n*x['text_pre_snr'] for n,x in auxiliary)/seen,
                 'visual_pre_mean_norm_min':min(x['visual_pre_mean_norm_min'] for _,x in auxiliary),
                 'text_pre_mean_norm_min':min(x['text_pre_mean_norm_min'] for _,x in auxiliary),
                 'production_loss_parity_max_abs':max(x['production_loss_parity_abs'] for _,x in auxiliary),
                 'by_sample_count':{}}
            for s in grid:
                runs=[weighted_aggregate(batches[s][i]) for i in range(a.mc_repeats)]
                summary=mean_se(runs)
                out['by_sample_count'][str(s)]={'repeats':runs,'summary':summary}
                print(f'  {split} S={s}: total={summary["total_gap"]["mean"]:.5g}'
                      f' MAE={summary["mae_total_gap"]["mean"]:.5g} '
                      f'SE={summary["total_gap"]["mc_se"]}',flush=True)
            result['splits'][split]=out
    # Write atomically; an interrupted evaluation leaves no "completed" JSON.
    temp=target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(target)
    print(f'[OK] {target}')


if __name__=='__main__':main()
