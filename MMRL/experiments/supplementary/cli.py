"""CLI for independent native training, offline E1, E2/E3/E4, and bounded E5.

Run from MMRL with python -m experiments.supplementary.cli <command>.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from .core import (ROOT, MODES, EXPERIMENT_NAMES, SCHEMA, location, make_args,
                   setup, restore, checkpoint_path, reference_loader, save_json)


def configure_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.chdir(ROOT)


def _manifest(args, ns, cfg, study, trajectory):
    return {'schema': SCHEMA, 'status':'running', 'dataset':args.dataset,
        'method':args.method, 'experiment_name':EXPERIMENT_NAMES[args.method],
        'model_registry_name':MODES[args.method],
        'config_yaml':str(Path(ns.method_config_file).resolve()),
        'shot':args.shot, 'seed':args.seed, 'output_root':str(Path(args.output).resolve()),
        'data_root':args.data_root, 'study':study,
        'epochs_override':args.epochs, 'mc_train_override':args.mc_train,
        'extra':[], 'trajectory':trajectory, 'init_snapshot':bool(getattr(args,'init_snapshot',False) or trajectory),
        'config_sha256':hashlib.sha256(cfg.dump().encode()).hexdigest(),
        'native_max_epoch':int(cfg.OPTIM.MAX_EPOCH),
        'native_n_mc_train':int(cfg.BAYESRT_MMRL.N_MC_TRAIN),
        'native_n_mc_test':int(cfg.BAYESRT_MMRL.N_MC_TEST),
        'native_r_prior_mode':str(cfg.BAYESRT_MMRL.R_PRIOR_MODE),
        'native_t_train_mean':bool(cfg.BAYESRT_MMRL.T_TRAIN_MEAN),
        'native_fusion':str(cfg.BAYESRT_MMRL.EVAL_FUSION_VARIANT),
        'native_det_var_clamp':(float(cfg.BAYESRT_MMRL.DET_VAR_CLAMP) if args.method=='davi' else None),
        'batch_size':int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE)}


def _effective_cfg(ns):
    """Resolve config without constructing a redundant CLIP model."""
    import run
    from core.config import setup_cfg
    original=ns.method_config_file
    sanitized=run._prepare_method_config_for_kplusval(ns)
    try:
        cfg=setup_cfg(ns)
    finally:
        ns.method_config_file=original
        if sanitized:
            Path(sanitized).unlink(missing_ok=True)
    return cfg


def train_native(args):
    configure_gpu(args.gpu)
    study='native'
    out=location(args.output,study,args.dataset,args.method,args.shot,args.seed)
    if out.exists():
        raise FileExistsError(f'Training output exists, no overwrite: {out}')
    # All native YAML values remain intact unless caller explicitly overrides.
    ns=make_args(args.dataset,args.method,args.shot,args.seed,args.output,study,
                 args.data_root,args.epochs,args.mc_train,trajectory=args.trajectory,
                 init_snapshot=args.init_snapshot)
    from .core import register_original
    import run
    run._import_runtime_modules()
    register_original()
    if args.trajectory or args.init_snapshot:
        from .runner import register_trajectory
        register_trajectory()
    cfg=_effective_cfg(ns)
    if str(cfg.BAYESRT_MMRL.EVAL_FUSION_VARIANT)!='static_logit':
        raise RuntimeError('Incompatible native inference variant')
    out.mkdir(parents=True)
    meta=_manifest(args,ns,cfg,study,args.trajectory)
    save_json(out/'manifest.json',meta)
    (out/'effective_config.yaml').write_text(cfg.dump(),encoding='utf-8')
    start=time.perf_counter()
    # Production run.main uses native TrainerX lifecycle, KL schedule, AMP,
    # optimizer, epochs, validation and final checkpoint selection.
    run.main(ns)
    path=out/'refactor_model'
    if not list(path.glob('*.pth.tar*')):
        raise RuntimeError(f'Native training returned without a model checkpoint in {path}')
    meta.update(status='complete',total_process_seconds=time.perf_counter()-start)
    save_json(out/'manifest.json',meta)
    print(f'[native training complete] {out}')


def _set_train_context(trainer, epoch, num_batches):
    trainer.epoch=epoch
    trainer.max_epoch=int(trainer.cfg.OPTIM.MAX_EPOCH)
    trainer.num_batches=num_batches


def train_budget(args):
    """E5 controlled update budget. Explicitly NOT native TrainerX train() loop.

    Includes DataLoader waiting time, forward, backward and optimizer updates;
    excludes model setup, evaluation, diagnostics and checkpoint I/O.
    Scheduler, KL and AMP use production trainer.forward_backward exactly.
    Must actually reach the budget, else run is marked incomplete and rejected.
    """
    configure_gpu(args.gpu)
    import torch
    if args.method not in ('davi','branch'):
        raise ValueError('E5 only compares DAVI vs AVI-Branch')
    if args.budget_seconds <= 0:
        raise ValueError('E5 requires positive --budget-seconds')
    study='matched_time'
    out=location(args.output,study,args.dataset,args.method,args.shot,args.seed)
    if out.exists():raise FileExistsError(f'Refuse overwrite: {out}')
    ns=make_args(args.dataset,args.method,args.shot,args.seed,args.output,study,
                 args.data_root,args.epochs,args.mc_train)
    trainer,cfg=setup(ns)
    out.mkdir(parents=True)
    meta=_manifest(args,ns,cfg,study,False)
    meta['budget_seconds']=args.budget_seconds
    meta['budget_clock']='synchronized training wall-clock from before first DataLoader iteration; includes data wait'
    save_json(out/'manifest.json',meta)
    (out/'effective_config.yaml').write_text(cfg.dump(),encoding='utf-8')
    trainer.model.train()
    n_batches=len(trainer.dm.train_loader_x)
    if n_batches<=0:raise RuntimeError('Empty training loader')
    steps=0;history=[]; stop=False;completed_epochs=0;last_step_seconds=None
    if torch.cuda.is_available():torch.cuda.synchronize()
    began=time.perf_counter()
    for epoch in range(int(cfg.OPTIM.MAX_EPOCH)):
        _set_train_context(trainer,epoch,n_batches)
        trainer.set_model_mode('train')
        if hasattr(trainer,'before_epoch'):
            trainer.before_epoch()
        # Explicit iterator allows last-step duration to include data wait.
        iterator=iter(trainer.dm.train_loader_x)
        index=0
        while True:
            step_start=time.perf_counter()
            try:batch=next(iterator)
            except StopIteration:break
            trainer.batch_idx=index
            result=trainer.forward_backward(batch)
            if torch.cuda.is_available():torch.cuda.synchronize()
            last_step_seconds=time.perf_counter()-step_start
            steps+=1;index+=1
            elapsed=time.perf_counter()-began
            if steps==1 or steps%args.log_every==0:
                history.append({'epoch':epoch+1,'updates':steps,'train_wall_seconds':elapsed,
                                'loss':float(result['loss']),'accuracy':float(result['acc'])})
            if elapsed >= args.budget_seconds:
                stop=True
                break
        if not stop:completed_epochs+=1
        if stop:break
    if torch.cuda.is_available():torch.cuda.synchronize()
    elapsed=time.perf_counter()-began
    # Native lightweight saver, explicit last-update checkpoint (not 'best').
    trainer.save_model(epoch,str(out),is_best=False)
    meta.update(status='complete' if stop else 'budget_not_reached',
                updates=steps,elapsed_training_seconds=elapsed,
                completed_epochs=completed_epochs,last_update_checkpoint_epoch=epoch+1,
                last_step_seconds=last_step_seconds)
    save_json(out/'manifest.json',meta)
    save_json(out/'training.json',{'history':history,'budget_seconds':args.budget_seconds,
        'elapsed_training_seconds':elapsed,'updates':steps,
        'budget_reached':stop,'overshoot_seconds':max(0,elapsed-args.budget_seconds),
        'last_step_seconds':last_step_seconds,'last_checkpoint_epoch':epoch+1,
        'completed_epochs':completed_epochs,
        'last_checkpoint_selection':'latest (last optimizer update, NOT val-best)'})
    if not stop:
        raise RuntimeError('E5 run did not consume budget before native epoch cap; reduce budget or explicitly change epochs. Invalid for matched-time table.')
    print(f'[E5] {args.method} updates={steps} train_wall={elapsed:.3f}s budget={args.budget_seconds:.3f}s')


def _rows_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def _reference_eval(args, trainer, meta, ck):
    from .diagnostics import reference_diagnostics, posterior_summary, gradient_quality
    import torch
    cfg=trainer.method.cfg.BAYESRT_MMRL
    # DAVI uses its native scale/clamp. AVI has NO deterministic training loss:
    # optional --counterfactual-davi computes a labeled DAVI surrogate there.
    if meta['method']=='davi':
        davi_params=(float(cfg.DET_VAR_SCALE),float(cfg.DET_VAR_CLAMP))
        tag='native_davi'
    elif args.counterfactual_davi:
        import yaml
        data=yaml.safe_load((ROOT/'configs/methods/det_bayesrt_mmrl_clamp.yaml').read_text(encoding='utf-8'))
        spec=data['BAYESRT_MMRL']
        davi_params=(float(spec['DET_VAR_SCALE']),float(spec['DET_VAR_CLAMP']))
        tag='counterfactual_davi_on_avi'
    else:
        davi_params=None
        tag='native_avi_mc_only'
    out=Path(args.run_dir)/'diagnostics'/f'{ck["selection"]}'
    if ck['epoch'] is not None:
        out=out/f'epoch_{ck["epoch"]}'
    out=out/f'S{args.samples}_R{args.mc_repeats}'
    if args.smoke:
        out=out/'smoke'
    if out.exists() and not args.overwrite:
        raise FileExistsError('Diagnostics exist: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    result={}
    # An initialization snapshot is optional; never fabricate prompt-change values.
    init_state=None
    initial=Path(args.run_dir)/'trajectory'/'refactor_model'/'model.pth.tar-0'
    if initial.exists():
        init_state=torch.load(initial,map_location='cpu',weights_only=False)['state_dict']
    for split in args.splits:
        loader=reference_loader(trainer,split,args.batch_size)
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        repeat_summaries=[];first_rows=None;first_vectors=None
        for repeat in range(args.mc_repeats):
            # Independent posterior reference draws. Evaluation cannot advance the
            # native training RNG even if invoked inside an outer process.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(40300+int(meta['seed'])+(0 if split=='test' else 100)+100003*repeat)
                summary,rows,vectors=reference_diagnostics(
                    trainer,loader,args.samples,args.chunk,args.limit_batches,
                    davi_params,args.vectors and repeat==0,args.parity and repeat==0)
            repeat_summaries.append(summary)
            if repeat==0:first_rows,first_vectors=rows,vectors
            elif summary['input_sha256']!=repeat_summaries[0]['input_sha256']:
                raise RuntimeError('MC repeats do not use identical fixed reference inputs')
        summary=dict(repeat_summaries[0]);summary['mc_repeats']=args.mc_repeats
        if args.mc_repeats>1:
            import statistics as stats
            numeric=['original_nll','branch_nll','gap1']+(['det_nll','gap2'] if davi_params else [])
            summary['mc_repeat_sd']={k:stats.stdev(v[k] for v in repeat_summaries) for k in numeric}
            summary['mc_repeat_spread']={k:max(v[k] for v in repeat_summaries)-min(v[k] for v in repeat_summaries) for k in numeric}
            summary['mc_reference_stable']=all(summary['mc_repeat_spread'][k] <= args.mc_tolerance for k in numeric)
            if not summary['mc_reference_stable'] and not args.smoke:
                raise RuntimeError('MC reference repeats vary by more than --mc-tolerance; increase --samples or report instability')
        else:
            summary['mc_reference_stable']=None  # not tested; do not claim convergence
        summary.update({'dataset':meta['dataset'],'method':meta['method'],
                        'shot':meta['shot'],'seed':meta['seed'],'split':split,
                        'objective_type':tag,'partial':args.smoke,
                        'checkpoint':ck,'native_predictive':False,
                        'diagnostic_transform':'fixed test-style transform'})
        summary.update(posterior_summary(trainer.method,init_state))
        sub=out/split
        sub.mkdir(exist_ok=True)
        save_json(sub/'metrics.json',summary)
        _rows_csv(sub/'per_image.csv',first_rows)
        if args.mc_repeats>1:
            save_json(sub/'independent_mc_repeats.json',repeat_summaries)
        if args.vectors and first_vectors:
            save_json(sub/'moment_vectors.json',first_vectors)
        result[split]=summary
        print(f'[reference {split}] {sub} N={len(first_rows)} S={args.samples} repeats={args.mc_repeats}')
    if args.gradient:
        if meta['method']!='davi':raise ValueError('Gradient E3 needs DAVI model')
        if args.limit_batches is not None and not args.smoke:
            raise RuntimeError('Partial gradient requires smoke flag')
        batch=next(iter(reference_loader(trainer,'train_fixed',args.batch_size)))
        grad=gradient_quality(trainer,batch,reference_samples=args.grad_ref,
            chunk=args.grad_chunk,repeats=args.grad_repeats)
        if not grad['reference_stable'] and not args.smoke:
            raise RuntimeError('E3 MC reference unstable: increase --grad-ref before paper use')
        save_json(out/'gradient.json',{**grad,'dataset':meta['dataset'],
            'method':meta['method'],'seed':meta['seed'],'shot':meta['shot'],
            'checkpoint':ck,'fixed_train_augmentation':False})
    return result


def evaluate(args):
    configure_gpu(args.gpu)
    from .diagnostics import native_predictions
    trainer,meta,ck=restore(args.run_dir,args.gpu,args.checkpoint,args.epoch)
    if meta.get('status')!='complete':
        raise RuntimeError('Refusing to evaluate incomplete training run')
    if meta['study']=='matched_time' and args.checkpoint!='latest':
        raise RuntimeError('E5 is last-update checkpoint only: --checkpoint latest')
    if args.limit_batches is not None and not args.smoke:
        raise ValueError('--limit-batches requires --smoke')
    if args.mode in ('both','reference'):
        _reference_eval(args,trainer,meta,ck)
    if args.mode in ('both','formal'):
        out=Path(args.run_dir)/'formal'/ck['selection']
        if ck['epoch'] is not None:
            out=out/f'epoch_{ck["epoch"]}'
        out=out/('smoke' if args.smoke else 'complete')
        if out.exists() and not args.overwrite:
            raise FileExistsError('Formal report exists: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        # Uses N_MC_TEST from the native YAML (not --samples high-S reference).
        formal=native_predictions(trainer,split='test',limit=args.limit_batches)
        formal.update({'dataset':meta['dataset'],'method':meta['method'],
            'shot':meta['shot'],'seed':meta['seed'],'partial':args.smoke,
            'checkpoint':ck, 'source':'native forward_eval + select_eval_logits; formal FS'})
        save_json(out/'test.json',formal)
        print(f'[formal test] {out} N_MC_TEST={formal["n_mc_test"]}')



def doctor(args):
    """GPU environment check, no optimizer step and no checkpoint mutation."""
    configure_gpu(args.gpu)
    import torch
    from .core import (cached_encode,cached_draw,validate_cache_parity,native_moments,
                       register_original)
    from .diagnostics import det_ce_per_image, sampled_ce
    ns=make_args(args.dataset,args.method,args.shot,args.seed,args.output,
                 'doctor',args.data_root,None,None)
    trainer,cfg=setup(ns)
    method=trainer.method
    model=method.model
    r=model.image_encoder.bayes_proj_rep
    t=model.text_posterior
    if r is None or t is None:raise RuntimeError('Missing asymmetric posterior')
    assert str(r.sigma_mode)=='output' and str(t.sigma_mode)=='output'
    assert not t.train_mean and not t.posterior_mean.requires_grad
    if args.method=='davi':
        assert str(cfg.BAYESRT_MMRL.R_PRIOR_MODE)=='random_mmrl'
        assert torch.count_nonzero(r.prior_mean)==0
        assert float(cfg.BAYESRT_MMRL.DET_VAR_CLAMP)==50.0
    else:
        assert str(cfg.BAYESRT_MMRL.R_PRIOR_MODE)=='zero'
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not any(n.startswith('representation_learner.') for n,p in trainable):
        raise RuntimeError('Shared representation learner is not trainable')
    batch=next(iter(trainer.dm.train_loader_x))
    x=batch['img'][:args.batch_size].to(method.device)
    y=batch['label'][:args.batch_size].long().to(method.device)
    if len(y)==0:raise RuntimeError('Empty train batch')
    if torch.cuda.is_available():torch.cuda.synchronize()
    # Model stays trainable (no optimizer update); native forward_train includes KL+alignment.
    with torch.autocast(device_type=x.device.type,enabled=(str(cfg.BAYESRT_MMRL.PREC)=='amp' and x.is_cuda)):
        outputs=method.forward_train({'img':x,'label':y})
    total=method.loss(outputs)
    if not torch.isfinite(total):raise RuntimeError('Nonfinite native training loss')
    grads=torch.autograd.grad(total,[p for _,p in trainable],allow_unused=True)
    nonzero=sum(1 for g in grads if g is not None and torch.isfinite(g).all() and g.abs().sum()>0)
    if nonzero==0:raise RuntimeError('No usable gradients in native training')
    model.eval()
    with torch.no_grad():
        cache=cached_encode(model,x,method.num_classes)
        parity=validate_cache_parity(model,x,method.num_classes,cache)
        if args.method=='davi':
            mom=native_moments(model,x,cfg.BAYESRT_MMRL)
            det=(float(cfg.BAYESRT_MMRL.ALPHA)*det_ce_per_image(
                mom['mu_main'],mom['var_main'],y,1.,50.).mean()+
                (1-float(cfg.BAYESRT_MMRL.ALPHA))*det_ce_per_image(
                mom['mu_rep'],mom['var_rep'],y,1.,50.).mean())
            from methods.det_bayesrt_mmrl.losses import det_jensen_ce
            production=(float(cfg.BAYESRT_MMRL.ALPHA)*det_jensen_ce(
                mom['mu_main'],mom['var_main'],y,var_scale=1.,var_clamp=50.)+
                (1-float(cfg.BAYESRT_MMRL.ALPHA))*det_jensen_ce(
                mom['mu_rep'],mom['var_rep'],y,var_scale=1.,var_clamp=50.))
            if not torch.allclose(det,production,atol=1e-4,rtol=5e-5):
                raise RuntimeError('DAVI diagnostic/native DetCE disagreement')
        else:
            native=model.forward_joint(x,3,use_posterior_mean=False)
            cc=native['logits_main_stack'][...,:method.num_classes]
            rr=native['logits_rep_stack'][...,:method.num_classes]
            o,b,_,_,_=sampled_ce(cc,rr,y,float(cfg.BAYESRT_MMRL.ALPHA))
            if (b-o).min()<-1e-5:raise RuntimeError('Jensen failure in real model samples')
    result={'schema':2,'model':args.method,'registry_method':str(cfg.METHOD.NAME),
            'training_loss':float(total.detach()),'nonzero_trainable_grad_tensors':nonzero,
            'trainable_tensors':len(trainable),'native_sample_cache_max_abs':parity,
            'r_prior_mode':str(cfg.BAYESRT_MMRL.R_PRIOR_MODE),
            'r_prior_mean_norm':float(r.prior_mean.norm()),
            't_train_mean':bool(t.train_mean),
            'n_mc_train_config':int(cfg.BAYESRT_MMRL.N_MC_TRAIN),
            'n_mc_test_config':int(cfg.BAYESRT_MMRL.N_MC_TEST),
            'fusion':str(cfg.BAYESRT_MMRL.EVAL_FUSION_VARIANT)}
    print(json.dumps(result,ensure_ascii=False,indent=2))


def trace(args):
    """E1 offline snapshots; never require a native model-best checkpoint.

    Use the snapshots that actually exist, even if epochs are missing or this
    is a short pilot. No diagnostic forward occurs in native training.
    """
    configure_gpu(args.gpu)
    import torch
    from .core import cached_encode, cached_draw, verify_checkpoint, available_trajectory_epochs
    from .diagnostics import sampled_ce, native_predictions
    root = Path(args.run_dir).resolve()
    manifest_path = root/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Missing E1 manifest: '+str(manifest_path))
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not meta.get('trajectory'):
        raise RuntimeError('E1 needs a run trained with --trajectory, not a best-val checkpoint')
    available = available_trajectory_epochs(root)
    if not available:
        raise FileNotFoundError('No E1 epoch snapshots under '+str(root/'trajectory'/'refactor_model'))
    # Even epoch 0 alone can be inspected during an initialization pilot.
    # No model-best file, full 0..MAX_EPOCH sequence or three seeds are required.
    trainer, restored_meta, _ = restore(root, args.gpu, 'trajectory', available[0])
    if meta.get('status') != 'complete':
        print('[E1] Run is not marked complete: calculating available snapshots only', flush=True)
    if available[0] != 0 or len(available) != available[-1]-available[0]+1:
        print('[E1] Non-contiguous trajectory: using available epochs '+str(available), flush=True)
    if trainer.val_loader is None:
        print('[E1] No validation loader: responsibility will be saved; Rep val ACC unavailable', flush=True)
    records = []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    for epoch in available:
        path = root/'trajectory'/'refactor_model'/f'model.pth.tar-{epoch}'
        trainer.load_model(str(root/'trajectory'), epoch=epoch)
        model = trainer.method.model
        model.eval()
        verify_checkpoint(model, path)
        rho = []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(900000 + int(restored_meta['seed']))
            for bi, batch in enumerate(reference_loader(trainer, 'train_fixed', args.batch_size)):
                if args.e1_batches and bi >= args.e1_batches:
                    break
                x = batch['img'].to(trainer.device)
                y = batch['label'].long().to(trainer.device)
                c, r = cached_draw(cached_encode(model, x, trainer.method.num_classes), args.e1_samples)
                rho.extend(sampled_ce(c, r, y,
                    float(trainer.method.cfg.BAYESRT_MMRL.ALPHA))[-1].mean(0).cpu().tolist())
            if not rho:
                raise RuntimeError('E1 trajectory has no fixed-training images')
            entry = {'epoch':epoch, 'dataset':restored_meta['dataset'],
                     'method':restored_meta['method'], 'seed':restored_meta['seed'],
                     'shot':restored_meta['shot'],
                     'rho_rep_median':statistics.median(rho),
                     'rho_rep_mean':statistics.mean(rho),
                     'rho_rep_p10':float(torch.quantile(torch.tensor(rho), .1)),
                     'rho_rep_p90':float(torch.quantile(torch.tensor(rho), .9)),
                     'rho_source':'train_fixed, posterior draws',
                     'rho_n_images':len(rho), 'e1_samples':args.e1_samples,
                     'e1_batches':args.e1_batches, 'batch_size':args.batch_size,
                     'branch_weight':1-float(trainer.method.cfg.BAYESRT_MMRL.ALPHA)}
            if trainer.val_loader is not None:
                native = native_predictions(trainer, split='val')
                entry['rep_val_acc'] = native['rep_accuracy']
                entry['validation_n'] = native['n_images']
                entry['rep_val_status'] = 'available'
            else:
                entry['rep_val_acc'] = None
                entry['validation_n'] = 0
                entry['rep_val_status'] = 'no_val_loader'
        records.append(entry)
        print(f'[E1] epoch={epoch} rho_rep_median={entry["rho_rep_median"]:.4f}', flush=True)
    out = root/'trace'
    out.mkdir(exist_ok=True)
    _rows_csv(out/'trajectory.csv', records)
    save_json(out/'trajectory.json', {'run':str(root), 'records':records,
        'reference_protocol':'train_fixed / native val (if available)',
        'e1_samples':args.e1_samples, 'e1_batches':args.e1_batches,
        'batch_size':args.batch_size, 'epochs_evaluated':available,
        'rep_val_status':'available' if trainer.val_loader is not None else 'no_val_loader'})


def summarize(args):
    """Produce independent S1--S4 tables; E5 includes validated pairs only."""
    from .reporting import make_tables
    root=Path(args.output)
    if not root.is_absolute():root=ROOT/root
    print(json.dumps(make_tables(root),indent=2,ensure_ascii=False))


def main(argv=None):
    p=argparse.ArgumentParser(description='Native DAVI-Adapter V7 supplementary experiments')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train',help='Native run.py training with optional E1 epoch snapshots')
    b=sub.add_parser('budget',help='E5 matched synchronized training wall clock')
    e=sub.add_parser('evaluate',help='Read-only reference and/or native formal inference')
    q=sub.add_parser('trace',help='Offline E1 trajectory from native per-epoch snapshots')
    s=sub.add_parser('summarize',help='Separate method/study/reference count summaries')
    d=sub.add_parser('doctor',help='GPU/native one-batch forward, gradient, posterior, logits parity checks')
    for x in (t,b,e,q,d):x.add_argument('--gpu',default='0')
    for x in (t,b):
        x.add_argument('--dataset',required=True,choices=('caltech101','eurosat','fgvc_aircraft'))
        x.add_argument('--method',required=True,choices=MODES)
        x.add_argument('--shot',type=int,default=16)
        x.add_argument('--seed',type=int,default=1)
        x.add_argument('--output',default='output_supplementary_v7')
        x.add_argument('--data-root',default='DATASETS')
        x.add_argument('--epochs',type=int,help='Optional override; native YAML when omitted')
        x.add_argument('--mc-train',type=int,help='Optional override; native YAML when omitted')
    for opt in ('--dataset','--method'):
        d.add_argument(opt,required=True,choices=('caltech101','eurosat','fgvc_aircraft') if opt=='--dataset' else MODES)
    d.add_argument('--shot',type=int,default=16)
    d.add_argument('--seed',type=int,default=1)
    d.add_argument('--output',default='output_supplementary_v7')
    d.add_argument('--data-root',default='DATASETS')
    d.add_argument('--batch-size',type=int,default=2)
    t.add_argument('--trajectory',action='store_true',help='Save initialization + every epoch snapshot for E1')
    t.add_argument('--init-snapshot',action='store_true',help='Save only epoch zero to measure E4 prompt change, no epoch history')
    b.add_argument('--budget-seconds',type=float,required=True)
    b.add_argument('--log-every',type=int,default=20)
    for x in (e,q):
        x.add_argument('--run-dir',required=True)
        x.add_argument('--batch-size',type=int,default=32)
    e.add_argument('--checkpoint',choices=('best','latest','trajectory'),default='best')
    e.add_argument('--epoch',type=int)
    e.add_argument('--mode',choices=('reference','formal','both'),default='both')
    e.add_argument('--samples',type=int,default=1024)
    e.add_argument('--chunk',type=int,default=16)
    e.add_argument('--mc-repeats',type=int,default=1,help='Independent S_ref draws on each fixed input; 1 means no MC precision estimate')
    e.add_argument('--mc-tolerance',type=float,default=0.03,help='Maximum repeat spread in NLL (nats/image) for paper outputs')
    e.add_argument('--splits',nargs='+',choices=('train_fixed','test'),default=['test'])
    e.add_argument('--vectors',action='store_true')
    e.add_argument('--gradient',action='store_true')
    e.add_argument('--grad-ref',type=int,default=512)
    e.add_argument('--grad-chunk',type=int,default=8)
    e.add_argument('--grad-repeats',type=int,default=5)
    e.add_argument('--counterfactual-davi',action='store_true',help='Optional DAVI surrogate on AVI posterior; NOT AVI training loss')
    e.add_argument('--no-parity',dest='parity',action='store_false',default=True,help='Debug only, never use for paper results')
    e.add_argument('--limit-batches',type=int)
    e.add_argument('--smoke',action='store_true')
    e.add_argument('--overwrite',action='store_true',help='Explicitly replace matching diagnostic output')
    q.add_argument('--e1-samples',type=int,default=10)
    q.add_argument('--e1-batches',type=int,default=4)
    s.add_argument('--output',default='output_supplementary_v7')
    a=p.parse_args(argv)
    if a.command in ('train','budget'):
        if a.shot<1 or a.seed<0 or (a.epochs is not None and a.epochs<1) or (a.mc_train is not None and a.mc_train<1):
            p.error('Invalid shot, seed, epoch or MC training count')
    if a.command=='evaluate':
        if a.samples<2 or a.chunk<1 or a.grad_ref<2 or a.grad_chunk<1 or a.grad_repeats<1 or a.mc_repeats<1 or a.mc_tolerance<=0:
            p.error('Invalid reference/gradient sample count')
        if a.limit_batches is not None and (a.limit_batches<1 or not a.smoke):
            p.error('--limit-batches >=1 requires --smoke')
        if not a.parity and not a.smoke:
            p.error('--no-parity is strictly smoke-only')
    if a.command=='train':train_native(a)
    elif a.command=='budget':
        if a.log_every<1:p.error('log-every must >=1')
        train_budget(a)
    elif a.command=='evaluate':evaluate(a)
    elif a.command=='trace':
        if a.e1_samples<1 or a.e1_batches<0 or a.batch_size<1:p.error('E1 samples/batch size must be positive; E1 batches >=0 (0 means all)')
        trace(a)
    elif a.command=='doctor':doctor(a)
    else:summarize(a)

if __name__=='__main__':
    main()
