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
        'extra':[], 'trajectory':trajectory, 'init_snapshot':bool(getattr(args,'init_snapshot',False) or trajectory or getattr(args,'e3_snapshot_epochs',[])),
        'e3_snapshot_epochs':list(getattr(args,'e3_snapshot_epochs',[])),
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
                 init_snapshot=args.init_snapshot,
                 e3_snapshot_epochs=tuple(args.e3_snapshot_epochs))
    from .core import register_original
    import run
    run._import_runtime_modules()
    register_original()
    if args.trajectory or args.init_snapshot or args.e3_snapshot_epochs:
        from .runner import register_trajectory
        register_trajectory()
    cfg=_effective_cfg(ns)
    if args.e3_snapshot_epochs and max(args.e3_snapshot_epochs)>int(cfg.OPTIM.MAX_EPOCH):
        raise ValueError('Requested E3 checkpoint exceeds native training epochs: '+
            f'{max(args.e3_snapshot_epochs)} > {int(cfg.OPTIM.MAX_EPOCH)}')
    if str(cfg.BAYESRT_MMRL.EVAL_FUSION_VARIANT)!='static_logit':
        raise RuntimeError('Incompatible native inference variant')
    out.mkdir(parents=True)
    meta=_manifest(args,ns,cfg,study,args.trajectory)
    save_json(out/'manifest.json',meta)
    (out/'effective_config.yaml').write_text(cfg.dump(),encoding='utf-8')
    start=time.perf_counter()
    # Production run.main uses native TrainerX lifecycle, KL schedule, AMP,
    # optimizer, epochs, validation and final checkpoint selection.
    # Only requested E3 epochs are snapshotted; the environment value is local
    # to this training process and is never needed for checkpoint restoration.
    old_e3 = os.environ.get('DAVI_SUPPLEMENT_E3_SAVE_EPOCHS')
    try:
        if args.e3_snapshot_epochs:
            os.environ['DAVI_SUPPLEMENT_E3_SAVE_EPOCHS'] = ','.join(map(str,args.e3_snapshot_epochs))
        else:
            os.environ.pop('DAVI_SUPPLEMENT_E3_SAVE_EPOCHS',None)
        run.main(ns)
    finally:
        if old_e3 is None:
            os.environ.pop('DAVI_SUPPLEMENT_E3_SAVE_EPOCHS',None)
        else:
            os.environ['DAVI_SUPPLEMENT_E3_SAVE_EPOCHS'] = old_e3
    path=out/'refactor_model'
    if not list(path.glob('*.pth.tar*')):
        raise RuntimeError(f'Native training returned without a model checkpoint in {path}')
    meta.update(status='complete',total_process_seconds=time.perf_counter()-start)
    save_json(out/'manifest.json',meta)
    print(f'[native training complete] {out}')


def _set_train_context(trainer, epoch, num_batches):
    # Mirror Dassl TrainerX.run_epoch + native OnlineExecutor epoch context.
    trainer.epoch = epoch
    trainer.max_epoch = int(trainer.cfg.OPTIM.MAX_EPOCH)
    trainer.num_batches = num_batches


def budget_training_updates(trainer, *, budget_seconds, log_every=20,
                            clock=None, synchronize=None, snapshot_callback=None):
    """E5: native optimizer updates up to a synchronized training-time budget.

    TrainerX.after_epoch() deliberately NOT invoked: it does selection,
    validation and checkpoint writes, all outside the E5 training-time clock.
    OnlineExecutor.forward_backward() controls native AMP, optimizer, KL,
    and calls update_lr() on the *last batch* of each completed epoch.
    A partial epoch never triggers that final-epoch LR step.

    clock/synchronize are injectable for exact CPU tests of timing boundaries.
    """
    import time
    clock = clock or time.perf_counter
    synchronize = synchronize or (lambda: None)
    if budget_seconds <= 0 or not (budget_seconds < float('inf')):
        raise ValueError('E5 budget_seconds must be finite and positive')
    if log_every < 1:
        raise ValueError('log_every must be a positive integer')
    loader = trainer.dm.train_loader_x
    num_batches = len(loader)
    if num_batches < 1:
        raise RuntimeError('E5 empty training loader')
    synchronize()
    # The diagnostic snapshots are excluded from elapsed training time, not
    # counted as optimizer updates, and never sample or evaluate the model.
    saved_phases=[]
    if snapshot_callback is not None:
        snapshot_callback(0,0.0,0,None,None)
        saved_phases.append(0)
    started = clock()
    excluded_snapshot_seconds = 0.0
    steps = 0
    completed_epochs = 0
    stopped = False
    history = []
    last_step_seconds = None
    last_update_epoch = None
    last_update_batch = None
    for epoch in range(int(trainer.cfg.OPTIM.MAX_EPOCH)):
        _set_train_context(trainer, epoch, num_batches)
        trainer.set_model_mode('train')
        trainer.before_epoch()
        iterator = iter(loader)
        count = 0
        for idx in range(num_batches):
            # Includes time waiting for the DataLoader and the complete update.
            step_start = clock()
            try:
                batch = next(iterator)
            except StopIteration as e:
                raise RuntimeError('DataLoader ended before its declared length') from e
            trainer.batch_idx = idx
            result = trainer.forward_backward(batch)
            synchronize()
            last_step_seconds = max(0.0, clock() - step_start)
            steps += 1
            count += 1
            last_update_epoch = epoch + 1
            last_update_batch = idx + 1
            elapsed = clock() - started - excluded_snapshot_seconds
            if snapshot_callback is not None:
                for phase in (25,50,75,100):
                    if phase not in saved_phases and elapsed >= budget_seconds*phase/100.0:
                        before_snapshot = clock()
                        snapshot_callback(phase,elapsed,steps,epoch+1,idx+1)
                        synchronize()
                        excluded_snapshot_seconds += max(0.0,clock()-before_snapshot)
                        saved_phases.append(phase)
            if steps == 1 or steps % log_every == 0:
                history.append(dict(epoch=epoch + 1, epoch_batch=idx + 1,
                    updates=steps, train_wall_seconds=elapsed,
                    loss=float(result['loss']), accuracy=float(result['acc'])))
            if elapsed >= budget_seconds:
                stopped = True
                break
        if count == num_batches:
            completed_epochs += 1
        if stopped:
            break
    synchronize()
    elapsed = clock() - started - excluded_snapshot_seconds
    return dict(time_curve_phases=saved_phases,
        excluded_snapshot_seconds=excluded_snapshot_seconds,history=history, budget_seconds=float(budget_seconds),
        elapsed_training_seconds=elapsed, updates=steps,
        budget_reached=stopped, overshoot_seconds=max(0., elapsed-budget_seconds),
        last_step_seconds=last_step_seconds,
        last_checkpoint_epoch=last_update_epoch,
        last_update_batch_in_epoch=last_update_batch,
        completed_epochs=completed_epochs,
        partial_epoch=(last_update_epoch is not None and last_update_epoch>completed_epochs),
        last_checkpoint_selection='latest (last optimizer update, NOT val-best)',
        excluded_lifecycle_hook='TrainerX.after_epoch (validation/checkpoint; no training update)',
        lr_scheduler_source='native OnlineExecutor.forward_backward last-batch update_lr')


def train_budget(args):
    """E5 matched-time native *updates*, not full TrainerX validation lifecycle."""
    configure_gpu(args.gpu)
    import torch
    from .core import sha256
    if args.method not in ('davi', 'branch'):
        raise ValueError('E5 only compares DAVI vs AVI-Branch')
    study = 'matched_time'
    out = location(args.output, study, args.dataset, args.method, args.shot, args.seed)
    if out.exists():
        raise FileExistsError(f'E5 will not overwrite an existing run: {out}')
    ns = make_args(args.dataset,args.method,args.shot,args.seed,args.output,study,
                   args.data_root,args.epochs,args.mc_train)
    trainer,cfg = setup(ns)
    out.mkdir(parents=True)
    meta = _manifest(args,ns,cfg,study,False)
    meta['budget_seconds'] = args.budget_seconds
    meta['e5_time_curve_requested'] = bool(args.time_curve)
    meta['budget_clock'] = ('synchronized training wall-clock; includes DataLoader wait, '
                            'native forward/backward, optimizer, native LR, excludes '
                            'model setup, validation, diagnostics and checkpoint I/O')
    meta['train_optimizer_name'] = str(cfg.OPTIM.NAME)
    meta['train_lr'] = float(cfg.OPTIM.LR)
    meta['train_lr_scheduler'] = str(cfg.OPTIM.LR_SCHEDULER)
    meta['train_warmup_epoch'] = int(cfg.OPTIM.WARMUP_EPOCH)
    meta['train_precision'] = str(cfg.BAYESRT_MMRL.PREC)
    meta['torch_version'] = torch.__version__
    meta['cuda_device'] = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    save_json(out/'manifest.json',meta)
    (out/'effective_config.yaml').write_text(cfg.dump(),encoding='utf-8')
    trainer.model.train()
    clock_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    # Snapshots are taken *after* a complete native optimizer update. Phase 0
    # is the genuine initialized native state. Checkpoint writes are excluded
    # from training budget and do not advance torch RNG.
    checkpoints=[]
    def snapshot(phase, elapsed, updates, epoch, batch):
        trainer.save_model((epoch or 0)-1,str(out/'time_curve'),is_best=False,
                           model_name=f'model.pth.tar-{phase}')
        cp=out/'time_curve'/'refactor_model'/f'model.pth.tar-{phase}'
        checkpoints.append({'phase':phase,'train_wall_seconds':elapsed,
           'updates':updates,'real_epoch':epoch,'batch_in_epoch':batch,
           'checkpoint_sha256':sha256(cp),'path':str(cp)})
    info = budget_training_updates(trainer,budget_seconds=args.budget_seconds,
            log_every=args.log_every,synchronize=clock_sync,
            snapshot_callback=snapshot if args.time_curve else None)
    info['time_curve_checkpoints']=checkpoints
    epoch = info['last_checkpoint_epoch']
    if epoch is None:
        raise RuntimeError('E5 produced no optimizer updates')
    # Lightweight native checkpoint, outside the timed region.
    trainer.save_model(epoch-1,str(out),is_best=False)
    ckpath = out/'refactor_model'/f'model.pth.tar-{epoch}'
    if not ckpath.is_file():
        raise FileNotFoundError(f'E5 budget checkpoint not saved: {ckpath}')
    info['last_checkpoint_sha256'] = sha256(ckpath)
    info['checkpoint_path'] = str(ckpath)
    meta.update(status='complete' if info['budget_reached'] else 'budget_not_reached',
                updates=info['updates'],elapsed_training_seconds=info['elapsed_training_seconds'],
                completed_epochs=info['completed_epochs'],
                last_update_checkpoint_epoch=epoch,
                last_update_batch_in_epoch=info['last_update_batch_in_epoch'],
                partial_epoch=info['partial_epoch'],
                last_checkpoint_sha256=info['last_checkpoint_sha256'],
                last_step_seconds=info['last_step_seconds'])
    save_json(out/'training.json',info)
    save_json(out/'manifest.json',meta)
    if not info['budget_reached']:
        print('[E5 WARNING] Epoch cap reached before requested budget. '
              'Saved incomplete pilot; it will be rejected from Table S4.',flush=True)
    print(f'[E5] {args.method} updates={info["updates"]} '
          f'train_wall={info["elapsed_training_seconds"]:.3f}s '
          f'budget={args.budget_seconds:.3f}s '
          f'completed_epochs={info["completed_epochs"]} '
          f'partial_epoch={info["partial_epoch"]}')

def _rows_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def pool_mc_references(repeats, tolerance):
    """Average independent MC runs; keep repeat spread as a precision diagnostic.

    Deterministic quantities (e.g. DAVI loss) remain equal across repeats. A
    precision warning never blocks a pilot or discards the already-computed data.
    """
    if not repeats:
        raise ValueError('No independent MC repeats')
    if len({r['input_sha256'] for r in repeats}) != 1:
        raise RuntimeError('MC repeats used different fixed reference inputs')
    import math
    summary=dict(repeats[0])
    numeric=[]
    ignored={'n_images','s_ref','gap_identity_abs'}
    for key,value in repeats[0].items():
        if key not in ignored and type(value) in (int,float) and all(
            type(r.get(key)) in (int,float) and math.isfinite(r[key]) for r in repeats):
            numeric.append(key)
    summary.update({key:statistics.mean(r[key] for r in repeats) for key in numeric})
    summary['mc_repeats']=len(repeats)
    monitored=['original_nll','branch_nll','gap1']+(['gap2'] if 'gap2' in summary else [])
    if len(repeats)>1:
        summary['mc_repeat_sd']={k:statistics.stdev(r[k] for r in repeats) for k in numeric}
        summary['mc_repeat_spread']={k:max(r[k] for r in repeats)-min(r[k] for r in repeats)
                                     for k in numeric}
        summary['mc_reference_stable']=all(summary['mc_repeat_spread'][k]<=tolerance
                                           for k in monitored)
    else:
        summary['mc_repeat_sd']={}
        summary['mc_repeat_spread']={}
        summary['mc_reference_stable']=None
    summary['mc_reference_tolerance']=tolerance
    summary['mc_precision_warning']=(len(repeats)>1 and not summary['mc_reference_stable'])
    if 'det_nll' in summary:
        summary['gap_identity_abs']=abs((summary['gap1']+summary['gap2'])-
                                         (summary['det_nll']-summary['original_nll']))
    return summary


def paired_mc_moment_differences(first, another):
    """Per-logit absolute change between independent MC estimates on same images.

    Values are absolute logit units (means) or squared-logit units (variances).
    This measures MC reference noise, not error against DAVI propagated moments.
    """
    if len(first)!=len(another):
        raise RuntimeError('Repeated moment vectors have different image counts')
    keys=('mu_main_mc','var_main_mc','mu_rep_mc','var_rep_mc')
    totals={key:0.0 for key in keys};counts={key:0 for key in keys}
    for a,b in zip(first,another):
        for key in keys:
            if len(a[key])!=len(b[key]):
                raise RuntimeError('Repeated moment vectors have mismatched class dimensions')
            totals[key]+=sum(abs(float(x)-float(y)) for x,y in zip(a[key],b[key]))
            counts[key]+=len(a[key])
    return {key+'_repeat_abs_delta':totals[key]/counts[key] if counts[key] else None for key in keys}


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
    from .reference_cache import REFERENCE_SCHEMA, initial_snapshot_hash, is_current
    init_sha=initial_snapshot_hash(args.run_dir)
    out=Path(args.run_dir)/'diagnostics'/f'{ck["selection"]}'
    if ck['epoch'] is not None:
        out=out/f'epoch_{ck["epoch"]}'
    out=out/f'S{args.samples}_R{args.mc_repeats}'
    if args.smoke:
        out=out/'smoke'
    out.mkdir(parents=True,exist_ok=True)
    result={}
    # An initialization snapshot is optional; never fabricate prompt-change values.
    init_state=None
    initial=Path(args.run_dir)/'trajectory'/'refactor_model'/'model.pth.tar-0'
    if initial.exists():
        init_state=torch.load(initial,map_location='cpu',weights_only=False)['state_dict']
    for split in args.splits:
        sub=out/split
        existing=sub/'metrics.json'
        if existing.exists() and not args.overwrite and is_current(
                existing,checkpoint_sha256=ck['checkpoint_sha256'],
                initial_sha256=init_sha,samples=args.samples,repeats=args.mc_repeats,
                batch_size=args.batch_size,tolerance=args.mc_tolerance,split=split,
                limit_batches=args.limit_batches,require_vectors=args.vectors,
                require_e4=True):
            result[split]=json.loads(existing.read_text(encoding='utf-8'))
            print(f'[SKIP current reference split] {sub}')
            continue
        if existing.exists() and not args.overwrite:
            print(f'[REFERENCE] Stale/legacy diagnostics; recomputing only: {sub}',flush=True)
        loader=reference_loader(trainer,split,args.batch_size)
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        repeat_summaries=[];first_rows=None;first_vectors=None;moment_repeat_deltas=[]
        for repeat in range(args.mc_repeats):
            # Independent posterior reference draws. Evaluation cannot advance the
            # native training RNG even if invoked inside an outer process.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(40300+int(meta['seed'])+(0 if split=='test' else 100)+100003*repeat)
                summary,rows,vectors=reference_diagnostics(
                    trainer,loader,args.samples,args.chunk,args.limit_batches,
                    davi_params,(args.vectors or (davi_params is not None and args.mc_repeats>1)),args.parity and repeat==0)
            repeat_summaries.append(summary)
            if repeat==0:first_rows,first_vectors=rows,vectors
            elif summary['input_sha256']!=repeat_summaries[0]['input_sha256']:
                raise RuntimeError('MC repeats do not use identical fixed reference inputs')
            if repeat>0 and davi_params is not None and first_vectors and vectors:
                moment_repeat_deltas.append(paired_mc_moment_differences(first_vectors,vectors))
        summary=pool_mc_references(repeat_summaries,args.mc_tolerance)
        if moment_repeat_deltas:
            summary['mc_moment_repeat_abs_delta']={key:statistics.mean(d[key]
                for d in moment_repeat_deltas) for key in moment_repeat_deltas[0]}
        summary['moment_vectors_reference_repeat']=0 if args.vectors else None
        if summary['mc_precision_warning']:
            print(f'[E2 WARNING] MC reference repeat spread exceeds {args.mc_tolerance} '
                  f'nats/image; results saved for inspection, increase S if needed',flush=True)
        summary.update({'dataset':meta['dataset'],'method':meta['method'],
                        'shot':meta['shot'],'seed':meta['seed'],'split':split,
                        'objective_type':tag,'partial':args.smoke,
                        'checkpoint':ck,'native_predictive':False,
                        'diagnostic_transform':'fixed test-style transform',
                        'reference_schema':REFERENCE_SCHEMA,
                        'diagnostic_batch_size':args.batch_size,
                        'diagnostic_limit_batches':args.limit_batches,
                        'initial_checkpoint_sha256':init_sha})
        summary.update(posterior_summary(trainer.method,init_state))
        sub=out/split
        sub.mkdir(exist_ok=True)
        save_json(sub/'metrics.json',summary)
        _rows_csv(sub/'per_image.csv',first_rows)
        if args.mc_repeats>1:
            save_json(sub/'independent_mc_repeats.json',repeat_summaries)
        if args.vectors and first_vectors:
            save_json(sub/'moment_vectors.json',first_vectors)
        else:
            (sub/'moment_vectors.json').unlink(missing_ok=True)
        if args.mc_repeats==1:
            (sub/'independent_mc_repeats.json').unlink(missing_ok=True)
        result[split]=summary
        print(f'[reference {split}] {sub} N={len(first_rows)} S={args.samples} repeats={args.mc_repeats}')
    return result


def select_e3_fixed_batch(trainer, size, *, strategy='stratified', selection_seed=2027):
    """A deterministic batch from the *full* fixed-training input pool.

    The native FS dataset split is untouched. We construct an evaluation-style
    view using the same deterministic transform as the E2 train_fixed protocol.
    For stratified selection, shuffle indices within each label and interleave
    labels, avoiding the class-ordered first-batch bias of SequentialSampler.
    """
    import random
    import torch
    from torch.utils.data import DataLoader, Subset
    from collections import Counter, defaultdict
    if size < 1: raise ValueError('E3 batch size must be positive')
    source = reference_loader(trainer, 'train_fixed', size).dataset
    if not hasattr(source,'data_source'):
        raise RuntimeError('E3 requires the native labeled Dassl train_fixed dataset')
    examples = list(source.data_source)
    if not examples:
        raise RuntimeError('E3 train_fixed dataset is empty')
    if strategy == 'first':
        indices = list(range(min(size, len(examples))))
    elif strategy == 'stratified':
        rng = random.Random(int(selection_seed))
        groups = defaultdict(list)
        for i, item in enumerate(examples):
            groups[int(item.label)].append(i)
        labels = sorted(groups)
        rng.shuffle(labels)
        for label in labels:
            rng.shuffle(groups[label])
        indices = []
        while len(indices) < min(size, len(examples)):
            changed = False
            for label in labels:
                if groups[label]:
                    indices.append(groups[label].pop())
                    changed = True
                    if len(indices) >= min(size, len(examples)):
                        break
            if not changed:
                break
    else:
        raise ValueError('E3 batch strategy must be stratified or first')
    view=Subset(source, indices)
    batch=next(iter(DataLoader(view,batch_size=len(indices),shuffle=False,
                               num_workers=0,drop_last=False)))
    labels=batch['label'].long()
    if len(labels) != len(indices) or len(set(indices)) != len(indices):
        raise RuntimeError('Invalid E3 fixed input selection')
    digest=hashlib.sha256()
    digest.update(batch['img'].detach().cpu().contiguous().numpy().tobytes())
    digest.update(labels.cpu().contiguous().numpy().tobytes())
    return batch, dict(selection_strategy=strategy,selection_seed=int(selection_seed),
                       batch_size_requested=int(size),n_images=int(len(labels)),
                       n_unique_classes=len(set(labels.tolist())),
                       label_histogram={str(k):v for k,v in sorted(Counter(labels.tolist()).items())},
                       selected_indices=indices,
                       selected_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                       input_sha256=digest.hexdigest(),
                       transform_protocol='native train_fixed deterministic test transform')


def e3_output_tag(ref, chunk, repeats, batch_size, strategy, seed, sizes):
    """Include every E3 diagnostic control, never confuse different batch runs."""
    return (f'refS{ref}_chunk{chunk}_R{repeats}_B{batch_size}_'
            f'sel{strategy}_seed{seed}_sizes'+'-'.join(str(s) for s in sizes))


def _gradient_noise_eval(args,trainer,meta,ck):
    from .diagnostics import mc_gradient_noise_by_group
    from .core import sha256
    if meta['method']!='branch':
        raise ValueError('Gradient noise analysis requires native AVI-Branch checkpoint')
    base=Path(args.run_dir)/'diagnostics'/ck['selection']
    if ck['epoch'] is not None:base=base/f'epoch_{ck["epoch"]}'
    tag=(f'J{args.noise_draws}_B{args.batch_size}_sel{args.grad_batch_strategy}'
         f'_seed{args.grad_input_seed}_rng{args.noise_seed}')
    out=base/'gradient_noise'/tag
    path=out/'gradient_noise.json'
    batch,inp=select_e3_fixed_batch(trainer,args.batch_size,strategy=args.grad_batch_strategy,
                                     selection_seed=args.grad_input_seed)
    expected={'draws':args.noise_draws,'batch_size':args.batch_size,
              'input_seed':args.grad_input_seed,'strategy':args.grad_batch_strategy,
              'noise_seed':args.noise_seed}
    if path.exists() and not args.overwrite:
        previous=json.loads(path.read_text())
        if (not previous.get('partial',False)
            and previous.get('checkpoint',{}).get('checkpoint_sha256')==ck['checkpoint_sha256']
            and previous.get('input',{}).get('input_sha256')==inp['input_sha256']
            and previous.get('noise_parameters')==expected):
            print(f'[SKIP verified E3 gradient noise] {path}');return
    measured=mc_gradient_noise_by_group(trainer,batch,draws=args.noise_draws,seed=args.noise_seed)
    out.mkdir(parents=True,exist_ok=True)
    save_json(path,{**measured,'checkpoint':ck,'dataset':meta['dataset'],
      'method':meta['method'],'shot':meta['shot'],'seed':meta['seed'],
      'input':inp,'partial':bool(args.smoke),
      'noise_parameters':expected})
    print(f'[E3 gradient noise] {path}')


def _gradient_eval(args, trainer, meta, ck):
    """Independent E3, with recorded fixed inputs and configuration isolation."""
    from .diagnostics import gradient_quality
    if meta['method']!='davi':
        raise ValueError('Gradient E3 requires a DAVI checkpoint')
    base=Path(args.run_dir)/'diagnostics'/ck['selection']
    if ck['epoch'] is not None:
        base=base/f'epoch_{ck["epoch"]}'
    tag=e3_output_tag(args.grad_ref,args.grad_chunk,args.grad_repeats,
                      args.batch_size,args.grad_batch_strategy,args.grad_input_seed,args.grad_sizes)
    out=base/'gradients'/tag
    if args.smoke:
        out=out/'smoke'
    if (out/'gradient.json').exists() and not args.overwrite:
        raise FileExistsError('E3 result already exists (Shell launcher can skip or recompute): '+str(out/'gradient.json'))
    batch, batch_info=select_e3_fixed_batch(trainer,args.batch_size,
                                           strategy=args.grad_batch_strategy,
                                           selection_seed=args.grad_input_seed)
    grad=gradient_quality(trainer,batch,reference_samples=args.grad_ref,
                          chunk=args.grad_chunk,repeats=args.grad_repeats,
                          sizes=tuple(args.grad_sizes),check_parity=args.grad_parity)
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'gradient.json',{**grad,'dataset':meta['dataset'],
        'method':meta['method'],'seed':meta['seed'],'shot':meta['shot'],
        'checkpoint':ck,'fixed_train_augmentation':False,
        'input':batch_info, 'e3_parameters':{
            'batch_size':args.batch_size,'reference_samples':args.grad_ref,
            'chunk':args.grad_chunk,'repeats':args.grad_repeats,
            'mc_sizes':list(args.grad_sizes),
            'batch_strategy':args.grad_batch_strategy,
            'input_seed':args.grad_input_seed,
            'chunk_gradient_parity_enabled':args.grad_parity},
        'e3_output_tag':tag, 'partial':bool(args.smoke),
        'e3_reference_warning':not grad['reference_stable']})
    if not grad['reference_stable']:
        print('[E3 WARNING] Independent MC gradient references do not agree; saved for inspection',flush=True)
    print(f'[E3 gradient] {out} fixed_images={batch_info["n_images"]} '
          f'classes={batch_info["n_unique_classes"]} '
          f'input_sha256={batch_info["input_sha256"][:12]}')


def _e4_evolution_eval(args, trainer, meta, ck):
    """Read-only E4 time-course diagnostics on fixed class-spread train images.

    No evaluation-data augmentation and no mutation of the restored checkpoint.
    Sampling is solely for computing the MC-mean Rep logit margin; covariance
    trace and posterior-mean norm come directly from the native posterior.
    """
    from .diagnostics import posterior_summary, mc_mean_logit_margin
    import torch
    from .core import cached_encode, cached_draw
    from .reference_cache import initial_snapshot_hash
    model = trainer.method.model
    model.eval()
    batch, inp = select_e3_fixed_batch(trainer, args.e4_images,
                                       strategy='stratified', selection_seed=args.e4_input_seed)
    tag = (f'images{args.e4_images}_S{args.e4_draws}_chunk{args.e4_chunk}'
           f'_B{args.e4_batch}_seed{args.e4_input_seed}')
    out = Path(args.run_dir)/'diagnostics'/ck['selection']
    if ck.get('epoch') is not None:
        out = out/f'epoch_{ck["epoch"]}'
    out = out/'e4_evolution'/tag/'metrics.json'
    init_sha=initial_snapshot_hash(args.run_dir)
    controls=dict(images=args.e4_images,draws=args.e4_draws,chunk=args.e4_chunk,
                  batch=args.e4_batch,input_seed=args.e4_input_seed,
                  transform_protocol=inp['transform_protocol'])
    if out.is_file() and not args.overwrite:
        try:
            old=json.loads(out.read_text(encoding='utf-8'))
            if (old.get('schema')==1 and old.get('checkpoint',{}).get('checkpoint_sha256')==ck['checkpoint_sha256']
                and old.get('initial_checkpoint_sha256')==init_sha
                and old.get('settings')==controls
                and old.get('input',{}).get('input_sha256')==inp['input_sha256']):
                print('[SKIP valid E4 evolution] '+str(out))
                return
        except (OSError,ValueError,KeyError):
            pass
    classes=trainer.method.num_classes
    all_margins=[]
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        # Identical RNG schedule at fixed image/seed/epoch, independently
        # of the method's training RNG. No sampling changes saved weights.
        torch.manual_seed(61103+int(meta['seed'])+1009*int(ck['epoch'] or 0))
        images=batch['img']; labels=batch['label'].long()
        for start in range(0,len(labels),args.e4_batch):
            x=images[start:start+args.e4_batch].to(trainer.method.device)
            y=labels[start:start+args.e4_batch].to(trainer.method.device)
            cache=cached_encode(model,x,classes)
            mean_logits=torch.zeros((len(y),classes),device=y.device,dtype=torch.float64)
            done=0
            while done<args.e4_draws:
                k=min(args.e4_chunk,args.e4_draws-done)
                _,rep_logits=cached_draw(cache,k)
                mean_logits+=rep_logits.double().sum(0)
                done+=k
            mean_logits/=args.e4_draws
            margin=mc_mean_logit_margin(mean_logits,y)
            all_margins.extend(margin.detach().cpu().tolist())
    import statistics
    values=posterior_summary(trainer.method)
    record=dict(schema=1, method=meta['method'], dataset=meta['dataset'],
                shot=meta['shot'], seed=meta['seed'], study=meta['study'],
                checkpoint=ck, initial_checkpoint_sha256=init_sha,
                input=inp, settings=controls, n_images=len(all_margins),
                rep_margin_type='mc_mean_logit',
                rep_margin_median=float(statistics.median(all_margins)),
                rep_margin_mean=float(statistics.mean(all_margins)),
                rep_margin_positive_fraction=float(sum(m>0 for m in all_margins)/len(all_margins)),
                partial=False, **values)
    save_json(out,record)
    print(f'[E4 evolution] epoch={ck["epoch"]}, method={meta["method"]}: {out}')


def evaluate(args):
    configure_gpu(args.gpu)
    from .diagnostics import native_predictions
    trainer,meta,ck=restore(args.run_dir,args.gpu,args.checkpoint,args.epoch)
    if meta.get('status')!='complete':
        raise RuntimeError('Refusing to evaluate incomplete training run')
    if meta['study']=='matched_time' and args.checkpoint not in ('latest','time_curve'):
        raise RuntimeError('E5 supports latest and saved time_curve diagnostic snapshots only')
    if args.checkpoint=='time_curve' and (meta['study']!='matched_time' or args.mode!='reference'):
        raise RuntimeError('E5 time_curve checkpoints are only for read-only reference diagnostics')
    if args.limit_batches is not None and not args.smoke:
        raise ValueError('--limit-batches requires --smoke')
    if args.mode in ('both','reference'):
        _reference_eval(args,trainer,meta,ck)
    if args.mode=='gradient':
        _gradient_eval(args,trainer,meta,ck)
    if args.mode=='gradient_noise':
        _gradient_noise_eval(args,trainer,meta,ck)
    if args.mode=='e4_evolution':
        _e4_evolution_eval(args,trainer,meta,ck)
    if args.mode in ('both','formal'):
        out=Path(args.run_dir)/'formal'/ck['selection']
        if ck['epoch'] is not None:
            out=out/f'epoch_{ck["epoch"]}'
        out=out/('smoke' if args.smoke else 'complete')
        from .reference_cache import formal_is_current
        destination=out/'test.json'
        if destination.is_file() and not args.overwrite and formal_is_current(
                destination,ck['checkpoint_sha256']):
            print(f'[SKIP verified current formal test] {destination}')
            return
        if destination.is_file() and not args.overwrite:
            print(f'[FORMAL] Stale source checkpoint, recomputing: {destination}',flush=True)
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


def native_dataset(value):
    """Accept any registered-by-config native MMRL dataset basename."""
    import argparse
    import re
    from .core import ROOT
    if not re.fullmatch(r'[a-z][a-z0-9_]*', value):
        raise argparse.ArgumentTypeError(
            f'Invalid dataset config basename: {value!r} (use lowercase letters/digits/underscores)')
    config = ROOT / 'configs' / 'datasets' / (value + '.yaml')
    if not config.is_file():
        raise argparse.ArgumentTypeError(
            f'Native MMRL dataset config not found: {config}. '
            'Use a configs/datasets/*.yaml basename, and make sure the data/loader is available.')
    return value


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
        x.add_argument('--dataset',required=True,type=native_dataset)
        x.add_argument('--method',required=True,choices=MODES)
        x.add_argument('--shot',type=int,default=16)
        x.add_argument('--seed',type=int,default=1)
        x.add_argument('--output',default='output_supplementary_v7')
        x.add_argument('--data-root',default='DATASETS')
        x.add_argument('--epochs',type=int,help='Optional override; native YAML when omitted')
        x.add_argument('--mc-train',type=int,help='Optional override; native YAML when omitted')
    for opt in ('--dataset','--method'):
        d.add_argument(opt,required=True,type=native_dataset) if opt=='--dataset' else d.add_argument(opt,required=True,choices=MODES)
    d.add_argument('--shot',type=int,default=16)
    d.add_argument('--seed',type=int,default=1)
    d.add_argument('--output',default='output_supplementary_v7')
    d.add_argument('--data-root',default='DATASETS')
    d.add_argument('--batch-size',type=int,default=2)
    t.add_argument('--trajectory',action='store_true',help='Save initialization + every epoch snapshot for E1')
    t.add_argument('--e3-snapshot-epochs',nargs='+',type=int,default=[],
                   help='On a NEW DAVI training run, save only selected early/middle epochs for E3')
    t.add_argument('--init-snapshot',action='store_true',help='Save only epoch zero to measure E4 prompt change, no epoch history')
    b.add_argument('--budget-seconds',type=float,required=True)
    b.add_argument('--log-every',type=int,default=20)
    b.add_argument('--time-curve',action='store_true',
      help='Save 0/25/50/75/100 percent wall-time native snapshots (outside training clock)')
    for x in (e,q):
        x.add_argument('--run-dir',required=True)
        x.add_argument('--batch-size',type=int,default=32)
    e.add_argument('--checkpoint',choices=('best','latest','trajectory','time_curve'),default='best')
    e.add_argument('--epoch',type=int)
    e.add_argument('--mode',choices=('reference','formal','both','gradient','gradient_noise','e4_evolution'),default='both')
    e.add_argument('--samples',type=int,default=1024)
    e.add_argument('--chunk',type=int,default=16)
    e.add_argument('--mc-repeats',type=int,default=1,help='Independent S_ref draws on each fixed input; 1 means no MC precision estimate')
    e.add_argument('--mc-tolerance',type=float,default=0.03,help='Maximum repeat spread in NLL (nats/image) for paper outputs')
    e.add_argument('--splits',nargs='+',choices=('train_fixed','test'),default=['test'])
    e.add_argument('--vectors',action='store_true')
    e.add_argument('--gradient',action='store_true',help='Deprecated: use --mode gradient (standalone E3)')
    e.add_argument('--e4-images',type=int,default=128)
    e.add_argument('--e4-draws',type=int,default=64)
    e.add_argument('--e4-chunk',type=int,default=16)
    e.add_argument('--e4-batch',type=int,default=32)
    e.add_argument('--e4-input-seed',type=int,default=2027)
    e.add_argument('--noise-draws',type=int,default=10)
    e.add_argument('--noise-seed',type=int,default=99173)
    e.add_argument('--grad-ref',type=int,default=512)
    e.add_argument('--grad-chunk',type=int,default=8)
    e.add_argument('--grad-repeats',type=int,default=5)
    e.add_argument('--grad-sizes',nargs='+',type=int,default=[1,5,10,50],
                   help='Independent MC sample budgets for E3 curve')
    e.add_argument('--grad-batch-strategy',choices=('stratified','first'),default='stratified',
                   help='Stratified is the E3 default; first is only for legacy diagnostics')
    e.add_argument('--grad-input-seed',type=int,default=2027,
                   help='Reproducible selection of labeled fixed-training inputs')
    e.add_argument('--skip-grad-parity',dest='grad_parity',action='store_false',default=True,
                   help='Skip identical-posterior-draw full/chunk gradient parity (debug only)')
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
        if a.gradient:
            p.error('--gradient is retired; use --mode gradient to run E3 separately from E2')
        if a.e4_images<1 or a.e4_draws<1 or a.e4_chunk<1 or a.e4_batch<1 or a.e4_input_seed<0 or a.noise_draws<2 or a.noise_seed<0 or a.samples<2 or a.chunk<1 or a.grad_ref<2 or a.grad_chunk<1 or a.grad_repeats<1 or a.mc_repeats<1 or a.mc_tolerance<=0 or a.batch_size<1 or not a.grad_sizes or any(z<1 for z in a.grad_sizes):
            p.error('Invalid reference/gradient sample count')
        if a.limit_batches is not None and (a.limit_batches<1 or not a.smoke):
            p.error('--limit-batches >=1 requires --smoke')
        if not a.parity and not a.smoke:
            p.error('--no-parity is strictly smoke-only')
    if a.command=='train':
        if a.e3_snapshot_epochs:
            if a.method!='davi' or a.trajectory or a.init_snapshot:
                p.error('--e3-snapshot-epochs requires DAVI and cannot mix with snapshot modes')
            if len(set(a.e3_snapshot_epochs))!=len(a.e3_snapshot_epochs) or any(
                    epoch<1 for epoch in a.e3_snapshot_epochs):
                p.error('--e3-snapshot-epochs must contain distinct positive epoch numbers')
        train_native(a)
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
