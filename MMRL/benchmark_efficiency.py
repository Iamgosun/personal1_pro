"""Run real training epochs, then benchmark OnlineExecutor updates.

Launch from repository MMRL/ with the same YAML layers used by run_plan.sh.
Does not change any model, optimizer, or training implementation.
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('TORCH_NUM_THREADS', '12')
os.environ.setdefault('TORCH_INTEROP_THREADS', '4')

import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed
from core.config import setup_cfg
from datasets.fair_val_protocol import install_kplusval_datasetbase_patch
from run import _import_runtime_modules, _prepare_method_config_for_kplusval

METHOD_FILES = {
    'MMRL': 'configs/methods/mmrl.yaml',
    'BayesRTMMRL': 'configs/methods/bayesrt_mmrl.yaml',
    'DetBayesRTMMRL': 'configs/methods/det_bayesrt_mmrl.yaml',
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=os.environ.get('DATA_ROOT', 'DATASETS'))
    p.add_argument('--dataset', required=True, help='run_plan.sh dataset key, e.g. caltech101')
    p.add_argument('--method', choices=METHOD_FILES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--shots', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=None, help='Default: mmrl_family.yaml (32)')
    p.add_argument('--backbone', default='ViT-B/16')
    p.add_argument('--pretrain-epochs', type=int, default=6, help='ACTUALLY train N full epochs before measurement, unlike old --epoch flag')
    p.add_argument('--warmup', type=int, default=20, help='Real untimed optimizer updates AFTER pretraining')
    p.add_argument('--steps', type=int, default=100, help='Measured optimizer updates, possibly spanning epochs')
    p.add_argument('--output-dir', default='output_efficiency')
    p.add_argument('--profile-flops', action='store_true', help='Optional, incomplete operator-level FLOPs; NOT full training FLOPs')
    p.add_argument('--profile-steps', type=int, default=1)
    p.add_argument('--exp-config', default='')
    p.add_argument('--method-config-file', default='', help='Use exact YAML of main experiment')
    p.add_argument('--runtime-config-file', default='configs/runtime/mmrl_family.yaml')
    p.add_argument('--protocol-config-file', default='configs/protocols/fs.yaml')
    return p.parse_args()


def make_cfg(a, out):
    opts = [
        'DATASET.NUM_SHOTS', str(a.shots),
        'DATASET.SUBSAMPLE_CLASSES', 'all',
        'MODEL.BACKBONE.NAME', a.backbone,
        # Benchmark one fit, not multi-candidate HPO. Record this departure.
        'HPO.ENABLED', 'False',
    ]
    if a.batch_size is not None:
        opts += ['DATALOADER.TRAIN_X.BATCH_SIZE', str(a.batch_size)]
    args = argparse.Namespace(
        root=a.root, output_dir=str(out),
        dataset_config_file=f'configs/datasets/{a.dataset}.yaml',
        method_config_file=a.method_config_file or METHOD_FILES[a.method],
        protocol_config_file=a.protocol_config_file,
        runtime_config_file=a.runtime_config_file,
        exp_config=a.exp_config,
        method=a.method, protocol='FS', exec_mode='online', seed=a.seed,
        trainer='RefactorRunner', opts=opts,
    )
    for path in (args.dataset_config_file, args.method_config_file,
                 args.protocol_config_file, args.runtime_config_file):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    sanitized = _prepare_method_config_for_kplusval(args)
    try:
        cfg = setup_cfg(args)
    finally:
        if sanitized:
            Path(sanitized).unlink(missing_ok=True)
    if cfg.METHOD.NAME != a.method or cfg.METHOD.EXEC_MODE != 'online':
        raise ValueError(f'Unexpected method/mode {cfg.METHOD.NAME}/{cfg.METHOD.EXEC_MODE}; check config')
    return cfg, args


def epoch_batches(trainer, epoch):
    """Yield real loader batches, with true 0-based indices and epoch context.

    The last batch invokes OnlineExecutor's scheduler update exactly as in a normal epoch.
    """
    trainer.epoch = epoch
    trainer.num_batches = len(trainer.train_loader_x)
    if trainer.num_batches == 0:
        raise ValueError('Empty train loader')
    for idx, batch in enumerate(trainer.train_loader_x):
        trainer.batch_idx = idx
        yield batch


def active_beta(method, epoch, max_epoch):
    if not hasattr(method, 'set_kl_beta'):
        return None
    warmup_epochs = int(getattr(method, 'kl_warmup_epochs', 0))
    if warmup_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return 0.0
    ramp_epochs = max(1, max_epoch - warmup_epochs - 1)
    return min(1.0, float(epoch - warmup_epochs) / float(ramp_epochs))


def profile_one_update(trainer, batch):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        trainer.forward_backward(batch)
        torch.cuda.synchronize()
    events = prof.key_averages()
    return sum(float(e.flops or 0) for e in events) / 1e9


def main():
    a = arguments()
    if a.shots < 1 or a.steps < 1 or a.warmup < 0 or a.pretrain_epochs < 0 or a.profile_steps < 1:
        raise ValueError('Invalid shots/steps/warmup/pretrain-epochs/profile-steps')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose precisely one GPU: CUDA_VISIBLE_DEVICES=0')
    out = Path(a.output_dir) / a.dataset / a.method / f'shots_{a.shots}' / f'seed_{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    _import_runtime_modules()
    cfg, parsed = make_cfg(a, out)
    install_kplusval_datasetbase_patch(cfg)
    set_random_seed(a.seed)
    trainer = build_trainer(cfg)
    trainer.model.train()
    if getattr(trainer, 'scaler', None) is not None:
        pass  # Existing executor handles AMP GradScaler internally.
    max_epoch = int(trainer.max_epoch)
    if a.pretrain_epochs >= max_epoch:
        raise ValueError(f'pretrain-epochs={a.pretrain_epochs} must be less than max_epoch={max_epoch}')
    n_batches = len(trainer.train_loader_x)
    if n_batches <= 0:
        raise ValueError('Empty train loader')
    print(f'[benchmark] {a.method} {a.dataset} seed={a.seed} batch={cfg.DATALOADER.TRAIN_X.BATCH_SIZE} '
          f'actual pretrain epochs={a.pretrain_epochs}; batches/epoch={n_batches}; full schedule={max_epoch}')

    # Pretraining: real optimizer updates, accurate KL context and LR schedule.
    for epoch in range(a.pretrain_epochs):
        for batch in epoch_batches(trainer, epoch):
            trainer.forward_backward(batch)
        print(f'[pretrain] completed epoch {epoch+1}/{a.pretrain_epochs}')

    elapsed = []
    betas = []
    profiled = []
    steps_seen = 0
    untimed_seen = 0
    first_measured_epoch = None
    last_measured_epoch = None
    measured_memory_reset = False
    # IMPORTANT: warmup, profiling and timed steps progress through real epochs.
    # Pretraining ends only at epoch boundaries; no fake batch_idx or schedule jumps.
    for epoch in range(a.pretrain_epochs, max_epoch):
        for batch in epoch_batches(trainer, epoch):
            if untimed_seen < a.warmup:
                trainer.forward_backward(batch)
                untimed_seen += 1
                continue
            if not measured_memory_reset:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                measured_memory_reset = True
            # The input batch is already fetched; synchronize on each side of timed work.
            torch.cuda.synchronize()
            start = time.perf_counter()
            trainer.forward_backward(batch)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1e3)
            betas.append(active_beta(trainer.method, epoch, max_epoch))
            first_measured_epoch = epoch if first_measured_epoch is None else first_measured_epoch
            last_measured_epoch = epoch
            steps_seen += 1
            if steps_seen == a.steps:
                break
        if steps_seen == a.steps:
            break
    if steps_seen < a.steps:
        raise RuntimeError(f'Only {steps_seen} measured steps fit into remaining {max_epoch - a.pretrain_epochs} epochs; '
                           f'reduce --steps/--warmup or --pretrain-epochs')
    mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)

    # Profiling occurs separately, AFTER timing, to avoid contaminating latency/memory.
    # It performs additional optimizer updates, so never use those weights for accuracy.
    # Run a fresh loader iterator: do not reuse the last measured minibatch.
    if a.profile_flops:
        if last_measured_epoch >= max_epoch - 1:
            print('[warning] FLOPs profile runs at the last measured epoch without advancing LR schedule')
        trainer.epoch = last_measured_epoch
        loader_iter = iter(trainer.train_loader_x)
        for i in range(a.profile_steps):
            trainer.batch_idx = i % n_batches
            if i and trainer.batch_idx == 0:
                loader_iter = iter(trainer.train_loader_x)
            profiled.append(profile_one_update(trainer, next(loader_iter)))

    result = {
        'dataset': a.dataset, 'method': a.method, 'seed': a.seed,
        'gpu': torch.cuda.get_device_name(0), 'num_shots': int(cfg.DATASET.NUM_SHOTS),
        'batch_size': int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
        'precision': str(trainer.method.get_precision()), 'optimizer': str(cfg.OPTIM.NAME),
        'learning_rate': float(cfg.OPTIM.LR), 'max_epoch': max_epoch,
        'kl_warmup_epochs': getattr(trainer.method, 'kl_warmup_epochs', None),
        'pretrain_epochs_completed': a.pretrain_epochs,
        'measurement_first_epoch_zero_based': first_measured_epoch,
        'measurement_last_epoch_zero_based': last_measured_epoch,
        'measurement_kl_beta_min': min([x for x in betas if x is not None], default=None),
        'measurement_kl_beta_max': max([x for x in betas if x is not None], default=None),
        'method_config_file': a.method_config_file or METHOD_FILES[a.method],
        'runtime_config_file': parsed.runtime_config_file,
        'protocol_config_file': parsed.protocol_config_file,
        'exp_config': a.exp_config,
        'warmup_steps': untimed_seen, 'steps': steps_seen, 'loader_batches_per_epoch': n_batches,
        'ms_per_update': statistics.mean(elapsed),
        'ms_per_update_sample_std': statistics.stdev(elapsed) if len(elapsed) > 1 else None,
        'updates_per_s': 1000.0 / statistics.mean(elapsed),
        'peak_allocated_mib': mem_alloc, 'peak_reserved_mib': mem_reserved,
        'gflops_profiler_per_update': statistics.mean(profiled) if profiled else None,
        'flops_scope': 'Only PyTorch Profiler recognized operator FLOPs; notably incomplete backward; NOT total train GFLOPs',
        'timing_scope': 'CPU wall clock with CUDA synchronize, prefetched batch, full OnlineExecutor.forward_backward including metrics/logging',
        'memory_scope': 'PyTorch CUDA process allocated/reserved high-water mark from after untimed warmup through timed steps',
        'epoch_scope': 'Actually pretrained full epochs using same OnlineExecutor and learning-rate scheduler as regular run',
        'hpo_scope': 'HPO disabled; one configured fit, no grid search',
        'note': 'timed benchmark stops mid-training; profiling (if enabled) runs extra post-timing optimizer updates',
    }
    (out / 'efficiency.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('Saved:', out / 'efficiency.json')


if __name__ == '__main__':
    main()
