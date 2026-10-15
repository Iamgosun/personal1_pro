"""Fixed-budget, 10-epoch training-efficiency benchmark for MMRL-family methods.

Place next to benchmark_efficiency.py and launch from the MMRL/ directory.
This script executes *all* batches of every epoch with the real OnlineExecutor;
it does NOT call validation/test or save checkpoints.

train_time_min includes train-loader iteration, augmentation/data wait, all
optimizer updates, synchronization and step metrics for the 10 epochs.
ms_per_update excludes fetching the batch and the first --warmup updates.
Profiling is performed only AFTER training and is excluded from train_time_min.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_efficiency import (
    METHOD_FILES,
    make_cfg,
    epoch_batches,
    profile_one_update,
    _import_runtime_modules,
    install_kplusval_datasetbase_patch,
)

# benchmark_efficiency sets CPU thread environment before importing PyTorch.
import torch
from dassl.engine import build_trainer
from dassl.utils import set_random_seed


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='DATASETS')
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', choices=METHOD_FILES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--shots', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--backbone', default='ViT-B/16')
    p.add_argument('--train-epochs', type=int, default=10)
    p.add_argument('--warmup', type=int, default=20,
                   help='Real updates inside the 10 epochs, excluded only from per-update average')
    p.add_argument('--mc-train', type=int, default=50,
                   help='Actual N_MC_TRAIN for BayesRTMMRL; Det training uses no samples')
    p.add_argument('--output-dir', default='output_efficiency_10ep')
    p.add_argument('--profile-flops', action='store_true')
    p.add_argument('--profile-steps', type=int, default=3)
    p.add_argument('--method-config-file', default='')
    p.add_argument('--runtime-config-file', default='configs/runtime/mmrl_family.yaml')
    p.add_argument('--protocol-config-file', default='configs/protocols/fs.yaml')
    p.add_argument('--exp-config', default='')
    return p.parse_args()


def main():
    a = arguments()
    if not (a.train_epochs >= 1 and a.shots >= 1 and a.batch_size >= 1
            and a.warmup >= 0 and a.mc_train >= 1 and a.profile_steps >= 1):
        raise ValueError('train-epochs, shots, batch-size, mc-train and profile-steps must be positive; warmup >= 0')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU via CUDA_VISIBLE_DEVICES')

    out = (Path(a.output_dir) / a.dataset / a.method
           / f'shots_{a.shots}' / f'seed_{a.seed}')
    out.mkdir(parents=True, exist_ok=True)

    _import_runtime_modules()
    cfg, parsed = make_cfg(a, out)
    # make_cfg returns a frozen cfg. Override the LR schedule as well as the loop
    # duration: otherwise 10 epochs would run the first 10 epochs of a 50-epoch
    # cosine schedule, which is a different experiment.
    cfg.defrost()
    cfg.OPTIM.MAX_EPOCH = a.train_epochs
    if a.method == 'BayesRTMMRL':
        cfg.BAYESRT_MMRL.N_MC_TRAIN = a.mc_train
    cfg.freeze()

    install_kplusval_datasetbase_patch(cfg)
    set_random_seed(a.seed)
    trainer = build_trainer(cfg)
    trainer.model.train()

    n_batches = len(trainer.train_loader_x)
    if n_batches == 0:
        raise ValueError('Empty train loader')
    num_updates = n_batches * a.train_epochs
    if a.warmup >= num_updates:
        raise ValueError(f'--warmup={a.warmup} must be smaller than total updates={num_updates}')
    if int(trainer.max_epoch) != a.train_epochs:
        raise RuntimeError(f'Unexpected trainer.max_epoch={trainer.max_epoch}; expected {a.train_epochs}')
    actual_mc = getattr(trainer.method, 'n_mc_train', None)
    if a.method == 'BayesRTMMRL' and actual_mc != a.mc_train:
        raise RuntimeError(f'Actual MC samples={actual_mc}, expected={a.mc_train}')

    print(f'[10-epoch] {a.dataset} | {a.method} | seed={a.seed} '
          f'| epochs={a.train_epochs} | batches/epoch={n_batches} '
          f'| batch_size={a.batch_size} | actual MC={actual_mc}')

    measured_ms = []
    epoch_seconds = []
    step_count = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    full_train_start = time.perf_counter()
    for epoch in range(a.train_epochs):
        epoch_start = time.perf_counter()
        for batch in epoch_batches(trainer, epoch):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            trainer.forward_backward(batch)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter() - step_start) * 1e3
            if step_count >= a.warmup:
                measured_ms.append(step_ms)
            step_count += 1
        torch.cuda.synchronize()
        epoch_s = time.perf_counter() - epoch_start
        epoch_seconds.append(epoch_s)
        print(f'[train] {epoch + 1}/{a.train_epochs} '
              f'epoch_time={epoch_s:.2f}s updates={step_count}/{num_updates}')
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - full_train_start
    peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # Profile in separate additional forward/backward updates. These updates
    # are NOT included in the 10-epoch training time, or in the latency/memory.
    profiled_gflops = []
    if a.profile_flops:
        trainer.epoch = a.train_epochs - 1
        for i, batch in zip(range(a.profile_steps),
                            _repeat_batches(trainer.train_loader_x)):
            trainer.batch_idx = i % n_batches
            profiled_gflops.append(profile_one_update(trainer, batch))

    result = {
        'dataset': a.dataset,
        'method': a.method,
        'seed': a.seed,
        'gpu': torch.cuda.get_device_name(0),
        'num_shots': a.shots,
        'batch_size': a.batch_size,
        'epochs_completed': a.train_epochs,
        'max_epoch': int(trainer.max_epoch),
        'optimizer': str(cfg.OPTIM.NAME),
        'learning_rate': float(cfg.OPTIM.LR),
        'precision': str(trainer.method.get_precision()),
        'n_mc_train_effective': actual_mc if a.method == 'BayesRTMMRL' else None,
        'method_config_file': parsed.method_config_file,
        'runtime_config_file': parsed.runtime_config_file,
        'hpo_enabled': bool(cfg.HPO.ENABLED),
        'updates_total': step_count,
        'updates_timed': len(measured_ms),
        'warmup_updates_within_training': a.warmup,
        'train_time_min': train_seconds / 60.0,
        'train_time_s': train_seconds,
        'epoch_seconds': epoch_seconds,
        'ms_per_update': statistics.mean(measured_ms),
        'ms_per_update_sample_std': (
            statistics.stdev(measured_ms) if len(measured_ms) > 1 else None),
        'updates_per_s': 1000.0 / statistics.mean(measured_ms),
        'peak_allocated_mib': peak_allocated_mib,
        'peak_reserved_mib': peak_reserved_mib,
        'gflops_profiler_per_update': (
            statistics.mean(profiled_gflops) if profiled_gflops else None),
        'flops_scope': 'PyTorch Profiler-recognized operator FLOPs for extra post-training updates; incomplete, not total training FLOPs',
        'train_time_scope': 'All loader iteration and optimizer updates over exactly N epochs; excludes model/dataloader creation, validation, testing, checkpoints and post-training FLOPs profiling',
        'step_time_scope': 'Synchronized OnlineExecutor.forward_backward; excludes batch retrieval; first N real updates used as warmup only for this mean',
        'memory_scope': 'CUDA allocated/reserved peak after model build, across exactly N training epochs, before profiling',
        'note': 'Fixed-budget training-only benchmark; no accuracy/validation/test measured',
    }
    target = out / 'efficiency.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                      encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k in ('method', 'dataset', 'seed', 'epochs_completed',
                               'updates_total', 'n_mc_train_effective',
                               'train_time_min', 'ms_per_update',
                               'peak_allocated_mib', 'gflops_profiler_per_update')},
                     indent=2, ensure_ascii=False))
    print('Saved:', target)


def _repeat_batches(loader):
    while True:
        yield from loader


if __name__ == '__main__':
    main()
