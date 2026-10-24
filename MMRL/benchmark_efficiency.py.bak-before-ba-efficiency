"""MMRL efficiency benchmark, with non-invasive forward component profiling.

Drop-in replacement for the existing benchmark_efficiency.py. Existing CLI,
training, timing, and JSON keys remain compatible. When --profile-flops is
passed, adds profiler-based forward Encoding / Probabilistic adaptation splits.

IMPORTANT:
- Component FLOPs count recognized PyTorch ATen operations, not full FLOPs.
- Component CUDA time sums profiler-attributed kernel durations, NOT wall time.
- Backward kernels are NOT assigned to their originating forward scopes; they
  appear in OTHER (backward/optimizer/metrics and untagged forward operations).
- Markers are installed only AFTER the timing measurement to avoid perturbation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import statistics
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
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
    'DetBayesRTMMRL_CLAMP': 'configs/methods/det_bayesrt_mmrl_clamp.yaml',
}
METHOD_ALIASES = {'DetBayesRTMMRL_CLAMP': 'DetBayesRTMMRL'}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=os.environ.get('DATA_ROOT', 'DATASETS'))
    p.add_argument('--dataset', required=True)
    p.add_argument('--method', choices=METHOD_FILES, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--shots', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--backbone', default='ViT-B/16')
    p.add_argument('--pretrain-epochs', type=int, default=6,
                   help='Actually train N full epochs before measurement')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--output-dir', default='output_efficiency')
    p.add_argument('--profile-flops', action='store_true',
                   help='Profiler FLOPs and forward encoding/head CUDA breakdown')
    p.add_argument('--profile-steps', type=int, default=1)
    p.add_argument('--exp-config', default='')
    p.add_argument('--method-config-file', default='')
    p.add_argument('--runtime-config-file', default='configs/runtime/mmrl_family.yaml')
    p.add_argument('--protocol-config-file', default='configs/protocols/fs.yaml')
    return p.parse_args()


def make_cfg(a, out):
    opts = ['DATASET.NUM_SHOTS', str(a.shots),
            'DATASET.SUBSAMPLE_CLASSES', 'all',
            'MODEL.BACKBONE.NAME', a.backbone,
            'HPO.ENABLED', 'False']
    if a.batch_size is not None:
        opts += ['DATALOADER.TRAIN_X.BATCH_SIZE', str(a.batch_size)]
    args = argparse.Namespace(
        root=a.root, output_dir=str(out),
        dataset_config_file=f'configs/datasets/{a.dataset}.yaml',
        method_config_file=a.method_config_file or METHOD_FILES[a.method],
        protocol_config_file=a.protocol_config_file,
        runtime_config_file=a.runtime_config_file,
        exp_config=a.exp_config,
        method=METHOD_ALIASES.get(a.method, a.method),
        protocol='FS', exec_mode='online', seed=a.seed,
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
    expected = METHOD_ALIASES.get(a.method, a.method)
    if cfg.METHOD.NAME != expected or cfg.METHOD.EXEC_MODE != 'online':
        raise ValueError(f'Unexpected method/mode: {cfg.METHOD.NAME}/{cfg.METHOD.EXEC_MODE}')
    return cfg, args


def epoch_batches(trainer, epoch):
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
    warm = int(getattr(method, 'kl_warmup_epochs', 0))
    if warm <= 0:
        return 1.0
    if epoch < warm:
        return 0.0
    return min(1.0, float(epoch-warm)/float(max(1,max_epoch-warm-1)))


def _scoped(method, label):
    """Wrap an existing bound callable; no source-model edits needed."""
    from torch.profiler import record_function
    @wraps(method)
    def call(*args, **kwargs):
        with record_function(label):
            return method(*args, **kwargs)
    return call


@contextmanager
def _forward_profile_markers(trainer):
    """Temporarily assign profiler scopes, restoring ALL originals afterward.

    PROB/joint wraps the entire joint forward; ENC scopes inside take
    precedence in our attribution. PROB/loss scopes mark MC CE/Jensen CE;
    KL and predictive fusion are also probabilistic operations.
    """
    changes = []

    def patch(obj, name, label):
        if obj is None or not hasattr(obj, name):
            return
        # Avoid inadvertently replacing a class-level descriptor globally.
        had_own = name in getattr(obj, '__dict__', {})
        prev = getattr(obj, name)
        setattr(obj, name, _scoped(prev, label))
        changes.append((obj, name, prev, had_own))

    try:
        m = trainer.model
        method = trainer.method
        patch(m.representation_learner, 'forward', 'ENC/tokens')
        patch(m.text_encoder, 'forward_hidden', 'ENC/text')
        patch(m.image_encoder, 'encode_hidden', 'ENC/visual')
        patch(method.image_encoder_clip, 'forward', 'ENC/reference_visual')

        if hasattr(m, 'forward_joint_moments') and method.method_name == 'DetBayesRTMMRL':
            patch(m, 'forward_joint_moments', 'PROB/joint')
            # det_jensen_ce is a module-level name imported in model.py.
            det_module = importlib.import_module('methods.det_bayesrt_mmrl.model')
            patch(det_module, 'det_jensen_ce', 'PROB/jensen_ce')
        elif hasattr(m, 'forward_joint'):
            patch(m, 'forward_joint', 'PROB/joint')
            patch(method, '_expected_ce', 'PROB/expected_ce')
            patch(method, '_branch_logits_from_stacks', 'PROB/aggregation')
            patch(method, '_build_fusion_from_stacks', 'PROB/fusion')
        else:
            # Plain MMRL: its model forward is encoding + deterministic logits.
            # We label it ENC to avoid incorrectly treating it as Bayesian head.
            patch(m, 'forward', 'ENC/mmrl_model')
        patch(m, 'kl_terms', 'PROB/kl')
        yield
    finally:
        for obj, name, original, had_own in reversed(changes):
            if had_own:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def _phase(event):
    """Nearest enclosing marker wins; ENC is nested within PROB/joint."""
    cursor = event
    while cursor is not None:
        name = cursor.name
        if name.startswith('ENC/'):
            return 'encoding_forward'
        if name.startswith('PROB/'):
            return 'probabilistic_forward'
        cursor = cursor.cpu_parent
    return 'other_including_backward'


def _profiler_breakdown(prof):
    """Attribute recognized ATen FLOPs and CUDA-kernel *durations* to scopes.

    Never interpret the last category as only backward: it also includes
    optimizer, metrics, regularizers and any untagged forward operations.
    Kernel durations are summed (not a critical-path/latency measurement).
    """
    from torch.autograd import DeviceType
    flop_map = defaultdict(float)
    cuda_us = defaultdict(float)
    kernels_seen = 0
    events = prof.events()
    for ev in events:
        # PyTorch profiler attaches operator FLOPs to CPU dispatcher events.
        if ev.device_type == DeviceType.CPU:
            flop_map[_phase(ev)] += float(ev.flops or 0)
        elif ev.device_type == DeviceType.CUDA:
            kernels_seen += 1
            cuda_us[_phase(ev)] += float(ev.device_time_total or 0)

    # Older profiler versions sometimes omit standalone kernel records but
    # still report self CUDA time on operator records.
    if kernels_seen == 0:
        for ev in events:
            if ev.device_type == DeviceType.CPU:
                cuda_us[_phase(ev)] += float(ev.self_device_time_total or 0)

    groups = ('encoding_forward', 'probabilistic_forward', 'other_including_backward')
    result = {f'{g}_gflops_profiler': flop_map[g] / 1e9 for g in groups}
    result.update({f'{g}_cuda_kernel_ms_sum': cuda_us[g] / 1e3 for g in groups})
    result['gflops_profiler_per_update'] = sum(flop_map.values()) / 1e9
    result['cuda_kernel_ms_sum_per_update'] = sum(cuda_us.values()) / 1e3
    result['cuda_kernel_events'] = kernels_seen
    return result


def profile_one_update(trainer, batch, split=True):
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with _forward_profile_markers(trainer):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True) as prof:
            trainer.forward_backward(batch)
            torch.cuda.synchronize()
    return _profiler_breakdown(prof)


def _mean_profile(items):
    result = {}
    if not items:
        return result
    for key in items[0]:
        nums = [float(x[key]) for x in items]
        result[key] = statistics.mean(nums)
    return result


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

    max_epoch = int(trainer.max_epoch)
    if a.pretrain_epochs >= max_epoch:
        raise ValueError(f'pretrain-epochs={a.pretrain_epochs} must be < max_epoch={max_epoch}')
    n_batches = len(trainer.train_loader_x)
    if n_batches <= 0:
        raise ValueError('Empty train loader')
    print(f'[benchmark] {a.method} {a.dataset} seed={a.seed}, '
          f'pretrain={a.pretrain_epochs}, batches/epoch={n_batches}, max_epoch={max_epoch}')

    for epoch in range(a.pretrain_epochs):
        for batch in epoch_batches(trainer, epoch):
            trainer.forward_backward(batch)
        print(f'[pretrain] completed epoch {epoch+1}/{a.pretrain_epochs}')

    elapsed = []
    betas = []
    steps_seen = 0
    untimed_seen = 0
    first_measured_epoch = None
    last_measured_epoch = None
    measured_memory_reset = False
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
            torch.cuda.synchronize()
            start = time.perf_counter()
            trainer.forward_backward(batch)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1e3)
            betas.append(active_beta(trainer.method, epoch, max_epoch))
            if first_measured_epoch is None:
                first_measured_epoch = epoch
            last_measured_epoch = epoch
            steps_seen += 1
            if steps_seen >= a.steps:
                break
        if steps_seen >= a.steps:
            break
    if steps_seen < a.steps:
        raise RuntimeError(f'Only {steps_seen} measured steps fit into '
                           f'{max_epoch-a.pretrain_epochs} epochs; reduce steps/warmup')

    mem_alloc = torch.cuda.max_memory_allocated()/(1024**2)
    mem_reserved = torch.cuda.max_memory_reserved()/(1024**2)

    profiled = []
    profile_batch_sizes = []
    if a.profile_flops:
        # Separate post-timing optimizer updates; never contaminate latency.
        trainer.epoch = last_measured_epoch
        loader_iter = iter(trainer.train_loader_x)
        for i in range(a.profile_steps):
            trainer.batch_idx = i % n_batches
            if i and trainer.batch_idx == 0:
                loader_iter = iter(trainer.train_loader_x)
            batch = next(loader_iter)
            profile_batch_sizes.append(int(batch['img'].shape[0]))
            profiled.append(profile_one_update(trainer, batch))
    parts = _mean_profile(profiled)

    result = {
        'dataset': a.dataset, 'method': a.method, 'seed': a.seed,
        'gpu': torch.cuda.get_device_name(0),
        'num_shots': int(cfg.DATASET.NUM_SHOTS),
        'batch_size': int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
        'n_mc_train': (int(trainer.method.n_mc_train) if a.method == 'BayesRTMMRL' else None),
        'precision': str(trainer.method.get_precision()),
        'optimizer': str(cfg.OPTIM.NAME),
        'learning_rate': float(cfg.OPTIM.LR),
        'max_epoch': max_epoch,
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
        'warmup_steps': untimed_seen, 'steps': steps_seen,
        'loader_batches_per_epoch': n_batches,
        'ms_per_update': statistics.mean(elapsed),
        'ms_per_update_sample_std': statistics.stdev(elapsed) if len(elapsed)>1 else None,
        'updates_per_s': 1000.0/statistics.mean(elapsed),
        'peak_allocated_mib': mem_alloc,
        'peak_reserved_mib': mem_reserved,
        'gflops_profiler_per_update': parts.get('gflops_profiler_per_update'),
        'profile_steps': len(profiled),
        'profile_batch_sizes': profile_batch_sizes,
        'flops_scope': 'Only profiler-recognized ATen operation FLOPs, incomplete backward and uncounted operations.',
        'component_scope': 'Encoding/head FLOPs and CUDA kernel durations are FORWARD ONLY; backward, optimizer, common losses and untagged operations are counted in other_including_backward. Non-additive wall clock: kernel duration sums, not latency.',
        'timing_scope': 'CPU wall clock with CUDA synchronize, prefetched batch, full OnlineExecutor.forward_backward including metrics/logging',
        'memory_scope': 'PyTorch CUDA process allocated/reserved high-water mark from after untimed warmup through timed steps',
        'epoch_scope': 'Actually pretrained full epochs using same OnlineExecutor and learning-rate scheduler as regular run',
        'hpo_scope': 'HPO disabled; one configured fit, no grid search',
        'note': 'timed benchmark stops mid-training; profiling (if enabled) runs extra post-timing optimizer updates',
    }
    result.update(parts)
    destination = out / 'efficiency.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('Saved:', destination)


if __name__ == '__main__':
    main()
