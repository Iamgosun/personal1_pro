from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")

import torch

torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))

import argparse
import importlib


from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger

from core.config import setup_cfg
from core.utils import import_optional_modules

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    for key, val in sorted(vars(args).items()):
        print(f'{key}: {val}')
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def main(args):
    # Datasets are optional because users may not have every dataset module/file ready.
    import_optional_modules([
        'datasets.oxford_pets', 'datasets.oxford_flowers', 'datasets.fgvc_aircraft',
        'datasets.dtd', 'datasets.eurosat', 'datasets.stanford_cars', 'datasets.food101',
        'datasets.sun397', 'datasets.caltech101', 'datasets.ucf101', 'datasets.imagenet',
        'datasets.imagenetv2', 'datasets.imagenet_sketch', 'datasets.imagenet_a', 'datasets.imagenet_r',
    ])

    # Trainer registration is NOT optional. Fail loudly so registry errors are debuggable.
    importlib.import_module('trainers.refactor_runner')

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f'Setting fixed seed: {cfg.SEED}')
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--dataset-config-file', type=str, default='')
    parser.add_argument('--method-config-file', type=str, default='')
    parser.add_argument('--protocol-config-file', type=str, default='')
    parser.add_argument('--runtime-config-file', type=str, default='')
    parser.add_argument('--exp-config', type=str, default='')
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--protocol', type=str, default='')
    parser.add_argument('--exec-mode', type=str, default='')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--trainer', type=str, default='RefactorRunner')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--model-dir', type=str, default='')
    parser.add_argument('--load-epoch', type=int, default=None)
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    main(parser.parse_args())
