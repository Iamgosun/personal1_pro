from __future__ import annotations

import os
import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from core.registry import EXECUTOR_REGISTRY, METHOD_REGISTRY

# ensure method / executor registration side effects
import methods.mmrl  # noqa: F401
import methods.mmrlpp  # noqa: F401
import methods.clip_adapters  # noqa: F401
import executors.online_executor  # noqa: F401
import executors.cache_executor  # noqa: F401


@TRAINER_REGISTRY.register()
class RefactorRunner(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.METHOD.EXEC_MODE in {'online', 'cache'}

    def build_model(self):
        method_cls = METHOD_REGISTRY.get(self.cfg.METHOD.NAME)
        self.method = method_cls(self.cfg, self.dm, self.device).build()
        self.model = self.method.model
        if self.cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, self.cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)

        # move auxiliary modules if they exist
        for attr in ['image_encoder_clip', 'text_encoder_clip']:
            module = getattr(self.method, attr, None)
            if module is not None and hasattr(module, 'to'):
                module.to(self.device)

        optim_target = self.method.get_optimizer_target()
        self.optim = build_optimizer(optim_target, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model('refactor_model', self.model, self.optim, self.sched)

        prec = 'fp32'
        if self.cfg.METHOD.NAME == 'MMRL':
            prec = self.cfg.MMRL.PREC
        elif self.cfg.METHOD.NAME in {'MMRLpp', 'MMRLPP'}:
            prec = self.cfg.MMRLPP.PREC
        elif self.cfg.METHOD.NAME in {'ClipAdapters', 'ClipADAPTER'}:
            prec = self.cfg.CLIP_ADAPTERS.PREC
        self.scaler = GradScaler() if prec == 'amp' else None

        self.executor = EXECUTOR_REGISTRY.get(self.cfg.METHOD.EXEC_MODE)(self.method)
        self.executor.on_build(self)

        device_count = torch.cuda.device_count()
        if device_count > 1 and self.cfg.USE_CUDA:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        return self.executor.forward_backward(self, batch)

    @torch.no_grad()
    def test(self, split=None):
        return self.executor.test(self, split=split)

    def run_epoch(self):
        if self.cfg.METHOD.EXEC_MODE == 'cache':
            return self.executor.run_epoch(self)
        return super().run_epoch()

    def load_model(self, directory, epoch=None):
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return
        names = self.get_model_names()
        for name in names:
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(f'Model not found at "{model_path_prefix}"')
            model_path = None
            for file in os.listdir(model_path_prefix):
                if 'model-best.pth' in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if 'model.pth' in file:
                    model_path = osp.join(model_path_prefix, file)
            if model_path is None or not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path_prefix}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
