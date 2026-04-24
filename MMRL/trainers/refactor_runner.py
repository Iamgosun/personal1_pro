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
import methods.mmrl_mix  # noqa: F401
import methods.mmrlpp  # noqa: F401
import methods.bayes_mmrl  # noqa: F401
import methods.clip_adapters  # noqa: F401
import executors.online_executor  # noqa: F401
import executors.cache_executor  # noqa: F401


@TRAINER_REGISTRY.register()
class RefactorRunner(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.METHOD.EXEC_MODE in {"online", "cache"}

    def build_model(self):
        method_cls = METHOD_REGISTRY.get(self.cfg.METHOD.NAME)
        self.method = method_cls(self.cfg, self.dm, self.device).build()
        self.model = self.method.model

        if self.cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, self.cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # move auxiliary modules if they exist
        for attr in ["image_encoder_clip", "text_encoder_clip"]:
            module = getattr(self.method, attr, None)
            if module is not None and hasattr(module, "to"):
                module.to(self.device)

        # Build executor first so executor-side setup can run before optimizer creation.
        self.executor = EXECUTOR_REGISTRY.get(self.cfg.METHOD.EXEC_MODE)(self.method)
        self.executor.on_build(self)

        # Method-level pre-fit hook.
        # This is the right place for method-family specific initialization that must
        # happen before optimizer creation, e.g. adapter-family cache prebuild.
        self.method.on_fit_start(self)

        optim_target = self.method.get_optimizer_target()
        self.optim = build_optimizer(optim_target, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model("refactor_model", self.model, self.optim, self.sched)

        prec = self.method.get_precision()
        self.scaler = GradScaler() if prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1 and self.cfg.USE_CUDA:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        return self.executor.forward_backward(self, batch)

    @torch.no_grad()
    def test(self, split=None):
        return self.executor.test(self, split=split)

    def run_epoch(self):
        if self.cfg.METHOD.EXEC_MODE == "cache":
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

            # 优先加载 best；否则加载指定 epoch；否则加载最后一个 model.pth*
            files = sorted(os.listdir(model_path_prefix))

            if epoch is not None:
                target = f"model.pth.tar-{epoch}"
                for file in files:
                    if file == target:
                        model_path = osp.join(model_path_prefix, file)
                        break
            else:
                for file in files:
                    if "model-best.pth" in file:
                        model_path = osp.join(model_path_prefix, file)
                        break

                if model_path is None:
                    for file in files:
                        if "model.pth" in file:
                            model_path = osp.join(model_path_prefix, file)

            if model_path is None or not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path_prefix}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            ckpt_epoch = checkpoint["epoch"]

            # These buffers depend on current classnames.
            # In B2N, train_base and test_new have different class sets,
            # so they must be rebuilt from the current dataset, not loaded
            # from the base-class checkpoint.
            skip_keywords = (
                "prompt_embeddings",
                "tokenized_prompts",
            )

            current_state = self._models[name].state_dict()
            filtered_state_dict = {}
            skipped_keys = []

            for k, v in state_dict.items():
                if any(s in k for s in skip_keywords):
                    skipped_keys.append(k)
                    continue

                if k in current_state and current_state[k].shape != v.shape:
                    skipped_keys.append(k)
                    continue

                filtered_state_dict[k] = v

            print(f'Loading weights to {name} from "{model_path}" (epoch = {ckpt_epoch})')

            if skipped_keys:
                print("Skipped class-dependent or shape-mismatched keys:")
                for k in skipped_keys:
                    print(f"  - {k}")

            incompatible = self._models[name].load_state_dict(
                filtered_state_dict,
                strict=False,
            )

            if incompatible.missing_keys:
                print("Missing keys after loading:")
                for k in incompatible.missing_keys:
                    print(f"  - {k}")

            if incompatible.unexpected_keys:
                print("Unexpected keys after loading:")
                for k in incompatible.unexpected_keys:
                    print(f"  - {k}")

