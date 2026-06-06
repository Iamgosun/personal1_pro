from __future__ import annotations

import os
import os.path as osp
import copy
import csv
import json
import torch
import torch.nn as nn


import shutil



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
import methods.vcrm_mmrl  # noqa: F401
import methods.bayes_text_mmrl  # noqa: F401
import methods.bayesrt_mmrl  # noqa: F401
import methods.det_bayesrt_mmrl  # noqa: F401
import methods.fused_det_bayesrt_mmrl

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

      
    def _lightweight_keep_prefixes(self):
        """
        Only save method-specific trainable/lightweight parameters.

        Full CLIP weights are intentionally not saved. CLIP is rebuilt from
        pretrained weights before load_model(), then these lightweight weights
        are loaded on top with strict=False.

        ClipAdapters / CLAP / CAPEL:
            adapter.*

        MMRL / MMRLMix:
            representation_learner.*
            image_encoder.proj_rep.*

        BayesMMRL:
            representation_learner.*
            image_encoder.proj_rep.*
            image_encoder.bayes_proj_rep.*

        MMRLpp:
            representation_learner.*
            image_encoder.proj_rep.*
            image_encoder.A.*
            image_encoder.B.*
        """
        model = getattr(self.method, "model", None)

        if model is not None and hasattr(model, "adapter"):
            return ("adapter.",)

        return (
            "representation_learner.",
            "image_encoder.proj_rep",
            "image_encoder.visual.proj_rep",
            "image_encoder.bayes_proj_rep",
            "image_encoder.A.",
            "image_encoder.B.",
            "text_posterior.",
        )



    @staticmethod
    def _strip_module_prefix(key):
        if key.startswith("module."):
            return key[len("module."):]
        return key

    def _is_lightweight_key(self, key):
        key = self._strip_module_prefix(key)
        return key.startswith(self._lightweight_keep_prefixes())

    @staticmethod
    def _state_dict_nbytes(state_dict):
        total = 0
        for value in state_dict.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
        return total

    def _to_cpu_state_dict(self, state_dict):
        out = {}
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu()
            else:
                out[k] = v
        return out

    def _filter_state_dict_for_lightweight_checkpoint(self, state_dict):
        filtered = {
            k: v
            for k, v in state_dict.items()
            if self._is_lightweight_key(k)
        }

        if not filtered:
            print(
                "[LightweightCheckpoint] WARNING: filtered state_dict is empty; "
                "falling back to full state_dict."
            )
            return self._to_cpu_state_dict(state_dict)

        full_mb = self._state_dict_nbytes(state_dict) / (1024 ** 2)
        light_mb = self._state_dict_nbytes(filtered) / (1024 ** 2)

        print(
            "[LightweightCheckpoint] filtered state_dict: "
            f"{len(filtered)}/{len(state_dict)} tensors, "
            f"{light_mb:.2f} MB / {full_mb:.2f} MB"
        )

        return self._to_cpu_state_dict(filtered)

    def _is_expected_lightweight_missing_key(self, key):
        key = self._strip_module_prefix(key)


        trainable_prefixes = (
            "adapter.",
            "representation_learner.",
            "image_encoder.proj_rep",
            "image_encoder.visual.proj_rep",
            "image_encoder.bayes_proj_rep",
            "image_encoder.A.",
            "image_encoder.B.",
            "text_posterior.",
        )


        if key.startswith(trainable_prefixes):
            return False

        expected_missing_prefixes = (
            "image_encoder.",
            "text_encoder.",
            "tokenized_prompts",
            "prompt_embeddings",
            "logit_scale",
            "base_text_features",
            "text_embeddings_all",
        )

        return key.startswith(expected_missing_prefixes)

    
    def _is_b2n_test_new(self) -> bool:
        return (
            str(getattr(self.cfg.PROTOCOL, "NAME", "")).upper() == "B2N"
            and str(getattr(self.cfg.PROTOCOL, "PHASE", "")) == "test_new"
        )

    def _should_reinit_clip_adapter_for_b2n_test_new(self) -> bool:
        if not self._is_b2n_test_new():
            return False

        model = getattr(self.method, "model", None)
        if model is None:
            return False

        # ClipAdaptersModel owns `adapter`.
        return hasattr(model, "adapter")
    
    
    
    
    
    
    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        """
        Override Dassl TrainerX.save_model().

        Save only method-specific lightweight weights, not the frozen CLIP
        backbone. This makes checkpoints much smaller.

        File layout remains:
            <OUTPUT_DIR>/refactor_model/model.pth.tar-<epoch>
            <OUTPUT_DIR>/refactor_model/model-best.pth.tar
        """
        names = self.get_model_names()

        for name in names:
            model = self._models[name]

            state_dict = model.state_dict()
            state_dict = self._filter_state_dict_for_lightweight_checkpoint(state_dict)

            save_dir = osp.join(directory, name)
            os.makedirs(save_dir, exist_ok=True)

            if model_name:
                model_file = model_name
            else:
                model_file = f"model.pth.tar-{epoch + 1}"

            save_path = osp.join(save_dir, model_file)

            checkpoint = {
                "state_dict": state_dict,
                "epoch": epoch + 1,
                "val_result": val_result,
                "lightweight_checkpoint": True,
                "method_name": str(getattr(self.method, "method_name", "")),
                "cfg_method_name": str(getattr(self.cfg.METHOD, "NAME", "")),
            }

            # Deliberately do NOT save optimizer/scheduler here.
            # They are not needed for eval-only or B2N test_new, and can make
            # checkpoint files much larger.
            torch.save(checkpoint, save_path)
            print(f"[LightweightCheckpoint] saved to {save_path}")

            if is_best:
                best_path = osp.join(save_dir, "model-best.pth.tar")
                shutil.copyfile(save_path, best_path)
                print(f"[LightweightCheckpoint] copied best to {best_path}")       
            


    def forward_backward(self, batch):
        return self.executor.forward_backward(self, batch)

    @torch.no_grad()
    def test(self, split=None):
        return self.executor.test(self, split=split)

    def run_epoch(self):
        if self.cfg.METHOD.EXEC_MODE == "cache":
            return self.executor.run_epoch(self)
        return super().run_epoch()


    def train(self):
        return super().train()

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

            reinit_clip_adapter = self._should_reinit_clip_adapter_for_b2n_test_new()

            for k, v in state_dict.items():
                # B2N test_new for ClipAdapters:
                # The adapter parameters are class-specific. They were trained
                # on base classes and must not be loaded into the new-class model.
                # Keep the new adapter initialized from current new-class classnames.
                key_no_module = self._strip_module_prefix(k)
                if reinit_clip_adapter and key_no_module.startswith("adapter."):
                    skipped_keys.append(k)
                    continue

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

            missing_keys = [
                k
                for k in incompatible.missing_keys
                if not self._is_expected_lightweight_missing_key(k)
            ]

            expected_missing = [
                k
                for k in incompatible.missing_keys
                if self._is_expected_lightweight_missing_key(k)
            ]

            if missing_keys:
                print("Missing keys after loading:")
                for k in missing_keys:
                    print(f"  - {k}")

            if expected_missing:
                print(
                    "[LightweightCheckpoint] ignored expected missing frozen/backbone keys: "
                    f"{len(expected_missing)}"
                )

            if incompatible.unexpected_keys:
                print("Unexpected keys after loading:")
                for k in incompatible.unexpected_keys:
                    print(f"  - {k}")

