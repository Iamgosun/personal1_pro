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
from sklearn.model_selection import ParameterGrid



# ensure method / executor registration side effects
import methods.mmrl  # noqa: F401
import methods.mmrl_mix  # noqa: F401
import methods.mmrlpp  # noqa: F401
import methods.bayes_mmrl  # noqa: F401
import methods.clip_adapters  # noqa: F401
import executors.online_executor  # noqa: F401
import executors.cache_executor  # noqa: F401
import methods.vcrm_mmrl  # noqa: F401

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
            "image_encoder.bayes_proj_rep",
            "image_encoder.A.",
            "image_encoder.B.",
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
            "image_encoder.bayes_proj_rep",
            "image_encoder.A.",
            "image_encoder.B.",
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
            

    def _clip_adapter_grid_search_enabled(self) -> bool:
        if not hasattr(self.cfg, "CLIP_ADAPTERS"):
            return False

        init_name = str(getattr(self.cfg.CLIP_ADAPTERS, "INIT", ""))
        explicit = bool(getattr(self.cfg.CLIP_ADAPTERS, "GRID_SEARCH", False))
        legacy = "grid_search" in init_name.lower()
        return explicit or legacy

    def _set_optimizer_lr_from_grid(self, params):
        if "lr" not in params:
            return

        self.cfg.defrost()
        self.cfg.OPTIM.LR = float(params["lr"])
        self.cfg.freeze()

    def _rebuild_optimizer_for_grid(self):
        optim_target = self.method.get_optimizer_target()
        self.optim = build_optimizer(optim_target, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)

        # Update DASSL bookkeeping without re-registering duplicate names.
        if hasattr(self, "_optims"):
            self._optims["refactor_model"] = self.optim
        if hasattr(self, "_scheds"):
            self._scheds["refactor_model"] = self.sched

        prec = self.method.get_precision()
        self.scaler = GradScaler() if prec == "amp" else None

    def _reset_adapter_for_grid(self, params):
        adapter = self.method.model.adapter

        if hasattr(adapter, "reset_for_grid"):
            adapter.reset_for_grid(
                params,
                features_train=getattr(self, "features_train", None),
                labels_train=getattr(self, "labels_train", None),
            )
        elif hasattr(adapter, "reset_hparams"):
            adapter.reset_hparams(params)

        # Reinitialize CLAP constraint multipliers after resetting params.
        if getattr(adapter, "apply_constraint", "none") != "none":
            adapter.init_lagrangian_multipliers(
                self.labels_train.to(self.device),
                self.logits_train.to(self.device),
            )

        self._set_optimizer_lr_from_grid(params)
        self._rebuild_optimizer_for_grid()

    def _grid_max_epoch(self, params, default_max_epoch):
        adapter = self.method.model.adapter

        if bool(getattr(adapter, "is_tip_adapter", False)):
            if bool(getattr(adapter, "finetune_cache", False)):
                return int(getattr(self.cfg.CLIP_ADAPTERS, "TIPA_F_GRID_EPOCHS", 20))
            return 1

        return int(default_max_epoch)

    def _write_grid_summary(self, rows, best_row):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        json_path = osp.join(self.cfg.OUTPUT_DIR, "grid_search_summary.json")
        csv_path = osp.join(self.cfg.OUTPUT_DIR, "grid_search_summary.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rows": rows,
                    "best": best_row,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if rows:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"[GridSearch] saved summary to {json_path}")
        print(f"[GridSearch] saved summary to {csv_path}")

    def _run_clip_adapter_grid_search(self):
        if self.cfg.METHOD.EXEC_MODE != "cache":
            raise RuntimeError(
                "CLIP_ADAPTERS.GRID_SEARCH requires METHOD.EXEC_MODE=cache "
                "because CLAP-style adapter baselines are feature-cache based."
            )

        adapter = self.method.model.adapter
        grid = getattr(adapter, "grid_search_param", None)

        if not grid:
            raise RuntimeError(
                f"Grid search requested, but {adapter.__class__.__name__} "
                "does not define grid_search_param."
            )

        original_max_epoch = int(self.max_epoch)
        original_lr = float(self.cfg.OPTIM.LR)

        best_score = -1.0
        best_state = None
        best_row = None
        rows = []

        split = str(getattr(self.cfg.CLIP_ADAPTERS, "GRID_SEARCH_SPLIT", "val"))
        if split == "val" and self.val_loader is None:
            split = "test"

        print(f"[GridSearch] split={split}")
        print(f"[GridSearch] grid={grid}")

        self.before_train()

        for i, params in enumerate(ParameterGrid(grid), start=1):
            params = dict(params)
            print(f"[GridSearch] candidate {i}: {params}")

            self._reset_adapter_for_grid(params)

            self.start_epoch = 0
            self.epoch = 0
            self.max_epoch = self._grid_max_epoch(params, original_max_epoch)

            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()

                if "adaptative" in getattr(adapter, "apply_constraint", "none"):
                    adapter.outer_step()

                self.after_epoch()

            score = float(self.test(split=split))

            row = {k: v for k, v in params.items()}
            row["score"] = score
            row["split"] = split
            row["max_epoch"] = self.max_epoch
            rows.append(row)

            print(f"[GridSearch] candidate {i} score={score:.4f}")

            if score > best_score:
                best_score = score
                best_row = copy.deepcopy(row)
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)
            print(f"[GridSearch] best={best_row}")

        self._write_grid_summary(rows, best_row)

        self.cfg.defrost()
        self.cfg.OPTIM.LR = original_lr
        self.cfg.freeze()

        self.max_epoch = original_max_epoch
        self.after_train()

        return best_score


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
        if self._clip_adapter_grid_search_enabled():
            return self._run_clip_adapter_grid_search()
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

