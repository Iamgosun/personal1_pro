from __future__ import annotations

import os
import os.path as osp
import copy
import csv
import json
import torch
import torch.nn as nn
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

