from __future__ import annotations

import torch
from torch.cuda.amp import autocast

from core.registry import EXECUTOR_REGISTRY
from dassl.metrics import compute_accuracy

from .base_executor import BaseExecutor


@EXECUTOR_REGISTRY.register("online")
class OnlineExecutor(BaseExecutor):
    exec_mode = "online"

    def forward_backward(self, trainer, batch):
        prec = self.method.get_precision()
        if hasattr(self.method, "set_epoch_context"):
            self.method.set_epoch_context(trainer.epoch, trainer.max_epoch)
        payload = {
            "img": batch["img"].to(trainer.device),
            "label": batch["label"].to(trainer.device),
        }

        if hasattr(self.method, "set_kl_normalizer"):
            self.method.set_kl_normalizer(getattr(trainer, "num_batches", 1))



        if hasattr(self.method, "set_kl_beta"):
            warmup_epochs = int(getattr(self.method, "kl_warmup_epochs", 0))
            cur_epoch = int(trainer.epoch)          # 0-based
            total_epochs = int(trainer.max_epoch)   # 通常是 50

            if warmup_epochs <= 0:
                kl_beta = 1.0
            elif cur_epoch < warmup_epochs:
                # 前 warmup_epochs 个 epoch 完全不加 KL
                kl_beta = 0.0
            else:
                # 剩余 epoch 从 0 -> 1，最后一个 epoch 到 1
                ramp_epochs = max(1, total_epochs - warmup_epochs - 1)
                kl_beta = min(1.0, float(cur_epoch - warmup_epochs) / float(ramp_epochs))

            self.method.set_kl_beta(kl_beta)

        if prec == "amp":
            with autocast():
                outputs = self.method.forward_train(payload)
                loss = self.method.loss(outputs)

            trainer.optim.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optim)
            trainer.scaler.update()
        else:
            outputs = self.method.forward_train(payload)
            loss = self.method.loss(outputs)

            trainer.optim.zero_grad()
            loss.backward()
            trainer.optim.step()

        train_logits = self.method.select_train_logits(outputs)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(train_logits, outputs.labels)[0].item(),
        }

        if hasattr(outputs, "losses") and outputs.losses is not None:
            for key in [
                "data_term",
                "raw_kl_rep",
                "raw_kl_proj_rep",
                "kl_rep_term",
                "kl_proj_rep_term",
                "kl_term",
                "kl_normalizer",
                "kl_beta",
            ]:
                if key in outputs.losses:
                    value = outputs.losses[key]
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    loss_summary[key] = float(value)

        if (trainer.batch_idx + 1) == trainer.num_batches:
            trainer.update_lr()

        return loss_summary