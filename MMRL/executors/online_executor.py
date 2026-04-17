from __future__ import annotations

from torch.cuda.amp import autocast

from core.registry import EXECUTOR_REGISTRY
from dassl.metrics import compute_accuracy

from .base_executor import BaseExecutor


@EXECUTOR_REGISTRY.register('online')
class OnlineExecutor(BaseExecutor):
    exec_mode = 'online'

    def forward_backward(self, trainer, batch):
        prec = getattr(getattr(trainer.cfg, trainer.method.method_name if trainer.method.method_name in {'MMRL', 'MMRLpp'} else 'CLIP_ADAPTERS', None), 'PREC', 'fp32')
        if trainer.cfg.METHOD.NAME in {'MMRL', 'MMRLpp', 'MMRLPP'}:
            prec = trainer.cfg.MMRL.PREC if trainer.cfg.METHOD.NAME == 'MMRL' else trainer.cfg.MMRLPP.PREC
        else:
            prec = trainer.cfg.CLIP_ADAPTERS.PREC

        payload = {'img': batch['img'].to(trainer.device), 'label': batch['label'].to(trainer.device)}
        if prec == 'amp':
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

        train_logits = outputs.aux_logits.get('fusion', outputs.logits)
        loss_summary = {'loss': loss.item(), 'acc': compute_accuracy(train_logits, outputs.labels)[0].item()}
        if (trainer.batch_idx + 1) == trainer.num_batches:
            trainer.update_lr()
        return loss_summary
