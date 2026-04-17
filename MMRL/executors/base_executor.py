from __future__ import annotations

from core.types import EvalContext
from evaluation.metrics import accuracy
from evaluation.protocol_router import select_eval_logits


class BaseExecutor:
    exec_mode = 'base'

    def __init__(self, method):
        self.method = method

    def on_build(self, trainer):
        return None

    def forward_backward(self, trainer, batch):
        raise NotImplementedError

    def build_eval_context(self, trainer, split: str):
        return EvalContext(
            protocol=trainer.cfg.PROTOCOL.NAME,
            dataset_name=trainer.cfg.DATASET.NAME,
            split=split,
            subsample_classes=getattr(trainer.cfg.DATASET, 'SUBSAMPLE_CLASSES', 'all'),
            phase=getattr(trainer.cfg.PROTOCOL, 'PHASE', None),
        )

    def test(self, trainer, split=None):
        trainer.set_model_mode('eval')
        trainer.evaluator.reset()
        if split is None:
            split = trainer.cfg.TEST.SPLIT
        if split == 'val' and trainer.val_loader is not None:
            data_loader = trainer.val_loader
        else:
            split = 'test'
            data_loader = trainer.test_loader
        eval_ctx = self.build_eval_context(trainer, split)
        print(f'Evaluate on the *{split}* set')
        for batch in data_loader:
            image = batch['img'].to(trainer.device)
            label = batch['label'].to(trainer.device)
            outputs = self.method.forward_eval({'img': image, 'label': label}, eval_ctx)
            routed = select_eval_logits(self.method.method_name, outputs, eval_ctx)
            trainer.evaluator.process(routed, label)
        results = trainer.evaluator.evaluate()
        for k, v in results.items():
            trainer.write_scalar(f'{split}/{k}', v, trainer.epoch)
        return list(results.values())[0]
