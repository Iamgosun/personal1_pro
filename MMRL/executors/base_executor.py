from __future__ import annotations

from core.types import EvalContext
from evaluation.metrics import (
    build_classification_calibration_report,
    save_metric_report,
)
from evaluation.protocol_router import select_eval_logits


class BaseExecutor:
    exec_mode = "base"

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
            subsample_classes=getattr(trainer.cfg.DATASET, "SUBSAMPLE_CLASSES", "all"),
            phase=getattr(trainer.cfg.PROTOCOL, "PHASE", None),
        )

    def test(self, trainer, split=None):
        trainer.set_model_mode("eval")
        trainer.evaluator.reset()

        if split is None:
            split = trainer.cfg.TEST.SPLIT

        if split == "val" and trainer.val_loader is not None:
            data_loader = trainer.val_loader
        else:
            split = "test"
            data_loader = trainer.test_loader

        eval_ctx = self.build_eval_context(trainer, split)
        print(f"Evaluate on the *{split}* set")

        all_logits = []
        all_labels = []

        for batch in data_loader:
            image = batch["img"].to(trainer.device)
            label = batch["label"].to(trainer.device)

            outputs = self.method.forward_eval({"img": image, "label": label}, eval_ctx)
            routed = select_eval_logits(self.method.method_name, outputs, eval_ctx)

            trainer.evaluator.process(routed, label)

            all_logits.append(routed.detach().cpu())
            all_labels.append(label.detach().cpu())

        legacy_results = trainer.evaluator.evaluate()

        logits = all_logits[0] if len(all_logits) == 1 else __import__("torch").cat(all_logits, dim=0)
        labels = all_labels[0] if len(all_labels) == 1 else __import__("torch").cat(all_labels, dim=0)

        report = build_classification_calibration_report(
            logits=logits,
            labels=labels,
            n_bins=15,
        )

        report["method_name"] = self.method.method_name
        report["split"] = split
        report["protocol"] = eval_ctx.protocol
        report["dataset_name"] = eval_ctx.dataset_name
        report["subsample_classes"] = eval_ctx.subsample_classes
        report["phase"] = eval_ctx.phase
        report["legacy_evaluator"] = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in legacy_results.items()
        }

        saved_paths = save_metric_report(trainer.cfg.OUTPUT_DIR, split, report)

        for k, v in report["metrics"].items():
            trainer.write_scalar(f"{split}/{k}", v, trainer.epoch)

        print("=> structured result")
        for k, v in report["metrics"].items():
            if isinstance(v, float):
                print(f"* {k}: {v:.4f}")
            else:
                print(f"* {k}: {v}")

        print(f"Saved metrics JSON to {saved_paths['json']}")
        print(f"Saved metrics CSV to {saved_paths['metrics_csv']}")
        print(f"Saved calibration bins CSV to {saved_paths['bins_csv']}")

        return float(report["metrics"]["accuracy"])