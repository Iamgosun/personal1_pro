from __future__ import annotations

import torch

from core.types import EvalContext
from evaluation.metrics import (
    apply_temperature,
    build_classification_calibration_report,
    fit_temperature,
    save_metric_report,
    selective_prediction_report,
)
from evaluation.protocol_router import select_eval_logits as legacy_select_eval_logits


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

    def _select_eval_logits(self, outputs, eval_ctx):
        if hasattr(self.method, "select_eval_logits"):
            return self.method.select_eval_logits(outputs, eval_ctx)

        return legacy_select_eval_logits(
            self.method.method_name,
            outputs,
            eval_ctx,
        )

    def _collect_logits_and_labels(
        self,
        trainer,
        data_loader,
        eval_ctx,
        process_evaluator: bool = False,
    ):
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                image = batch["img"].to(trainer.device)
                label = batch["label"].to(trainer.device)

                outputs = self.method.forward_eval(
                    {
                        "img": image,
                        "label": label,
                    },
                    eval_ctx,
                )

                routed = self._select_eval_logits(outputs, eval_ctx)

                if process_evaluator:
                    trainer.evaluator.process(routed, label)

                all_logits.append(routed.detach().cpu())
                all_labels.append(label.detach().cpu())

        if len(all_logits) == 0:
            raise RuntimeError("No batches were found during evaluation.")

        logits = all_logits[0] if len(all_logits) == 1 else torch.cat(all_logits, dim=0)
        labels = all_labels[0] if len(all_labels) == 1 else torch.cat(all_labels, dim=0)

        return logits, labels

    def _add_common_report_fields(self, report, trainer, split, eval_ctx, legacy_results):
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
        return report

    def _maybe_add_temperature_scaling(
        self,
        report,
        trainer,
        split,
        logits,
        labels,
    ):
        """Fit T on val split and report calibrated metrics on test split.

        Temperature is learned only from validation logits. The test logits are
        never used to fit T, avoiding test-set leakage.
        """
        if split != "test":
            report["temperature_scaling"] = {
                "enabled": False,
                "reason": "Temperature is only fitted when evaluating the test split.",
            }
            return report

        if trainer.val_loader is None:
            report["temperature_scaling"] = {
                "enabled": False,
                "reason": "trainer.val_loader is None.",
            }
            return report

        val_ctx = self.build_eval_context(trainer, "val")
        val_logits, val_labels = self._collect_logits_and_labels(
            trainer=trainer,
            data_loader=trainer.val_loader,
            eval_ctx=val_ctx,
            process_evaluator=False,
        )

        temperature = fit_temperature(
            logits=val_logits,
            labels=val_labels,
            device=trainer.device,
        )

        calibrated_logits = apply_temperature(logits, temperature)

        calibrated_report = build_classification_calibration_report(
            logits=calibrated_logits,
            labels=labels,
            n_bins=10,
        )

        report["temperature_scaling"] = {
            "enabled": True,
            "temperature": temperature,
            "fit_split": "val",
            "eval_split": split,
            "metrics_before": report["metrics"],
            "metrics_after": calibrated_report["metrics"],
        }
        report["metrics_calibrated"] = calibrated_report["metrics"]
        report["prediction_calibrated"] = calibrated_report["prediction"]
        report["calibration_calibrated"] = calibrated_report["calibration"]
        report["selective_prediction_calibrated"] = calibrated_report["selective_prediction"]

        return report

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

        logits, labels = self._collect_logits_and_labels(
            trainer=trainer,
            data_loader=data_loader,
            eval_ctx=eval_ctx,
            process_evaluator=True,
        )

        legacy_results = trainer.evaluator.evaluate()

        report = build_classification_calibration_report(
            logits=logits,
            labels=labels,
            n_bins=10,
        )

        report = self._add_common_report_fields(
            report=report,
            trainer=trainer,
            split=split,
            eval_ctx=eval_ctx,
            legacy_results=legacy_results,
        )

        report = self._maybe_add_temperature_scaling(
            report=report,
            trainer=trainer,
            split=split,
            logits=logits,
            labels=labels,
        )

        saved_paths = save_metric_report(trainer.cfg.OUTPUT_DIR, split, report)

        for k, v in report["metrics"].items():
            trainer.write_scalar(f"{split}/{k}", v, trainer.epoch)

        if "metrics_calibrated" in report:
            for k, v in report["metrics_calibrated"].items():
                trainer.write_scalar(f"{split}/{k}_calibrated", v, trainer.epoch)

        print("=> structured result")
        for k, v in report["metrics"].items():
            if isinstance(v, float):
                print(f"* {k}: {v:.4f}")
            else:
                print(f"* {k}: {v}")

        if "metrics_calibrated" in report:
            temperature = report["temperature_scaling"]["temperature"]
            print(f"=> temperature scaling: T = {temperature:.6f}")
            for k, v in report["metrics_calibrated"].items():
                if isinstance(v, float):
                    print(f"* {k}_calibrated: {v:.4f}")
                else:
                    print(f"* {k}_calibrated: {v}")

        print(f"Saved metrics JSON to {saved_paths['json']}")
        print(f"Saved metrics CSV to {saved_paths['metrics_csv']}")
        print(f"Saved calibration bins CSV to {saved_paths['bins_csv']}")

        if "metrics_calibrated_csv" in saved_paths:
            print(
                "Saved calibrated metrics CSV to "
                f"{saved_paths['metrics_calibrated_csv']}"
            )

        if "bins_calibrated_csv" in saved_paths:
            print(
                "Saved calibrated calibration bins CSV to "
                f"{saved_paths['bins_calibrated_csv']}"
            )

        if "metrics_calibrated" in report:
            return float(report["metrics_calibrated"]["accuracy"])

        return float(report["metrics"]["accuracy"])