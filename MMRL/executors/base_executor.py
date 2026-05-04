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

    def _should_report_fusion_variants(self, trainer) -> bool:
        """
        Only BayesMMRL reports C/R fusion variants.

        Required BayesMMRL forward_eval aux_logits:
            - rep
            - fusion
            - fusion_static
            - fusion_dynamic_no_beta
        """
        if getattr(self.method, "method_name", "") != "BayesMMRL":
            return False

        if not hasattr(trainer.cfg, "BAYES_MMRL"):
            return False

        return bool(
            getattr(
                trainer.cfg.BAYES_MMRL,
                "REPORT_FUSION_VARIANTS",
                False,
            )
        )

    @staticmethod
    def _cat_or_first(chunks):
        if len(chunks) == 0:
            return None
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    def _collect_logits_and_labels(
        self,
        trainer,
        data_loader,
        eval_ctx,
        process_evaluator: bool = False,
        collect_fusion_variants: bool = False,
    ):
        """
        Efficient collection path.

        In a single forward_eval pass, this collects:
            1. routed logits used by the official evaluator
            2. labels
            3. optional fusion-variant logits for reporting

        No second evaluation pass is performed.

        Returns:
            if collect_fusion_variants is False:
                logits, labels

            if collect_fusion_variants is True:
                logits, labels, variant_logits

        variant_logits keys:
            - main
            - rep
            - fusion_static
            - fusion_dynamic_no_beta
            - fusion
        """
        all_logits = []
        all_labels = []

        variant_keys = [
            "main",
            "rep",
            "fusion_static",
            "fusion_dynamic_no_beta",
            "fusion",
        ]
        all_variant_logits = {key: [] for key in variant_keys}

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

                if collect_fusion_variants:
                    for key in variant_keys:
                        if key == "main":
                            logits_v = outputs.logits
                        else:
                            logits_v = outputs.aux_logits.get(key)

                        if logits_v is not None:
                            all_variant_logits[key].append(logits_v.detach().cpu())

        if len(all_logits) == 0:
            raise RuntimeError("No batches were found during evaluation.")

        logits = self._cat_or_first(all_logits)
        labels = self._cat_or_first(all_labels)

        if not collect_fusion_variants:
            return logits, labels

        variant_logits = {}
        for key, chunks in all_variant_logits.items():
            packed = self._cat_or_first(chunks)
            if packed is not None:
                variant_logits[key] = packed

        return logits, labels, variant_logits

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
            collect_fusion_variants=False,
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
        report["selective_prediction_calibrated"] = calibrated_report[
            "selective_prediction"
        ]

        return report

    @staticmethod
    def _add_fusion_variant_reports(report, variant_logits, labels):
        """
        Add full metrics for each fusion variant into JSON report.

        Also flatten key metrics into report["metrics"] so that the existing
        metrics CSV contains static and dynamic fusion results without modifying
        evaluation/metrics.py.

        Added JSON field:
            report["fusion_variants"][variant_name]

        Added CSV/top-level metric keys:
            fusion_variant_<variant_name>_<metric_name>
        """
        if not variant_logits:
            return report

        report["fusion_variants"] = {}

        for variant_name, logits_v in variant_logits.items():
            variant_report = build_classification_calibration_report(
                logits=logits_v,
                labels=labels,
                n_bins=10,
            )

            report["fusion_variants"][variant_name] = {
                "metrics": variant_report["metrics"],
                "prediction": variant_report["prediction"],
                "calibration": variant_report["calibration"],
                "selective_prediction": variant_report["selective_prediction"],
            }

            for metric_name, metric_value in variant_report["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    flat_key = f"fusion_variant_{variant_name}_{metric_name}"
                    report["metrics"][flat_key] = metric_value

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

        collect_fusion_variants = self._should_report_fusion_variants(trainer)

        collected = self._collect_logits_and_labels(
            trainer=trainer,
            data_loader=data_loader,
            eval_ctx=eval_ctx,
            process_evaluator=True,
            collect_fusion_variants=collect_fusion_variants,
        )

        if collect_fusion_variants:
            logits, labels, fusion_variant_logits = collected
        else:
            logits, labels = collected
            fusion_variant_logits = {}

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

        report = self._add_fusion_variant_reports(
            report=report,
            variant_logits=fusion_variant_logits,
            labels=labels,
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

        if "fusion_variants" in report:
            for variant_name, variant_report in report["fusion_variants"].items():
                metrics = variant_report.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        trainer.write_scalar(
                            f"{split}/fusion_variants/{variant_name}/{k}",
                            v,
                            trainer.epoch,
                        )

        if "metrics_calibrated" in report:
            for k, v in report["metrics_calibrated"].items():
                trainer.write_scalar(f"{split}/{k}_calibrated", v, trainer.epoch)

        print("=> structured result")
        for k, v in report["metrics"].items():
            if isinstance(v, float):
                print(f"* {k}: {v:.4f}")
            else:
                print(f"* {k}: {v}")

        if "fusion_variants" in report:
            print("=> fusion variants")
            for variant_name, variant_report in report["fusion_variants"].items():
                metrics = variant_report.get("metrics", {})
                acc = metrics.get("accuracy")
                nll = metrics.get("nll")
                ece = metrics.get("ece")

                parts = []
                if isinstance(acc, float):
                    parts.append(f"accuracy={acc:.4f}")
                if isinstance(nll, float):
                    parts.append(f"nll={nll:.4f}")
                if isinstance(ece, float):
                    parts.append(f"ece={ece:.4f}")

                if parts:
                    print(f"* {variant_name}: " + ", ".join(parts))
                else:
                    print(f"* {variant_name}: {metrics}")

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

        if "risk_coverage_curve_csv" in saved_paths:
            print(
                "Saved risk-coverage curve CSV to "
                f"{saved_paths['risk_coverage_curve_csv']}"
            )

        if "selective_summary_csv" in saved_paths:
            print(
                "Saved selective summary CSV to "
                f"{saved_paths['selective_summary_csv']}"
            )

        if "risk_coverage_curve_calibrated_csv" in saved_paths:
            print(
                "Saved calibrated risk-coverage curve CSV to "
                f"{saved_paths['risk_coverage_curve_calibrated_csv']}"
            )

        if "selective_summary_calibrated_csv" in saved_paths:
            print(
                "Saved calibrated selective summary CSV to "
                f"{saved_paths['selective_summary_calibrated_csv']}"
            )

        if "metrics_calibrated" in report:
            return float(report["metrics_calibrated"]["accuracy"])

        return float(report["metrics"]["accuracy"])