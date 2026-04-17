from __future__ import annotations


def select_eval_logits(method_name: str, outputs, eval_ctx):
    logits = outputs.logits
    logits_fusion = outputs.aux_logits.get("fusion")
    protocol = eval_ctx.protocol
    dataset = eval_ctx.dataset_name
    sub_cls = eval_ctx.subsample_classes or "all"

    if method_name in {"MMRL", "MMRLpp", "MMRLPP", "BayesMMRL"}:
        if protocol == "B2N":
            if sub_cls == "base" and logits_fusion is not None:
                return logits_fusion
            return logits

        if protocol == "FS" and logits_fusion is not None:
            return logits_fusion

        if protocol == "CD":
            if dataset == "ImageNet" and logits_fusion is not None:
                return logits_fusion
            return logits

    return logits