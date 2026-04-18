from __future__ import annotations


def select_eval_logits(method_name: str, outputs, eval_ctx):
    logits = outputs.logits
    logits_fusion = outputs.aux_logits.get("fusion")
    protocol = eval_ctx.protocol
    dataset = eval_ctx.dataset_name
    sub_cls = eval_ctx.subsample_classes or "all"

    if logits_fusion is None:
        return logits

    if method_name in {"MMRL", "MMRLMix", "MMRLpp", "MMRLPP", "BayesMMRL"}:
        if protocol == "B2N":
            if sub_cls == "base":
                return logits_fusion
            return logits

        if protocol == "FS":
            return logits_fusion

        if protocol == "CD":
            if dataset == "ImageNet":
                return logits_fusion
            return logits

    return logits