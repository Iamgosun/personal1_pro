from __future__ import annotations


def select_eval_output(task: str, dataset: str, sub_cls: str, logits, logits_fusion):
    task = str(task).upper()
    if task == "B2N":
        return logits_fusion if sub_cls == "base" else logits
    if task == "FS":
        return logits_fusion
    if task == "CD":
        return logits_fusion if dataset == "ImageNet" else logits
    raise ValueError("The TASK must be either B2N, CD, or FS.")
