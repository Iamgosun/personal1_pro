from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch


try:
    from ..evaluators.resultReporter import ResultReporter
except Exception:  # pragma: no cover - keep import soft for compatibility patching
    ResultReporter = None


def save_confidence_coverage_stats(output_dir: str, labels: torch.Tensor, logits: torch.Tensor, num_bins: int = 10):
    probs = torch.softmax(logits, dim=-1)
    confidence, preds = probs.max(dim=1)
    correctness = (preds == labels).float()

    bins = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=confidence.device)
    rows = []
    ece = torch.tensor(0.0, device=confidence.device)
    aece = torch.tensor(0.0, device=confidence.device)

    for i in range(num_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == num_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        num_samples = int(mask.sum().item())
        if num_samples == 0:
            rows.append(
                {
                    "bin": i,
                    "left": float(left.item()),
                    "right": float(right.item()),
                    "num_samples": 0,
                    "coverage": 0.0,
                    "accuracy": None,
                    "mean_confidence": None,
                    "abs_gap": None,
                }
            )
            continue
        acc = correctness[mask].mean()
        conf = confidence[mask].mean()
        gap = torch.abs(acc - conf)
        coverage = num_samples / max(1, labels.numel())
        ece += gap * coverage
        aece += gap
        rows.append(
            {
                "bin": i,
                "left": float(left.item()),
                "right": float(right.item()),
                "num_samples": num_samples,
                "coverage": coverage,
                "accuracy": float(acc.item()),
                "mean_confidence": float(conf.item()),
                "abs_gap": float(gap.item()),
            }
        )

    aece = aece / num_bins
    rows.append(
        {
            "bin": "summary",
            "left": None,
            "right": None,
            "num_samples": int(labels.numel()),
            "coverage": 1.0,
            "accuracy": None,
            "mean_confidence": f"ECE={ece.item():.4f}, AECE={aece.item():.4f}",
            "abs_gap": None,
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "confidence_coverage_stats.csv"), index=False)


def save_final_reports(model, cfg, device, features_test, labels_test, features_ood: Optional[torch.Tensor] = None):
    if ResultReporter is None:
        print("[Stage2] ResultReporter not available; skip final report generation.")
        return None
    reporter = ResultReporter(
        model=model,
        cfg=cfg,
        device=device,
        features_test=features_test,
        labels_test=labels_test,
        features_ood=features_ood,
    )
    return reporter.save()
