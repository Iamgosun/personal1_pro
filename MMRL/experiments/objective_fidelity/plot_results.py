"""Visual evidence for the three-method objective fidelity study.

Reads actual fidelity.json, test reports and training logs. Never invents data.
Charts automatically omitted when their required measurements are missing.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from summarize import ROOT, METHODS, test_metrics

LABELS = {"original_mc": "Original-MC", "branch_mc": "Branch-MC", "davi": "DAVI"}


def make_plots(output_root, profile):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"[WARN] Charts require matplotlib + numpy: {e}; no plots created")
        return []

    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    files = sorted(root.glob("*/*/*/shot_*/seed*/fidelity.json"))
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    rows = [r for r in rows if r.get("mode") in METHODS and not r.get("diagnostic_partial") and r.get("diagnostic_schema") == 5]
    if profile != "all":
        rows = [r for r in rows if r.get("profile") == profile]
    for r in rows:
        folder = root / r["profile"] / r["mode"] / r["dataset"] / f"shot_{r['shot']}" / f"seed{r['seed']}"
        r.update(test_metrics(folder))
        r["_folder"] = folder
    if not rows:
        print("[WARN] No completed diagnostics available for plotting")
        return []

    out = root / "plots" / profile
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 165, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.18,
                         "axes.spines.top": False, "axes.spines.right": False})
    # Plot aggregate over seeds. Each method appears only when actual records exist.
    groups = sorted({(r["profile"], r["dataset"], r["shot"]) for r in rows})
    labels = [f"{p}/{d}\n{shot}-shot" if profile == "all" else f"{d}\n{shot}-shot"
              for p, d, shot in groups]
    colors = {"original_mc": "#cc6677", "branch_mc": "#4477aa", "davi": "#228833"}
    offsets = {"original_mc": -0.26, "branch_mc": 0.0, "davi": 0.26}
    saved = []

    def aggregate(metric, group, method):
        vals = [float(r[metric]) for r in rows if (r["profile"], r["dataset"], r["shot"]) == group
                and r["mode"] == method and metric in r and r[metric] not in (None, "")
                and math.isfinite(float(r[metric]))]
        if not vals:
            return None
        a = np.asarray(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0, len(a)

    def grouped_bars(ax, metric, title, ylabel, log=False, nonnegative=False):
        for m in METHODS:
            xs, ys, err = [], [], []
            for i, group in enumerate(groups):
                x = aggregate(metric, group, m)
                if x is not None:
                    xs.append(i + offsets[m]); ys.append(x[0]); err.append(x[1])
            if xs:
                ax.bar(xs, ys, width=0.24, label=LABELS[m], color=colors[m],
                       yerr=(err if any(v > 0 for v in err) else None), capsize=2)
        ax.set_xticks(range(len(groups)), labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        if nonnegative:
            ax.set_ylim(bottom=0)
        if len(groups) > 5:
            ax.tick_params(axis="x", labelrotation=35)

    def save(fig, filename):
        fig.tight_layout(pad=2.0)
        path = out / filename
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(path)
        print(f"[PLOT] {path}")

    # Chart 1: all reference objectives, labeled by split.
    fig, ax = plt.subplots(2, 3, figsize=(max(14, len(groups)*3.2), 8.0))
    for i, (split, split_label) in enumerate((("train_fixed", "Fixed train"), ("test", "Held-out test"))):
        for j, (metric, name) in enumerate((("original_nll", "Original mixture"),
                                            ("branch_nll", "Branch expectation"),
                                            ("det_nll", "Deterministic"))):
            grouped_bars(ax[i, j], f"{split}_{metric}", f"{split_label}: {name} NLL", "NLL", log=True)
    ax[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Three objectives evaluated on fixed inputs | same checkpoint per row", fontsize=13)
    save(fig, "fig01_reference_objectives.png")

    # Chart 2: split-specific direct Main/Rep supervision metrics.
    for split, split_label in (("train_fixed", "Fixed train"), ("test", "Held-out test")):
        if not any(f"{split}_main_nll" in r and f"{split}_rep_nll" in r for r in rows):
            continue
        fig, ax = plt.subplots(2, 2, figsize=(max(11, len(groups)*2.5), 8.0))
        grouped_bars(ax[0, 0], f"{split}_main_nll", "Main branch NLL", "NLL", log=True)
        grouped_bars(ax[0, 1], f"{split}_rep_nll", "Representation branch NLL", "NLL", log=True)
        grouped_bars(ax[1, 0], f"{split}_main_accuracy", "Main branch accuracy", "%")
        grouped_bars(ax[1, 1], f"{split}_rep_accuracy", "Representation branch accuracy", "%")
        ax[1, 0].set_ylim(0, 102)
        ax[1, 1].set_ylim(0, 102)
        ax[0, 0].legend(loc="best", fontsize=9)
        fig.suptitle(f"Branch supervision | {split_label} | same MC reference protocol", fontsize=13)
        save(fig, f"fig02_branch_supervision_{split}.png")

    # Chart 3: separate split rows, separate axes per mathematical gap.
    fig, ax = plt.subplots(2, 3, figsize=(max(14, len(groups)*3.2), 8.0))
    for i, (split, name) in enumerate((("train_fixed", "Fixed train"), ("test", "Held-out test"))):
        grouped_bars(ax[i, 0], f"{split}_mix_gap", f"{name}: Branch - Original", "Jensen gap")
        grouped_bars(ax[i, 1], f"{split}_det_gap", f"{name}: Deterministic - Branch", "Moment gap")
        grouped_bars(ax[i, 2], f"{split}_total_gap", f"{name}: Deterministic - Original", "Total gap")
    ax[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Within-checkpoint gaps; never subtract across checkpoints", fontsize=12)
    save(fig, "fig03_gap_decomposition.png")

    # Chart 4: paired train and test comparisons. These are diagnostic
    # predictions; formal test still uses static_logit in the original trainer.
    fig, ax = plt.subplots(2, 2, figsize=(max(11, len(groups)*2.5), 8.0))
    for i, (split, name) in enumerate((("train_fixed", "Fixed train"), ("test", "Held-out test"))):
        grouped_bars(ax[i, 0], f"{split}_static_prob_accuracy", f"{name}: static_prob", "%")
        grouped_bars(ax[i, 1], f"{split}_static_logit_accuracy", f"{name}: static_logit", "%")
        for a in ax[i]:
            a.set_ylim(0, 102)
    ax[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle("Same frozen checkpoint | fusion diagnostic only | training unchanged", fontsize=12)
    save(fig, "fig04_fusion_diagnostic.png")

    # Chart 5: held-out test as recorded by native trainer, no reinference.
    if any("test_accuracy" in r for r in rows):
        fig, ax = plt.subplots(1, 2, figsize=(max(11, len(groups)*2.5), 4.3))
        grouped_bars(ax[0], "test_accuracy", "Held-out test accuracy", "%")
        grouped_bars(ax[1], "test_nll", "Held-out test NLL", "NLL")
        ax[0].set_ylim(0, 102)
        ax[0].legend(loc="best", fontsize=9)
        fig.suptitle("Native trainer's static_logit test results", fontsize=12)
        save(fig, "fig05_test_accuracy_nll.png")

    # Chart 6: posterior summary across learned q's.
    if any("r_trace" in r for r in rows):
        fig, ax = plt.subplots(1, 2, figsize=(max(11, len(groups)*2.5), 4.3))
        grouped_bars(ax[0], "r_trace", "Visual posterior covariance trace", "trace")
        grouped_bars(ax[1], "t_trace", "Text posterior covariance trace", "trace")
        ax[0].legend(loc="best", fontsize=9)
        fig.suptitle("Posterior scale (not a measure of functional equivalence)", fontsize=12)
        save(fig, "fig06_posterior_traces.png")

    # Chart 7: only after summarize.py added cross-method diagnostic fields.
    csv_path = root / f"objective_fidelity_three_method_{profile}.csv"
    if csv_path.exists():
        import csv
        with csv_path.open(encoding="utf-8", newline="") as fp:
            crows = list(csv.DictReader(fp))
        metrics = ["r_gaussian_w2_rms_to_branch_mc", "t_gaussian_w2_rms_to_branch_mc",
                   "visual_prompt_cosine_to_branch_mc", "text_prompt_cosine_to_branch_mc",
                   "text_feature_cosine_to_branch_mc"]
        if any(r.get(metrics[0]) for r in crows):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
            for m in ("original_mc", "davi"):
                selected = [r for r in crows if r.get("mode") == m]
                for j, metric in enumerate(metrics[:2]):
                    vals = [float(r[metric]) for r in selected if r.get(metric)]
                    if vals:
                        ax[0].bar(j + (0.0 if m == "original_mc" else 0.37),
                                  float(np.mean(vals)), 0.35, color=colors[m], label=LABELS[m] if j == 0 else None)
                for j, metric in enumerate(metrics[2:]):
                    vals = [float(r[metric]) for r in selected if r.get(metric)]
                    if vals:
                        ax[1].bar(j + (0.0 if m == "original_mc" else 0.37),
                                  float(np.mean(vals)), 0.35, color=colors[m])
            ax[0].set_xticks([0.18, 1.18], ["Visual W2 RMS", "Text W2 RMS"])
            ax[0].set_title("Posterior distance vs Branch-MC")
            ax[0].legend(loc="best")
            ax[1].set_xticks([0.18, 1.18, 2.18], ["Visual prompts", "Text prompts", "Text features"])
            ax[1].set_title("Cosine similarity vs Branch-MC")
            ax[1].set_ylim(-1.05, 1.05)
            fig.suptitle("Parameter/feature agreement; pooled seeds for exploratory view")
            save(fig, "fig07_posterior_prompt_distance.png")

    # Chart 8: training learning dynamics, from recorded per-epoch logs.
    def train_curves(folder):
        path = folder / "log.txt"
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8", errors="replace")
        reg = re.compile(r"epoch \[(\d+)/(\d+)\] batch \[(\d+)/(\d+)\].*?\bloss\s+[-+\d.eE]+\s+\(([-+\d.eE]+)\)\s+acc\s+[-+\d.eE]+\s+\(([-+\d.eE]+)\)")
        per_epoch = {}
        for m in reg.finditer(text):
            ep, total_ep, batch, batches = map(int, m.group(1, 2, 3, 4))
            # log values are running averages *within* each epoch; last available line wins
            previous = per_epoch.get(ep)
            if previous is None or batch >= previous[0]:
                per_epoch[ep] = (batch, float(m.group(5)), float(m.group(6)))
        return {e: (v[1], v[2]) for e, v in sorted(per_epoch.items())}

    for p, d, shot in groups:
        seeds = sorted({r["seed"] for r in rows if (r["profile"], r["dataset"], r["shot"]) == (p, d, shot)})
        for seed in seeds:
            available = []
            for m in METHODS:
                matches = [r for r in rows if r["profile"] == p and r["dataset"] == d and r["shot"] == shot
                           and r["seed"] == seed and r["mode"] == m]
                if matches:
                    curve = train_curves(matches[0]["_folder"])
                    if curve:
                        available.append((m, curve))
            if not available:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
            for m, curve in available:
                ep = sorted(curve)
                ax[0].plot(ep, [curve[e][0] for e in ep], label=LABELS[m], color=colors[m], linewidth=2)
                ax[1].plot(ep, [curve[e][1] for e in ep], label=LABELS[m], color=colors[m], linewidth=2)
            ax[0].set_title("Logged training loss (different objectives)")
            ax[0].set_ylabel("loss")
            ax[1].set_title("Logged training accuracy (static_logit)")
            ax[1].set_ylabel("%")
            ax[1].set_ylim(0, 102)
            for a in ax:
                a.set_xlabel("epoch")
            ax[0].legend(loc="best")
            fig.suptitle(f"{d} | {shot}-shot | seed {seed} | {p}", fontsize=13)
            save(fig, f"fig08_training_{p}_{d}_shot{shot}_seed{seed}.png")
    print(f"[PLOT] Wrote {len(saved)} figures to {out}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="output_objective_fidelity_v3")
    ap.add_argument("--profile", choices=("native",), default="native")
    o = ap.parse_args()
    make_plots(o.output_root, o.profile)


if __name__ == "__main__":
    main()
