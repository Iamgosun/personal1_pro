"""Compact E6 publication-style diagnostic figures (one independent plot per file)."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import ROOT


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def figure(out, title, ylabel, series, ylim=None):
    plt.figure(figsize=(6.6, 3.7), dpi=160)
    for label, points in series.items():
        points = sorted(points)
        if points:
            plt.plot([x for x,_ in points], [y for _,y in points], marker="o", linewidth=1.5, markersize=3.5, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title, fontsize=11)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="output_supplementary_v7")
    a=p.parse_args()
    root=Path(a.output)
    root=(root if root.is_absolute() else ROOT/root).resolve()
    data=root/"summary"/"e6"
    dest=root/"figures"/"e6"
    features=rows(data/"feature_summary.csv")
    native=rows(data/"gradient_summary.csv")
    objective=rows(data/"objective_summary.csv")
    datasets=sorted({r["dataset"] for records in (features,native,objective) for r in records})
    for ds in datasets:
        ff=[r for r in features if r["dataset"]==ds]
        if ff:
            lines={}
            for label,field in (("R hidden","r_hidden_cos_mean"),("R projected","native_rep_cos_mean"),
                                ("Fixed head, different hidden","hidden_change_cos_mean"),
                                ("Fixed hidden, different head","head_change_cos_mean")):
                lines[label]=[(int(r["epoch"]),number(r.get(field))) for r in ff]
            figure(dest/f"{ds}_representation.png",f"{ds} | R representation attribution","Cosine similarity",lines,(-.05,1.05))
            figure(dest/f"{ds}_posterior_mean_similarity.png", f"{ds} | visual posterior mean similarity",
                   "Cosine(M_r AVI-B, M_r DAVI)", {"Visual posterior mean": [(int(r["epoch"]),number(r.get("r_mean_cos_mean"))) for r in ff]}, (-1.05,1.05))
        gg=[r for r in native if r["dataset"]==ds]
        if gg:
            fig={}
            for method in ("branch","davi"):
                rr=[r for r in gg if r["method"]==method]
                fig["AVI-B" if method=="branch" else "DAVI"]=[(int(r["epoch"]),number(r.get("dynamics_r_mean_fro_mean"))) for r in rr]
            figure(dest/f"{ds}_visual_mean.png",f"{ds} | visual posterior mean","||M_r||_F",fig)
            for group in ("r_mean","r_rho","t_rho"):
                lines={}
                for method in ("branch","davi"):
                    rr=[r for r in gg if r["method"]==method]
                    lines["AVI-B" if method=="branch" else "DAVI"]=[(int(r["epoch"]),
                           number(r.get(f"groups_{group}_total_grad_rms_mean"))) for r in rr]
                figure(dest/f"{ds}_{group}_gradient.png",f"{ds} | {group} native total gradient","Gradient RMS",lines)
        oo=[r for r in objective if r["dataset"]==ds and r.get("status")=="ok"]
        if oo:
            lines={}
            for method in ("branch","davi"):
                rr=[r for r in oo if r["method"]==method]
                lines["AVI-B checkpoint" if method=="branch" else "DAVI checkpoint"]=[
                    (int(r["epoch"]),number(r.get("groups_r_mean_mc_davi_cos_mean"))) for r in rr]
            figure(dest/f"{ds}_mean_objective_cos.png",f"{ds} | same-state objective gradients: M_r",
                   "Cosine(grad MC, grad DAVI)",lines,(-1.05,1.05))
    print("[E6 FIGURES]",dest)

if __name__=="__main__":main()
