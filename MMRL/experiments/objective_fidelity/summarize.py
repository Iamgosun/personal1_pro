"""Aggregate diagnostics without conflating native and controlled KL setups."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="output_objective_fidelity_v3")
    p.add_argument("--profile", choices=["native", "controlled", "all"], default="all")
    args = p.parse_args()
    root = Path(args.output_root)
    files = sorted(root.glob("*/*/*/shot_*/seed*/fidelity.json"))
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if args.profile != "all":
        rows = [r for r in rows if r["profile"] == args.profile]
    if not rows:
        raise RuntimeError(f"No fidelity.json found under {root} for {args.profile}")

    lookup = {(r["profile"], r["dataset"], r["shot"], r["seed"], r["mode"]): r for r in rows}
    for row in rows:
        file_root = root / row["profile"] / row["mode"] / row["dataset"] / f"shot_{row['shot']}" / f"seed{row['seed']}"
        timing = file_root / "elapsed_wall_seconds.txt"
        row["train_and_test_wall_seconds"] = float(timing.read_text()) if timing.exists() else ""
        key = (row["profile"], row["dataset"], row["shot"], row["seed"], "original_mc")
        ref = lookup.get(key)
        if ref is None:
            continue
        row["original_nll_difference_to_original_mc"] = row["original_nll"] - ref["original_nll"]
        row["r_trace_ratio_to_original_mc"] = row["r_trace"] / ref["r_trace"] if ref["r_trace"] > 0 else ""
        row["t_trace_ratio_to_original_mc"] = row["t_trace"] / ref["t_trace"] if ref["t_trace"] > 0 else ""
        # Full objectives can only be cross-compared under common prior/KL settings.
        if row["profile"] == "controlled":
            row["reference_difference_to_original_mc"] = (row["original_reference_loss"]
                                                            - ref["original_reference_loss"])
        import torch
        def posterior_file(r):
            return root / r["profile"] / r["mode"] / r["dataset"] / f"shot_{r['shot']}" / f"seed{r['seed']}" / "posterior_diagnostic.pt"
        a = torch.load(posterior_file(row), map_location="cpu", weights_only=True)["r_mean"].float()
        b = torch.load(posterior_file(ref), map_location="cpu", weights_only=True)["r_mean"].float()
        row["visual_mean_distance_to_original_mc"] = float((a - b).norm() / b.norm().clamp_min(1e-12))

    fields = list(dict.fromkeys(k for r in rows for k in r))
    dest = root / (f"objective_fidelity_{args.profile}.csv" if args.profile != "all" else "objective_fidelity_summary.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {dest.resolve()}")


if __name__ == "__main__":
    main()
