"""Three trained methods; Original/Branch/Det losses measured on EACH checkpoint.

Original-MC vs Branch-MC tests Jensen decomposition. Branch-MC vs DAVI tests
moment approximation. No changes are made to the repository's model/trainer.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("original_mc", "branch_mc", "davi")
ORDER = {m: i for i, m in enumerate(METHODS)}


def read_posterior(folder):
    path = folder / "posterior_diagnostic.pt"
    if not path.exists():
        return None
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def compare_to_branch(target, branch):
    """Paired Gaussian W2 (diagonal), effective prompts and features."""
    import torch
    import torch.nn.functional as F

    ans = {}
    a, b = target["r_mean"].float(), branch["r_mean"].float()
    if a.shape != b.shape:
        raise ValueError("Visual posterior mean shape mismatch")
    ans["r_mean_relative_distance_to_branch_mc"] = float((a - b).norm() / b.norm().clamp_min(1e-12))
    for prefix in ("r", "t"):
        ma, mb = target[f"{prefix}_mean"].float(), branch[f"{prefix}_mean"].float()
        sa, sb = target[f"{prefix}_sigma"].float(), branch[f"{prefix}_sigma"].float()
        if ma.shape != mb.shape or sa.shape != sb.shape or ma.shape != sa.shape:
            raise ValueError(f"{prefix} Gaussian posterior shapes differ")
        ans[f"{prefix}_gaussian_w2_rms_to_branch_mc"] = float(
            ((ma - mb).square() + (sa - sb).square()).mean().sqrt()
        )
    for prefix in ("visual", "text"):
        pa, pb = target[f"{prefix}_prompts"].float(), branch[f"{prefix}_prompts"].float()
        if pa.shape != pb.shape:
            raise ValueError(f"{prefix} effective prompt shape mismatch")
        ans[f"{prefix}_prompt_cosine_to_branch_mc"] = float(
            F.cosine_similarity(pa.flatten()[None], pb.flatten()[None]).item()
        )
    if "text_features" in target and "text_features" in branch:
        ta, tb = target["text_features"].float(), branch["text_features"].float()
        if ta.shape != tb.shape:
            raise ValueError("Text class feature shape mismatch")
        ans["text_feature_cosine_to_branch_mc"] = float(F.cosine_similarity(ta, tb, dim=-1).mean())
    ha, hb = target.get("image_hashes"), branch.get("image_hashes")
    ans["feature_input_identical"] = (ha is not None and hb is not None and sorted(ha) == sorted(hb))
    if ans["feature_input_identical"]:
        ia = sorted(range(len(ha)), key=lambda i: ha[i])
        ib = sorted(range(len(hb)), key=lambda i: hb[i])
        if not torch.equal(target["image_labels"][ia], branch["image_labels"][ib]):
            raise RuntimeError("Matching image hashes but nonmatching labels")
        for key in ("image_main_features", "image_rep_features"):
            fa, fb = target[key][ia].float(), branch[key][ib].float()
            if fa.shape != fb.shape:
                raise ValueError(f"{key} shape mismatch")
            ans[key.replace("_features", "_cosine_to_branch_mc")] = float(
                F.cosine_similarity(fa, fb, dim=-1).mean()
            )
    elif ha is not None and hb is not None:
        print("[WARN] Validation input tensors differ across checkpoints; no cross-model image cosine")
    return ans


def _walk_numeric(report, key):
    """Find an exact metric key within unknown test_report.json nesting."""
    if isinstance(report, dict):
        if key in report and isinstance(report[key], (int, float)):
            return float(report[key])
        for item in report.values():
            found = _walk_numeric(item, key)
            if found is not None:
                return found
    return None


def test_metrics(folder):
    keys = ("accuracy", "macro_f1", "nll", "ece", "brier")
    f = folder / "test_report.json"
    values = {}
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        values = {k: _walk_numeric(data, k) for k in keys}
    # The original trainer also logs the named, high-precision test metrics.
    if any(values.get(k) is None for k in keys):
        log = folder / "log.txt"
        if log.exists():
            content = log.read_text(encoding="utf-8", errors="replace")
            pos = content.rfind("=> structured result")
            if pos >= 0:
                section = content[pos:].split("Saved metrics", 1)[0]
                for k in keys:
                    if values.get(k) is None:
                        m = re.search(r"^\s*\* " + re.escape(k) + r":\s*([-+\d.eE]+)\s*$", section, re.M)
                        if m:
                            values[k] = float(m.group(1))
    return {"test_" + k: v for k, v in values.items() if v is not None}


def summarize(output_root, profile):
    root = Path(output_root)
    if not root.is_absolute():
        root = ROOT / root
    paths = sorted(root.glob("*/*/*/shot_*/seed*/fidelity.json"))
    rows = []
    for f in paths:
        row = json.loads(f.read_text(encoding="utf-8"))
        if row.get("mode") not in METHODS or row.get("diagnostic_partial") or row.get("diagnostic_schema") != 5:
            continue
        if row.get("profile") != "native":
            continue
        row = dict(row)
        folder = f.parent
        wall = folder / "elapsed_wall_seconds.txt"
        if wall.exists():
            row["train_and_test_wall_seconds"] = float(wall.read_text().strip())
        row.update(test_metrics(folder))
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No V5 fixed-train+test diagnostics under {root} for profile={profile}; run evaluate.py first")
    rows.sort(key=lambda r: (r["profile"], r["dataset"], int(r["shot"]), int(r["seed"]), ORDER[r["mode"]]))
    bykey = {(r["profile"], r["dataset"], r["shot"], r["seed"], r["mode"]): r for r in rows}
    snapshot_cache = {}

    def snapshot(row):
        k = (row["profile"], row["dataset"], row["shot"], row["seed"], row["mode"])
        if k not in snapshot_cache:
            folder = root / row["profile"] / row["mode"] / row["dataset"] / f"shot_{row['shot']}" / f"seed{row['seed']}"
            snapshot_cache[k] = read_posterior(folder)
        return snapshot_cache[k]

    for row in rows:
        base = (row["profile"], row["dataset"], row["shot"], row["seed"])
        branch = bykey.get((*base, "branch_mc"))
        original = bykey.get((*base, "original_mc"))
        if branch is not None:
            for split in ("train_fixed", "test"):
                digest_key = f"{split}_input_sha256"
                if row.get(digest_key) != branch.get(digest_key):
                    raise RuntimeError(
                        f"{split}: nonidentical reference images/labels for {base}; "
                        f"cannot compare {row['mode']} vs Branch-MC. "
                        "Check data order and deterministic transform."
                    )
                key = f"{split}_original_nll"
                row[f"{split}_original_nll_difference_to_branch_mc"] = row[key] - branch[key]
            row["r_trace_ratio_to_branch_mc"] = row["r_trace"] / branch["r_trace"] if branch["r_trace"] else None
            row["t_trace_ratio_to_branch_mc"] = row["t_trace"] / branch["t_trace"] if branch["t_trace"] else None
            if row["mode"] != "branch_mc":
                a, b = snapshot(row), snapshot(branch)
                if a is not None and b is not None:
                    row.update(compare_to_branch(a, b))
                else:
                    print(f"[WARN] Missing posterior snapshots for {base}: {row['mode']} vs branch_mc")
        if original is not None:
            for split in ("train_fixed", "test"):
                digest_key = f"{split}_input_sha256"
                if row.get(digest_key) != original.get(digest_key):
                    raise RuntimeError(f"{split}: unpaired inputs vs Original-MC for {base}")
                key = f"{split}_original_nll"
                row[f"{split}_original_nll_difference_to_original_mc"] = row[key] - original[key]
    cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    dest = root / f"objective_fidelity_three_method_{profile}.csv"
    with dest.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {len(rows)} rows -> {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="output_objective_fidelity_v3")
    ap.add_argument("--profile", choices=("native",), default="native")
    args = ap.parse_args()
    summarize(args.output_root, args.profile)


if __name__ == "__main__":
    main()
