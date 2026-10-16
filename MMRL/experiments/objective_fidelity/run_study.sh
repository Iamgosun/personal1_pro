#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
PROFILE="${PROFILE:-native}"
DATASETS="${DATASETS:-caltech101 }"
SEEDS="${SEEDS:-1 }"
MODES="${MODES:-original_mc branch_mc davi}"
SHOTS="${SHOTS:-16}"
SAMPLES="${SAMPLES:-10}"          # set 4096 for final gap estimates
CHUNK="${CHUNK:-16}"
FEATURES="${FEATURES:-128}"
REF_BATCH="${REF_BATCH:-32}"       # fixed-train / held-out-test evaluation batch size
OUT="${OUT:-output_objective_fidelity_v3}"
if [[ "$OUT" = /* ]]; then BASE="$OUT"; else BASE="$ROOT/$OUT"; fi
extra=()
if [[ -n "${EPOCHS:-}" ]]; then extra+=(--epochs "$EPOCHS"); fi
if [[ -n "${MC_TRAIN:-}" ]]; then extra+=(--mc-train "$MC_TRAIN"); fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    for mode in $MODES; do
      case "$mode" in original_mc|branch_mc|davi) ;; *) echo "Unknown mode: $mode" >&2; exit 2;; esac
      folder="$BASE/$PROFILE/$mode/$dataset/shot_$SHOTS/seed$seed"
      if [[ -f "$folder/elapsed_wall_seconds.txt" ]]; then
        # Do not silently reuse an old checkpoint under a new training budget.
        python - "$folder/experiment_manifest.json" "${EPOCHS:-}" "${MC_TRAIN:-}" <<'PY'
import json, sys
from pathlib import Path
f = Path(sys.argv[1])
if not f.exists():
    raise SystemExit(f"Missing manifest for completed checkpoint: {f}")
m = json.loads(f.read_text(encoding="utf-8"))
for name, opt in (("max_epochs", sys.argv[2]), ("n_mc_train", sys.argv[3])):
    if opt and int(opt) != int(m[name]):
        raise SystemExit(f"Existing checkpoint uses {name}={m[name]}, requested {opt}. Use a different OUT for a new train run.")
PY
        echo "[skip completed training] $folder"
      else
        python "$HERE/train.py" --mode "$mode" --profile "$PROFILE" --dataset "$dataset" \
          --shot "$SHOTS" --seed "$seed" --gpu "$GPU" --output-root "$OUT" "${extra[@]}"
      fi
      need_eval=1
      if [[ -f "$folder/fidelity.json" && -f "$folder/posterior_diagnostic.pt" && "${REEVAL:-0}" != "1" ]]; then
        if python - "$folder/fidelity.json" "$SAMPLES" "$folder/posterior_diagnostic.pt" "$FEATURES" "$REF_BATCH" <<'PY'
import json, sys, torch
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert int(m.get("diagnostic_schema", -1)) == 5
assert m.get("reference_input_protocol") == "fixed_train_and_heldout_test"
assert int(m.get("reference_batch_size", -1)) == int(sys.argv[5])
assert m.get("diagnostic_partial") is False
pt = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
assert {"r_mean", "r_sigma", "t_mean", "t_sigma", "visual_prompts", "text_prompts", "text_features"}.issubset(pt)
for split in ("train_fixed", "test"):
    needed = {f"{split}_{k}" for k in (
        "original_nll", "branch_nll", "det_nll", "main_nll", "rep_nll",
        "mix_gap", "det_gap", "total_gap", "main_accuracy", "rep_accuracy",
        "static_prob_accuracy", "static_logit_accuracy", "num_examples", "input_sha256",
    )}
    assert needed.issubset(m), (needed-set(m))
    assert int(m[f"{split}_mc_reference_samples"]) == int(sys.argv[2])
if int(sys.argv[4]) > 0:
    assert {"image_hashes", "image_labels", "image_main_features", "image_rep_features"}.issubset(pt)
    assert int(m.get("feature_diagnostic_images", 0)) == int(sys.argv[4])
PY
        then need_eval=0; fi
      fi
      if [[ "$need_eval" -eq 0 ]]; then
        echo "[skip completed diagnostic] $folder"
      else
        python "$HERE/evaluate.py" --mode "$mode" --profile "$PROFILE" --dataset "$dataset" \
          --shot "$SHOTS" --seed "$seed" --gpu "$GPU" --output-root "$OUT" \
          --samples "$SAMPLES" --chunk "$CHUNK" --features "$FEATURES" --reference-batch-size "$REF_BATCH"
      fi
    done
  done
done
python "$HERE/summarize.py" --profile "$PROFILE" --output-root "$OUT"
python "$HERE/plot_results.py" --profile "$PROFILE" --output-root "$OUT"
