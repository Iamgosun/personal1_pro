"""Single-source cache validation for E2/E4 fixed-input MC diagnostics.

Fail closed for legacy results: old diagnostics remain on disk but are never
passed off as new E4 measurements. This module never constructs a CLIP model.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from .core import sha256

REFERENCE_SCHEMA = 3  # R13: E2 separate Main/Rep pure Jensen gaps + E4 MC-mean margin


def initial_snapshot_hash(run_dir):
    path = Path(run_dir) / 'trajectory' / 'refactor_model' / 'model.pth.tar-0'
    return sha256(path) if path.is_file() else None


def is_current(path, *, checkpoint_sha256, initial_sha256, samples, repeats,
               batch_size, tolerance, split, require_vectors=False,
               require_e4=False, limit_batches=None):
    """Pure metadata/filesystem test; true means reusable without re-evaluation."""
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        if obj.get('partial') or obj.get('split') != split:
            return False
        if obj.get('reference_schema') != REFERENCE_SCHEMA:
            return False
        if obj.get('checkpoint', {}).get('checkpoint_sha256') != checkpoint_sha256:
            return False
        if obj.get('initial_checkpoint_sha256') != initial_sha256:
            return False
        if obj.get('s_ref') != int(samples) or obj.get('mc_repeats') != int(repeats):
            return False
        if obj.get('diagnostic_batch_size') != int(batch_size):
            return False
        if obj.get('diagnostic_limit_batches') != limit_batches:
            return False
        if not math.isclose(float(obj.get('mc_reference_tolerance',-1)),float(tolerance),rel_tol=0,abs_tol=1e-12):
            return False
        if not (path.parent/'per_image.csv').is_file():
            return False
        if int(repeats)>1 and not (path.parent/'independent_mc_repeats.json').is_file():
            return False
        if require_vectors and not (path.parent/'moment_vectors.json').is_file():
            return False
        if require_e4:
            if obj.get('rep_margin_type') != 'mc_mean_logit':
                return False
            if obj.get('rep_logp_margin_type') != 'posterior_predictive_log_probability':
                return False
        return True
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, OverflowError):
        return False


def formal_is_current(path, checkpoint_sha256):
    """Formal FS result belongs to exact checkpoint (not merely same seed)."""
    try:
        obj=json.loads(Path(path).read_text(encoding='utf-8'))
        return not obj.get('partial',False) and obj['checkpoint']['checkpoint_sha256']==checkpoint_sha256
    except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError):
        return False
