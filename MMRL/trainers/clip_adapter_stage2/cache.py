from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class CacheSpec:
    cache_id: str
    tensor_path: str
    manifest_path: str
    metadata: Dict


class FeatureCacheManager:
    """Feature cache with manifest metadata.

    This is a pragmatic stage-2 upgrade over `_get_feature_cache_path()`.
    It still stores one `.pt` tensor payload per split, but now also writes a
    small manifest so cache reuse is explicit and inspectable.
    """

    def __init__(self, cfg, initialization: str, root_dir: Optional[str] = None):
        self.cfg = cfg
        self.initialization = initialization
        backbone = cfg.MODEL.BACKBONE.NAME
        dataset = cfg.DATASET.NAME
        if root_dir is None:
            root_dir = os.environ.get(
                "MMRL_CLIP_FEATURE_CACHE",
                f"/root/autodl-tmp/DASSL_FLAIR_CLIP/Output/clip_features/{backbone}/{dataset}",
            )
        self.root_dir = root_dir
        self.tensor_dir = os.path.join(root_dir, "tensors")
        self.manifest_dir = os.path.join(root_dir, "manifest")
        os.makedirs(self.tensor_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

    def build_metadata(self, partition: str, reps: int, train_is_augf: bool) -> Dict:
        return {
            "schema_version": 2,
            "dataset": self.cfg.DATASET.NAME,
            "partition": partition,
            "seed": int(self.cfg.SEED),
            "shots": int(self.cfg.DATASET.NUM_SHOTS),
            "backbone": self.cfg.MODEL.BACKBONE.NAME,
            "initialization": self.initialization,
            "reps": int(reps),
            "train_is_augf": bool(train_is_augf),
            "subsample_classes": getattr(self.cfg.DATASET, "SUBSAMPLE_CLASSES", "all"),
        }

    def _hash_metadata(self, metadata: Dict) -> str:
        payload = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def build_spec(self, partition: str, reps: int, train_is_augf: bool) -> CacheSpec:
        metadata = self.build_metadata(partition, reps, train_is_augf)
        cache_id = self._hash_metadata(metadata)
        tensor_path = os.path.join(self.tensor_dir, f"{cache_id}.pt")
        manifest_path = os.path.join(self.manifest_dir, f"{cache_id}.json")
        return CacheSpec(cache_id=cache_id, tensor_path=tensor_path, manifest_path=manifest_path, metadata=metadata)

    def load(self, spec: CacheSpec):
        if not (os.path.exists(spec.tensor_path) and os.path.exists(spec.manifest_path)):
            return None
        with open(spec.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("metadata") != spec.metadata:
            return None
        return torch.load(spec.tensor_path, map_location="cpu")

    def save(self, spec: CacheSpec, payload: Dict):
        torch.save(payload, spec.tensor_path)
        with open(spec.manifest_path, "w", encoding="utf-8") as f:
            json.dump({"cache_id": spec.cache_id, "metadata": spec.metadata}, f, ensure_ascii=False, indent=2)
