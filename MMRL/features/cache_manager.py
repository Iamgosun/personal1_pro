from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class CacheSpec:
    cache_id: str
    tensor_path: str
    manifest_path: str
    metadata: Dict


class FeatureCacheManager:
    def __init__(self, cfg, root_dir: Optional[str] = None):
        self.cfg = cfg

        backbone = cfg.MODEL.BACKBONE.NAME.replace("/", "-")
        dataset = cfg.DATASET.NAME

        if root_dir is None:
            root_dir = str(Path("cache") / "clip_features" / backbone / dataset)

        self.root_dir = Path(root_dir)
        self.tensor_dir = self.root_dir / "tensors"
        self.manifest_dir = self.root_dir / "manifest"

        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _jsonable(value: Any):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [FeatureCacheManager._jsonable(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): FeatureCacheManager._jsonable(v)
                for k, v in value.items()
            }
        return str(value)

    @staticmethod
    def _cfg_get(cfg, name: str, default=None):
        if cfg is None:
            return default
        return getattr(cfg, name, default)



    def build_metadata(
        self,
        split: str,
        reps: int,
        train_aug: bool,
        mode: str,
        aggregation: str,
    ) -> Dict:
        clip_cfg = getattr(self.cfg, "CLIP_ADAPTERS", None)
        input_cfg = getattr(self.cfg, "INPUT", None)
        backbone_cfg = self.cfg.MODEL.BACKBONE
        metadata = {
            # Bump because CACHE_REPS semantics changed:
            # schema v4 averaged repeated train augmentations;
            # schema v5 can preserve them as a feature pool.
            # schema v6 supports adapter-stable feature-only cache keys.
            "schema_version": 6,

            "dataset": self.cfg.DATASET.NAME,
            "split": str(split),
            "seed": int(self.cfg.SEED),
            "shots": int(self.cfg.DATASET.NUM_SHOTS),
            "subsample_classes": self._jsonable(
                getattr(self.cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
            ),

            "backbone": self.cfg.MODEL.BACKBONE.NAME,
            "backbone_pretrained": self._jsonable(
                getattr(backbone_cfg, "PRETRAINED", True)
            ),
            "model_init_weights": self._jsonable(
                getattr(self.cfg.MODEL, "INIT_WEIGHTS", "")
            ),

            "method_name": self.cfg.METHOD.NAME,
            "method_family": self.cfg.METHOD.FAMILY,
            "protocol": self.cfg.PROTOCOL.NAME,
            "phase": self._jsonable(getattr(self.cfg.PROTOCOL, "PHASE", None)),

            # Image preprocessing affects cached features.
            "input_size": self._jsonable(self._cfg_get(input_cfg, "SIZE", None)),
            "input_interpolation": self._jsonable(
                self._cfg_get(input_cfg, "INTERPOLATION", None)
            ),
            "input_transforms": self._jsonable(
                self._cfg_get(input_cfg, "TRANSFORMS", None)
            ),
            "input_rrcrop_scale": self._jsonable(
                self._cfg_get(input_cfg, "RRCROP_SCALE", None)
            ),
            "input_pixel_mean": self._jsonable(
                self._cfg_get(input_cfg, "PIXEL_MEAN", None)
            ),
            "input_pixel_std": self._jsonable(
                self._cfg_get(input_cfg, "PIXEL_STD", None)
            ),

            "reps": int(reps),
            "train_aug": bool(train_aug),
            "aggregation": str(aggregation),
            "mode": str(mode),
        }

        feature_only_key = bool(
            self._cfg_get(clip_cfg, "CACHE_FEATURE_ONLY_KEY", False)
        )

        # For BayesAdapter / DREAM-BayesAdapter, cache training only needs CLIP
        # image features and labels. Do not include adapter-specific fields in
        # the cache key; otherwise switching BAYES_ADAPTER <-> DREAM_BAYES_ADAPTER
        # rebuilds the exact same CLIP feature pool.
        #
        # For CLAP constraint, TipA, CrossModal, etc., keep old behavior because
        # cached logits / feature-pool shape can depend on adapter settings.
        adapter_init = str(self._cfg_get(clip_cfg, "INIT", "unknown")).upper()
        adapter_constraint = str(self._cfg_get(clip_cfg, "CONSTRAINT", "none")).lower()
        is_bayes_like = (
            "BAYES_ADAPTER" in adapter_init
            or "DREAM_BAYES_ADAPTER" in adapter_init
            or "DREAMBAYES" in adapter_init
        )

        use_stable_feature_key = (
            feature_only_key
            and mode == "features_only"
            and adapter_constraint == "none"
            and is_bayes_like
        )

        if not use_stable_feature_key:
            # Cached payload includes logits in addition to image features.
            # Adapter/logit-affecting settings must therefore be part of the key.
            metadata.update(
                {
                    "method_tag": self._jsonable(getattr(self.cfg.METHOD, "TAG", "")),
                    "adapter_init": self._jsonable(
                        self._cfg_get(clip_cfg, "INIT", "unknown")
                    ),
                    "adapter_type": self._jsonable(
                        self._cfg_get(clip_cfg, "TYPE", "unknown")
                    ),
                    "adapter_constraint": self._jsonable(
                        self._cfg_get(clip_cfg, "CONSTRAINT", "unknown")
                    ),
                    "enhanced_base": self._jsonable(
                        self._cfg_get(clip_cfg, "ENHANCED_BASE", "none")
                    ),
                    "precision": self._jsonable(
                        self._cfg_get(clip_cfg, "PREC", "unknown")
                    ),
                    "n_samples": self._jsonable(
                        self._cfg_get(clip_cfg, "N_SAMPLES", None)
                    ),
                    "n_test_samples": self._jsonable(
                        self._cfg_get(clip_cfg, "N_TEST_SAMPLES", None)
                    ),
                    "clap_tipa_trainable_cache": self._jsonable(
                        self._cfg_get(clip_cfg, "CLAP_TIPA_TRAINABLE_CACHE", None)
                    ),
                    "clap_tipa_raw_affinity": self._jsonable(
                        self._cfg_get(clip_cfg, "CLAP_TIPA_RAW_AFFINITY", None)
                    ),
                    "cross_modal_resample_text": self._jsonable(
                        self._cfg_get(clip_cfg, "CROSS_MODAL_RESAMPLE_TEXT", None)
                    ),
                }
            )

        return metadata





    def build_cache_id(self, metadata: Dict) -> str:
        payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]



    def build_spec(
        self,
        split: str,
        reps: int,
        train_aug: bool,
        mode: str = "features_only",
        aggregation: str = "mean",
    ) -> CacheSpec:
        metadata = self.build_metadata(
            split=split,
            reps=reps,
            train_aug=train_aug,
            mode=mode,
            aggregation=aggregation,
        )
        cache_id = self.build_cache_id(metadata)

        return CacheSpec(
            cache_id=cache_id,
            tensor_path=str(self.tensor_dir / f"{cache_id}.pt"),
            manifest_path=str(self.manifest_dir / f"{cache_id}.json"),
            metadata=metadata,
        )


    def validate_cache(self, metadata: Dict, manifest: Dict) -> bool:
        return manifest.get("metadata") == metadata

    def load(self, spec: CacheSpec):
        tensor_path = Path(spec.tensor_path)
        manifest_path = Path(spec.manifest_path)

        if not tensor_path.exists() or not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        if not self.validate_cache(spec.metadata, manifest):
            return None

        try:
            return torch.load(tensor_path, map_location="cpu")
        except Exception:
            return None

    def save(self, spec: CacheSpec, payload: Dict):
        tensor_path = Path(spec.tensor_path)
        manifest_path = Path(spec.manifest_path)

        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_tensor_path = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
        tmp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

        torch.save(payload, tmp_tensor_path)
        tmp_manifest_path.write_text(
            json.dumps(
                {
                    "cache_id": spec.cache_id,
                    "metadata": spec.metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tmp_tensor_path.replace(tensor_path)
        tmp_manifest_path.replace(manifest_path)