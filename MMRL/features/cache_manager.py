from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
        backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '-')
        dataset = cfg.DATASET.NAME
        if root_dir is None:
            root_dir = str(Path('cache') / 'clip_features' / backbone / dataset)
        self.root_dir = Path(root_dir)
        self.tensor_dir = self.root_dir / 'tensors'
        self.manifest_dir = self.root_dir / 'manifest'
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def build_metadata(self, split: str, reps: int, train_aug: bool, mode: str) -> Dict:
        return {
            'schema_version': 3,
            'dataset': self.cfg.DATASET.NAME,
            'split': split,
            'seed': int(self.cfg.SEED),
            'shots': int(self.cfg.DATASET.NUM_SHOTS),
            'backbone': self.cfg.MODEL.BACKBONE.NAME,
            'method_name': self.cfg.METHOD.NAME,
            'method_family': self.cfg.METHOD.FAMILY,
            'protocol': self.cfg.PROTOCOL.NAME,
            'subsample_classes': getattr(self.cfg.DATASET, 'SUBSAMPLE_CLASSES', 'all'),
            'precision': self.cfg.CLIP_ADAPTERS.PREC if hasattr(self.cfg, 'CLIP_ADAPTERS') else 'unknown',
            'reps': int(reps),
            'train_aug': bool(train_aug),
            'mode': mode,
        }

    def build_cache_id(self, metadata: Dict) -> str:
        payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return hashlib.sha1(payload).hexdigest()[:16]

    def build_spec(self, split: str, reps: int, train_aug: bool, mode: str = 'features_only') -> CacheSpec:
        metadata = self.build_metadata(split, reps, train_aug, mode)
        cache_id = self.build_cache_id(metadata)
        return CacheSpec(
            cache_id=cache_id,
            tensor_path=str(self.tensor_dir / f'{cache_id}.pt'),
            manifest_path=str(self.manifest_dir / f'{cache_id}.json'),
            metadata=metadata,
        )

    def validate_cache(self, metadata: Dict, manifest: Dict) -> bool:
        return manifest.get('metadata') == metadata

    def load(self, spec: CacheSpec):
        tensor_path = Path(spec.tensor_path)
        manifest_path = Path(spec.manifest_path)
        if not tensor_path.exists() or not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not self.validate_cache(spec.metadata, manifest):
            return None
        return torch.load(tensor_path, map_location='cpu')

    def save(self, spec: CacheSpec, payload: Dict):
        torch.save(payload, spec.tensor_path)
        Path(spec.manifest_path).write_text(
            json.dumps({'cache_id': spec.cache_id, 'metadata': spec.metadata}, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
