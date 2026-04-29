from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
from backbones.text_encoders import CLIPTextEncoder
from .base import BaseAdapter


def _norm_name(x: str) -> str:
    x = str(x)
    x = x.replace("_", " ").replace("-", " ").replace("/", " ")
    x = re.sub(r"(?<!^)(?=[A-Z])", " ", x)
    x = re.sub(r"[^a-zA-Z0-9]+", " ", x)
    return " ".join(x.lower().split())


def _name_variants(x: str) -> list[str]:
    """
    Robust matching keys for dataset classnames and prompt-bank class names.

    Needed for cases such as Caltech101:
      dataset.classnames: face, leopard, motorbike, airplane
      prompt bank names:  Faces, Leopards, Motorbikes, Airplanes
    """
    base = _norm_name(x)
    variants = {base}

    compact = base.replace(" ", "")
    variants.add(compact)

    # Simple singular/plural variants.
    if base.endswith("ies") and len(base) > 3:
        variants.add(base[:-3] + "y")
    if base.endswith("es") and len(base) > 2:
        variants.add(base[:-2])
    if base.endswith("s") and len(base) > 1:
        variants.add(base[:-1])

    variants.add(base + "s")
    variants.add(base + "es")

    if base.endswith("y"):
        variants.add(base[:-1] + "ies")

    # Known Caltech101 aliases.
    alias_map = {
        "face": ["faces", "faces easy", "face easy"],
        "faces": ["face", "faces easy", "face easy"],
        "faces easy": ["face", "faces"],
        "leopard": ["leopards"],
        "leopards": ["leopard"],
        "motorbike": ["motorbikes"],
        "motorbikes": ["motorbike"],
        "airplane": ["airplanes"],
        "airplanes": ["airplane"],
    }

    for alias in alias_map.get(base, []):
        alias_norm = _norm_name(alias)
        variants.add(alias_norm)
        variants.add(alias_norm.replace(" ", ""))

    return [v for v in variants if v]


def _norm_dataset_key(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _safe_filename(x: str) -> str:
    x = str(x).replace("/", "-").replace("\\", "-")
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    return x.strip("_") or "unknown"


def _sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CapelAdapter(BaseAdapter):
    """
    CAPEL: Cluster-Aware Prompt Ensemble Learning.

    Implementation:
    - W is trainable and has shape [C, K, D].
    - alpha is implemented as trainable prompt_logits [C, K], normalized by
      softmax in get_prompt_weights().
    - Forward logits are computed in logits space, not by averaging text
      features into [C, D].
    - Cluster-preserving regularization is implemented in pc_loss() by
      minimizing H(k | x, y_true) over the K sub-classifiers of the ground-truth
      class.
    - Prompt text features are cached to avoid re-encoding C x K prompts every
      run.
    """

    initialization_name = "CAPEL"
    adapter_kind = "capel_prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, classnames: List[str]):
        super().__init__(cfg, clip_model, base_text_features)

        self.classnames = list(classnames)
        self.n_classes = int(base_text_features.shape[0])
        self.feat_dim = int(base_text_features.shape[1])

        cad = cfg.CLIP_ADAPTERS
        self.prompt_bank_path = str(
            getattr(
                cad,
                "CAPEL_PROMPT_BANK",
                "/root/autodl-tmp/MMRL/prompts/capel_prompt_bank_all.json",
            )
        )
        self.k = int(getattr(cad, "CAPEL_PROMPTS_PER_CLASS", 50))
        self.pc_lambda = float(getattr(cad, "CAPEL_PC_LAMBDA", 3.0))
        self.strict_prompt_bank = bool(getattr(cad, "CAPEL_STRICT_PROMPT_BANK", True))
        self.fallback_order = bool(getattr(cad, "CAPEL_FALLBACK_ORDER", False))

        self.use_feature_cache = bool(getattr(cad, "CAPEL_USE_FEATURE_CACHE", True))
        self.rebuild_feature_cache = bool(getattr(cad, "CAPEL_REBUILD_FEATURE_CACHE", False))
        self.feature_cache_dir = str(
            getattr(
                cad,
                "CAPEL_FEATURE_CACHE_DIR",
                "/root/autodl-tmp/MMRL/prompts/capel_feature_cache",
            )
        )

        print("[CAPEL] before building/loading prompt prototypes", flush=True)
        init_prototypes = self._build_or_load_capel_prototypes(
            cfg=cfg,
            clip_model=clip_model,
            base_text_features=base_text_features,
        )
        print("[CAPEL] after building/loading prompt prototypes", flush=True)

        # CAPEL trainable W: [C, K, D]
        self.prototypes = nn.Parameter(init_prototypes)

        self.prompt_logits = nn.Parameter(
            torch.zeros(
                self.n_classes,
                self.k,
                dtype=torch.float32,
                device=base_text_features.device,
            )
        )

        print(
            "[CAPEL] initialized: "
            f"dataset={cfg.DATASET.NAME}, "
            f"classes={self.n_classes}, "
            f"K={self.k}, "
            f"D={self.feat_dim}, "
            f"prompt_bank={self.prompt_bank_path}, "
            f"use_feature_cache={self.use_feature_cache}",
            flush=True,
        )



    def _select_dataset_bank(self, bank: dict, dataset_name: str) -> dict:
        """
        New prompt-bank format only:

        {
        "Caltech101": {
            "accordion": ["...", "..."],
            "airplanes": ["...", "..."]
        },
        "DescText": {
            "banded": ["...", "..."]
        }
        }

        Return:
        dataset_bank: dict[class_name, list[str]]
        """
        if not isinstance(bank, dict):
            raise ValueError(f"CAPEL prompt bank root must be dict, got {type(bank)}")

        datasets = {
            k: v
            for k, v in bank.items()
            if isinstance(v, dict)
        }

        if not datasets:
            raise ValueError("CAPEL prompt bank contains no dataset dictionaries.")

        # Direct match.
        if dataset_name in datasets:
            return datasets[dataset_name]

        normalized = {_norm_dataset_key(k): (k, v) for k, v in datasets.items()}
        key = _norm_dataset_key(dataset_name)

        # Normalized match, e.g. oxford_pets -> OxfordPets.
        if key in normalized:
            bank_key, value = normalized[key]
            print(
                f"[CAPEL] matched dataset bank by normalized key: "
                f"{dataset_name} -> {bank_key}",
                flush=True,
            )
            return value

        # Dataset-name aliases between MMRL/Dassl configs and new prompt-bank names.
        dataset_aliases = {
            "describabletextures": ["desctext", "dtd"],
            "dtd": ["desctext", "describabletextures"],
            "desctext": ["describabletextures", "dtd"],

            "caltech101": ["caltech101"],
            "ucf101": ["ucf101"],
            "food101": ["food101"],
            "eurosat": ["eurosat"],
            "imagenet": ["imagenet"],
            "sun397": ["sun397"],

            "oxfordpets": ["oxfordpets", "oxfordpet"],
            "oxfordpet": ["oxfordpets"],

            "oxfordflowers": ["oxfordflowers", "oxfordflower", "flowers102"],
            "oxfordflower": ["oxfordflowers", "flowers102"],
            "flowers102": ["oxfordflowers"],

            "stanfordcars": ["stanfordcars", "stanfordcar", "cars"],
            "stanfordcar": ["stanfordcars"],
            "cars": ["stanfordcars"],

            "fgvcaircraft": ["fgvcaircraft", "aircraft", "fgvc"],
            "aircraft": ["fgvcaircraft"],
            "fgvc": ["fgvcaircraft"],
        }

        for alias in dataset_aliases.get(key, []):
            alias_key = _norm_dataset_key(alias)
            if alias_key in normalized:
                bank_key, value = normalized[alias_key]
                print(
                    f"[CAPEL] matched dataset bank by alias: "
                    f"{dataset_name} -> {bank_key}",
                    flush=True,
                )
                return value

        available = ", ".join(sorted(datasets.keys()))
        raise KeyError(
            f"CAPEL prompt bank has no dataset '{dataset_name}'. "
            f"Available datasets: {available}"
        )



    def _load_prompts_for_current_classes(self, cfg) -> List[List[str]]:
        """
        Load prompts from the new flat prompt-bank format:

        {
        "DatasetName": {
            "class_name": [
            "prompt 1",
            ...
            "prompt 50"
            ]
        }
        }
        """
        path = Path(self.prompt_bank_path)
        if not path.is_file():
            raise FileNotFoundError(f"CAPEL prompt bank not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            bank = json.load(f)

        dataset_bank = self._select_dataset_bank(bank, cfg.DATASET.NAME)

        by_name = {}
        for raw_name, prompts in dataset_bank.items():
            if not isinstance(prompts, list):
                continue

            clean_prompts = [
                str(p).strip()
                for p in prompts
                if str(p).strip()
            ]

            for key in _name_variants(raw_name):
                by_name[key] = clean_prompts

        prompts_all = []
        missing = []

        for cls_name in self.classnames:
            prompts = None

            for key in _name_variants(cls_name):
                if key in by_name:
                    prompts = by_name[key]
                    break

            if prompts is None:
                missing.append(cls_name)
                prompts_all.append(None)
            else:
                prompts_all.append(prompts)

        if missing:
            available_preview = list(dataset_bank.keys())[:30]
            raise KeyError(
                "CAPEL prompt bank cannot match these dataset classnames: "
                f"{missing[:20]} "
                f"(total missing={len(missing)}). "
                f"Dataset={cfg.DATASET.NAME}. "
                f"First prompt-bank classes={available_preview}. "
                "Please make sure JSON class keys match dataset.classnames after "
                "normalization."
            )

        fixed = []
        for cls_name, prompts in zip(self.classnames, prompts_all):
            prompts = list(prompts or [])

            if len(prompts) < self.k:
                if self.strict_prompt_bank:
                    raise ValueError(
                        f"Class '{cls_name}' has only {len(prompts)} prompts, "
                        f"but CAPEL_PROMPTS_PER_CLASS={self.k}."
                    )

                repeats = (self.k + len(prompts) - 1) // max(1, len(prompts))
                prompts = (prompts * repeats)[: self.k]
            else:
                prompts = prompts[: self.k]

            fixed.append(prompts)

        return fixed





    def _cache_metadata(self, cfg, base_text_features: torch.Tensor) -> dict:
        prompt_bank = Path(self.prompt_bank_path)
        classnames_norm = [_norm_name(x) for x in self.classnames]
        classnames_hash = _sha1_text(json.dumps(classnames_norm, ensure_ascii=False))

        backbone_name = str(getattr(cfg.MODEL.BACKBONE, "NAME", "unknown"))
        dataset_name = str(cfg.DATASET.NAME)

        return {
            "format_version": 1,
            "method": "CAPEL",
            "dataset_name": dataset_name,
            "backbone_name": backbone_name,
            "prompt_bank_path": str(prompt_bank.resolve()) if prompt_bank.exists() else str(prompt_bank),
            "prompt_bank_sha1": _sha1_file(prompt_bank) if prompt_bank.exists() else "",
            "classnames_hash": classnames_hash,
            "n_classes": self.n_classes,
            "prompts_per_class": self.k,
            "feat_dim": self.feat_dim,
            "dtype": str(base_text_features.dtype),
        }

    def _cache_path(self, cfg, base_text_features: torch.Tensor) -> Path:
        meta = self._cache_metadata(cfg, base_text_features)

        dataset = _safe_filename(meta["dataset_name"])
        backbone = _safe_filename(meta["backbone_name"])
        bank_hash = meta["prompt_bank_sha1"][:10]
        cls_hash = meta["classnames_hash"][:10]

        filename = (
            f"{dataset}_{backbone}_"
            f"C{self.n_classes}_K{self.k}_D{self.feat_dim}_"
            f"bank{bank_hash}_cls{cls_hash}.pt"
        )

        return Path(self.feature_cache_dir) / filename

    def _load_cached_prototypes(
        self,
        cfg,
        base_text_features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_feature_cache:
            return None

        if self.rebuild_feature_cache:
            print("[CAPEL] CAPEL_REBUILD_FEATURE_CACHE=True, ignoring existing cache", flush=True)
            return None

        cache_path = self._cache_path(cfg, base_text_features)
        if not cache_path.is_file():
            print(f"[CAPEL] prompt feature cache not found: {cache_path}", flush=True)
            return None

        expected_meta = self._cache_metadata(cfg, base_text_features)

        try:
            payload = torch.load(cache_path, map_location="cpu")
        except Exception as e:
            print(f"[CAPEL] failed to load cache {cache_path}: {repr(e)}", flush=True)
            return None

        if not isinstance(payload, dict):
            print(f"[CAPEL] invalid cache payload type at {cache_path}: {type(payload)}", flush=True)
            return None

        cached_meta = payload.get("metadata", {})
        cached_features = payload.get("prototypes", None)

        keys_to_check = [
            "format_version",
            "method",
            "dataset_name",
            "backbone_name",
            "prompt_bank_sha1",
            "classnames_hash",
            "n_classes",
            "prompts_per_class",
            "feat_dim",
        ]

        for key in keys_to_check:
            if cached_meta.get(key) != expected_meta.get(key):
                print(
                    "[CAPEL] cache metadata mismatch, rebuilding: "
                    f"key={key}, cached={cached_meta.get(key)}, expected={expected_meta.get(key)}",
                    flush=True,
                )
                return None

        if not torch.is_tensor(cached_features):
            print(f"[CAPEL] cache has no tensor prototypes: {cache_path}", flush=True)
            return None

        expected_shape = (self.n_classes, self.k, self.feat_dim)
        if tuple(cached_features.shape) != expected_shape:
            print(
                "[CAPEL] cache tensor shape mismatch, rebuilding: "
                f"cached={tuple(cached_features.shape)}, expected={expected_shape}",
                flush=True,
            )
            return None

        print(f"[CAPEL] loaded prompt feature cache: {cache_path}", flush=True)

        return cached_features.to(
            device=base_text_features.device,
            dtype=base_text_features.dtype,
        )

    def _save_cached_prototypes(
        self,
        cfg,
        base_text_features: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> None:
        if not self.use_feature_cache:
            return

        cache_path = self._cache_path(cfg, base_text_features)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": self._cache_metadata(cfg, base_text_features),
            # Save in CPU float32 to make cache independent of AMP/fp32 runtime.
            "prototypes": prototypes.detach().cpu().float(),
        }

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(cache_path)

        print(f"[CAPEL] saved prompt feature cache: {cache_path}", flush=True)

    def _build_or_load_capel_prototypes(
        self,
        cfg,
        clip_model,
        base_text_features: torch.Tensor,
    ) -> torch.Tensor:
        cached = self._load_cached_prototypes(cfg, base_text_features)
        if cached is not None:
            return cached

        prototypes = self._build_capel_prototypes(
            cfg=cfg,
            clip_model=clip_model,
            base_text_features=base_text_features,
        )

        self._save_cached_prototypes(cfg, base_text_features, prototypes)
        return prototypes

    @torch.no_grad()
    def _build_capel_prototypes(self, cfg, clip_model, base_text_features: torch.Tensor) -> torch.Tensor:
        prompts_all = self._load_prompts_for_current_classes(cfg)
        total_prompts = sum(len(x) for x in prompts_all)

        print(
            f"[CAPEL] building prompt features: dataset={cfg.DATASET.NAME}, "
            f"classes={len(prompts_all)}, total_prompts={total_prompts}",
            flush=True,
        )

        text_encoder = CLIPTextEncoder(clip_model).to(base_text_features.device)
        token_embedding_device = clip_model.token_embedding.weight.device

        class_features = []
        for class_idx, prompts in enumerate(prompts_all):
            if class_idx % 10 == 0 or class_idx == len(prompts_all) - 1:
                print(
                    f"[CAPEL] encoding prompts {class_idx + 1}/{len(prompts_all)}",
                    flush=True,
                )

            tokens = clip.tokenize(prompts).to(token_embedding_device)
            prompt_embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)

            feats = text_encoder(
                prompt_embeddings.to(text_encoder.device),
                tokens.to(text_encoder.device),
            )
            feats = F.normalize(feats.float(), dim=-1)

            class_features.append(
                feats.to(
                    device=base_text_features.device,
                    dtype=base_text_features.dtype,
                )
            )

        prototypes = torch.stack(class_features, dim=0)  # [C, K, D]

        if prototypes.shape != (self.n_classes, self.k, self.feat_dim):
            raise RuntimeError(
                "CAPEL prototype shape mismatch: "
                f"got {tuple(prototypes.shape)}, "
                f"expected {(self.n_classes, self.k, self.feat_dim)}"
            )

        return prototypes

    @torch.no_grad()
    def prompt_weight_stats(self) -> dict:
        w = self.get_prompt_weights().detach().float()
        return {
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "mean": float(w.mean().item()),
            "std": float(w.std().item()),
        }


    def get_prompt_weights(self) -> torch.Tensor:
        """
        CAPEL learnable prompt attention weights.

        Old version:
        - prompt_logits has shape [C, K]
        - softmax over K gives per-class prompt weights
        - zero initialization gives uniform average-logit ensembling at start
        """
        return F.softmax(self.prompt_logits.float(), dim=-1).to(self.prototypes.dtype)
    

    
    def get_prototypes(self) -> torch.Tensor:
        raise RuntimeError(
            "CapelAdapter uses [C, K, D] prototypes and must be handled by "
            "adapter_kind == 'capel_prototype', not by lp_logits()."
        )

    def get_constraint_reference(self) -> torch.Tensor:
        """
        If a generic CLAP-style constraint is enabled with CAPEL, use the
        weighted feature-space average only as a constraint reference. This is
        not used for CAPEL logits; inference still uses logit-space ensembling.
        """
        weights = self.get_prompt_weights().unsqueeze(-1)  # [C, K, 1]
        return (self.prototypes * weights).sum(dim=1)

    def pc_loss(self, sub_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cluster-preserving conditional entropy.

        sub_logits: [B, C, K]
        labels:     [B]

        Uses P(k | x, y_true), i.e. softmax over the K sub-classifier logits for
        the ground-truth class, then minimizes entropy.
        """
        if sub_logits.ndim != 3:
            raise ValueError(f"CAPEL sub_logits must be [B, C, K], got {tuple(sub_logits.shape)}")

        b = sub_logits.shape[0]
        idx = torch.arange(b, device=sub_logits.device)
        true_sub_logits = sub_logits[idx, labels.long(), :]  # [B, K]

        p = F.softmax(true_sub_logits, dim=-1)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()
        return entropy