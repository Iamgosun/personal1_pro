from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from data.build import build_split_dataset


class CLIPFeatureExtractor:
    def __init__(self, trainer, cache_manager):
        self.trainer = trainer
        self.cache_manager = cache_manager

    def _select_source(self, split: str):
        if split == "train":
            return self.trainer.dm.dataset.train_x, True
        if split == "val":
            return self.trainer.dm.dataset.val, False
        return self.trainer.dm.dataset.test, False

    def _build_loader(self, split: str, train_aug: bool):
        data_source, default_train = self._select_source(split)
        dataset = build_split_dataset(
            self.trainer.cfg,
            data_source,
            is_train=(train_aug if split == "train" else default_train),
        )

        # Keep item order stable across reps.
        # Random image augmentations still happen through transform when train_aug=True.
        sampler = SequentialSampler(dataset)

        batch_size = (
            self.trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
            if split == "train"
            else self.trainer.cfg.DATALOADER.TEST.BATCH_SIZE
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    @staticmethod
    def _is_valid_cached_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False

        required = ("labels", "features")
        if any(key not in payload for key in required):
            return False

        labels = payload["labels"]
        features = payload["features"]

        if not torch.is_tensor(labels):
            return False
        if not torch.is_tensor(features):
            return False

        if labels.ndim != 1:
            return False
        if features.ndim != 2:
            return False

        n = int(labels.shape[0])
        if int(features.shape[0]) != n:
            return False

        return True
    @torch.no_grad()
    def _compute_logits_from_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recompute adapter-specific logits from cached CLIP image features.

        The feature cache is shared across all adapters. Logits are deliberately
        not reused from cache because they depend on the current adapter.
        """
        model_was_training = bool(self.trainer.model.training)
        method_was_training = bool(self.trainer.method.training)

        self.trainer.model.eval()
        self.trainer.method.eval()

        batch_size = int(self.trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE)
        logits_all = []

        try:
            n = int(features.shape[0])
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)

                feat_b = features[start:end].to(self.trainer.device)
                label_b = labels[start:end].to(self.trainer.device)

                outputs = self.trainer.method.forward_eval(
                    {"features": feat_b, "label": label_b},
                    None,
                )

                logits_all.append(outputs.logits.detach().cpu())

        finally:
            if model_was_training:
                self.trainer.model.train()
            if method_was_training:
                self.trainer.method.train()

        return torch.cat(logits_all, dim=0)

    def _aggregation_mode(self, split: str) -> str:
        """
        CLAP-style default:
        - train split uses a feature pool: reps augmentations are concatenated.
        - val/test keep mean aggregation, though reps is normally 1 there.
        """
        clip_cfg = getattr(self.trainer.cfg, "CLIP_ADAPTERS", None)
        mode = str(getattr(clip_cfg, "CACHE_AGGREGATION", "pool")).lower()

        if mode not in {"pool", "mean"}:
            raise ValueError(
                "CLIP_ADAPTERS.CACHE_AGGREGATION must be 'pool' or 'mean', "
                f"got {mode!r}"
            )

        if split != "train":
            return "mean"

        return mode

    def extract_split(
        self,
        split: str,
        reps: int = 1,
        train_aug: bool = False,
        mode: str = "features_only",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reps = int(reps)
        if reps < 1:
            raise ValueError(f"reps must be >= 1, got {reps}")

        aggregation = self._aggregation_mode(split)

        spec = self.cache_manager.build_spec(
            split=split,
            reps=reps,
            train_aug=train_aug,
            mode=mode,
            aggregation=aggregation,
        )

        cached = self.cache_manager.load(spec)
        if cached is not None and self._is_valid_cached_payload(cached):
            labels = cached["labels"]
            features = cached["features"]
            logits = self._compute_logits_from_features(features, labels)

            print(
                f"[FeatureCache] reuse shared CLIP feature cache: "
                f"split={split}, id={spec.cache_id}, path={spec.tensor_path}",
                flush=True,
            )

            return (
                labels.to(self.trainer.device),
                logits.to(self.trainer.device),
                features.to(self.trainer.device),
            )

        print(
            f"[FeatureCache] build shared CLIP feature cache: "
            f"split={split}, id={spec.cache_id}, path={spec.tensor_path}",
            flush=True,
        )

        model_was_training = bool(self.trainer.model.training)
        method_was_training = bool(self.trainer.method.training)

        self.trainer.model.eval()
        self.trainer.method.eval()

        try:
            loader = self._build_loader(split, train_aug)

            base_labels = None
            labels_all = []
            logits_all = []
            features_all = []

            for rep_idx in range(reps):
                labels_rep = []
                logits_rep = []
                features_rep = []

                iterator = tqdm(
                    loader,
                    desc=f"cache {split} rep {rep_idx + 1}/{reps}",
                    leave=False,
                )

                for batch in iterator:
                    with torch.no_grad():
                        images = batch["img"].to(self.trainer.device)
                        labels = batch["label"].to(self.trainer.device)

                        outputs = self.trainer.method.forward_eval(
                            {"img": images, "label": labels},
                            None,
                        )

                        feat = outputs.features.get("img")
                        if feat is None:
                            raise RuntimeError(
                                "Cache mode requires image features in "
                                "outputs.features['img']"
                            )

                        labels_rep.append(labels.detach().cpu())
                        logits_rep.append(outputs.logits.detach().cpu())
                        features_rep.append(feat.detach().cpu())

                labels_rep = torch.cat(labels_rep, dim=0)
                logits_rep = torch.cat(logits_rep, dim=0)
                features_rep = torch.cat(features_rep, dim=0)

                if base_labels is None:
                    base_labels = labels_rep
                elif not torch.equal(labels_rep, base_labels):
                    raise RuntimeError(
                        "Feature-cache extraction produced different label order "
                        "between reps. This would corrupt repeated augmented "
                        f"features. split={split}, rep={rep_idx + 1}"
                    )

                labels_all.append(labels_rep)
                logits_all.append(logits_rep)
                features_all.append(features_rep)

            if aggregation == "pool":
                # CLAP-style feature pool:
                # [reps, N, D] -> [reps * N, D]
                labels = torch.cat(labels_all, dim=0)
                logits = torch.cat(logits_all, dim=0)
                features = torch.cat(features_all, dim=0)
            else:
                # TTA-style mean feature. Kept only for ablations/backward compatibility.
                labels = base_labels
                logits = torch.stack(logits_all, dim=0).mean(0)
                features = torch.stack(features_all, dim=0).mean(0)

            logits = self._compute_logits_from_features(features, labels)

            payload = {
                "labels": labels,
                "features": features,
                "meta": {
                    "num_samples": int(labels.shape[0]),
                    "base_num_samples": int(base_labels.shape[0]),
                    "feature_dim": int(features.shape[-1]),
                    "reps": reps,
                    "train_aug": bool(train_aug),
                    "mode": mode,
                    "aggregation": aggregation,
                    "payload": "clip_features_and_labels_only",
                },
            }

            self.cache_manager.save(spec, payload)

        finally:
            if model_was_training:
                self.trainer.model.train()
            if method_was_training:
                self.trainer.method.train()

        return (
            labels.to(self.trainer.device),
            logits.to(self.trainer.device),
            features.to(self.trainer.device),
        )