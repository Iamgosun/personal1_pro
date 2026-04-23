from __future__ import annotations

from typing import Tuple

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
        sampler = self.trainer.train_loader_x.sampler if split == "train" else SequentialSampler(dataset)
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

    def extract_split(
        self,
        split: str,
        reps: int = 1,
        train_aug: bool = False,
        mode: str = "features_only",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.cache_manager.build_spec(split=split, reps=reps, train_aug=train_aug, mode=mode)
        cached = self.cache_manager.load(spec)
        if cached is not None:
            return (
                cached["labels"].to(self.trainer.device),
                cached["logits"].to(self.trainer.device),
                cached["features"].to(self.trainer.device),
            )

        model_was_training = bool(self.trainer.model.training)
        method_was_training = bool(self.trainer.method.training)

        self.trainer.model.eval()
        self.trainer.method.eval()

        try:
            loader = self._build_loader(split, train_aug)
            labels_all, logits_all, features_all = [], [], []

            for _ in range(reps):
                labels_rep, logits_rep, features_rep = [], [], []

                for batch in tqdm(loader):
                    with torch.no_grad():
                        images = batch["img"].to(self.trainer.device)
                        labels = batch["label"].to(self.trainer.device)

                        outputs = self.trainer.method.forward_eval({"img": images, "label": labels}, None)

                        labels_rep.append(labels.cpu())
                        logits_rep.append(outputs.logits.detach().cpu())

                        feat = outputs.features.get("img")
                        if feat is None:
                            raise RuntimeError("Cache mode requires image features in outputs.features['img']")
                        features_rep.append(feat.detach().cpu())

                labels_rep = torch.cat(labels_rep, dim=0)
                logits_rep = torch.cat(logits_rep, dim=0)
                features_rep = torch.cat(features_rep, dim=0)

                labels_all.append(labels_rep)
                logits_all.append(logits_rep)
                features_all.append(features_rep)

            labels = labels_all[0]
            logits = torch.stack(logits_all, dim=0).mean(0)
            features = torch.stack(features_all, dim=0).mean(0)

            self.cache_manager.save(
                spec,
                {
                    "labels": labels,
                    "logits": logits,
                    "features": features,
                    "meta": {
                        "num_samples": int(labels.shape[0]),
                        "feature_dim": int(features.shape[-1]),
                    },
                },
            )
        finally:
            if model_was_training:
                self.trainer.model.train()
            if method_was_training:
                self.trainer.method.train()

        return labels.to(self.trainer.device), logits.to(self.trainer.device), features.to(self.trainer.device)