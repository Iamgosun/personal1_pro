from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from dassl.data.data_manager import DatasetWrapper
from dassl.data.transforms import build_transform


class FeaturePipeline:
    """Owns split selection, DatasetWrapper construction and feature extraction."""

    def __init__(self, trainer, cache_manager):
        self.trainer = trainer
        self.cache_manager = cache_manager

    def parse_batch_test(self, batch):
        input_ = batch["img"].to(self.trainer.device)
        label = batch["label"].to(self.trainer.device)
        return input_, label

    def _build_dataset(self, partition: str, train_is_augf: bool):
        if partition == "train":
            raw_dataset = self.trainer.dm.dataset.train_x
            tfm = build_transform(self.trainer.cfg, is_train=train_is_augf)
            is_train = train_is_augf
        elif partition == "val":
            raw_dataset = self.trainer.dm.dataset.val
            tfm = build_transform(self.trainer.cfg, is_train=False)
            is_train = False
        elif partition == "test":
            raw_dataset = self.trainer.dm.dataset.test
            tfm = build_transform(self.trainer.cfg, is_train=False)
            is_train = False
        else:
            raise ValueError(f"Invalid partition: {partition}")

        dataset = DatasetWrapper(
            cfg=self.trainer.cfg,
            data_source=raw_dataset,
            transform=tfm,
            is_train=is_train,
        )
        return dataset

    def _build_loader(self, partition: str, dataset):
        sampler = self.trainer.train_loader_x.sampler if partition == "train" else SequentialSampler(dataset)
        batch_size = (
            self.trainer.cfg.DATALOADER.TEST.BATCH_SIZE
            if partition != "train"
            else self.trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        )
        pin_memory = torch.cuda.is_available() and self.trainer.cfg.USE_CUDA
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
        )

    def extract_features(self, partition: str, reps: int = 1, train_is_augf: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("Extracting features from:", partition)
        self.trainer.set_model_mode("eval")
        spec = self.cache_manager.build_spec(partition, reps, train_is_augf)
        cached = self.cache_manager.load(spec)
        if cached is not None:
            print(f"[Cache] Found cached features: {spec.tensor_path}")
            return (
                cached["labels"].to(self.trainer.device),
                cached["logits"].to(self.trainer.device),
                cached["features"].to(self.trainer.device),
            )

        dataset = self._build_dataset(partition, train_is_augf)
        data_loader = self._build_loader(partition, dataset)

        if "TipA" not in self.trainer.model.adapter.initialization:
            labels_ds, logits_ds, features_ds = [], [], []
            for _ in range(reps):
                for batch in tqdm(data_loader):
                    with torch.no_grad():
                        input_, label = self.parse_batch_test(batch)
                        logits, features = self.trainer.model(input_, return_features=True)
                        labels_ds.append(label)
                        logits_ds.append(logits.cpu())
                        features_ds.append(features.cpu())
            labels_ds = torch.cat(labels_ds, dim=0)
            logits_ds = torch.cat(logits_ds, dim=0)
            features_ds = torch.cat(features_ds, dim=0)
        else:
            labels_ds, logits_ds, features_ds = [], [], []
            for _ in range(reps):
                labels_rep, logits_rep, features_rep = [], [], []
                for batch in tqdm(data_loader):
                    with torch.no_grad():
                        input_, label = self.parse_batch_test(batch)
                        logits, features = self.trainer.model(input_, return_features=True)
                        labels_rep.append(label)
                        logits_rep.append(logits.cpu())
                        features_rep.append(features.cpu())
                labels_rep = torch.cat(labels_rep, dim=0)
                logits_rep = torch.cat(logits_rep, dim=0)
                features_rep = torch.cat(features_rep, dim=0)
                labels_ds.append(labels_rep.unsqueeze(0))
                logits_ds.append(logits_rep.unsqueeze(0))
                features_ds.append(features_rep.unsqueeze(0))
            labels_ds = torch.cat(labels_ds, dim=0)[0, :]
            logits_ds = torch.cat(logits_ds, dim=0).mean(0)
            features_ds = torch.cat(features_ds, dim=0).mean(0)

        self.cache_manager.save(
            spec,
            {"labels": labels_ds, "features": features_ds, "logits": logits_ds},
        )
        print(f"[Cache] Saved features to {spec.tensor_path}")
        return labels_ds.to(self.trainer.device), logits_ds.to(self.trainer.device), features_ds.to(self.trainer.device)
