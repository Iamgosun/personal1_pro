from __future__ import annotations

import datetime
import time

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset

from core.registry import EXECUTOR_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import AverageMeter, MetricMeter

from features.cache_dataset import FeatureDataset
from features.cache_manager import FeatureCacheManager
from features.extractor import CLIPFeatureExtractor
from .base_executor import BaseExecutor


@EXECUTOR_REGISTRY.register("cache")
class CacheExecutor(BaseExecutor):
    exec_mode = "cache"

    def _make_feature_loader(self, trainer, features, labels, logits=None, shuffle=True):
        return DataLoader(
            FeatureDataset(
                features.cpu(),
                labels.cpu(),
                None if logits is None else logits.cpu(),
            ),
            batch_size=trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def on_build(self, trainer):
        self.cache_manager = FeatureCacheManager(trainer.cfg)
        self.extractor = CLIPFeatureExtractor(trainer, self.cache_manager)

        cache_reps = int(getattr(trainer.cfg.CLIP_ADAPTERS, "CACHE_REPS", 1))
        cache_train_aug = bool(getattr(trainer.cfg.CLIP_ADAPTERS, "CACHE_TRAIN_AUG", True))

        trainer.labels_train, trainer.logits_train, trainer.features_train = (
            self.extractor.extract_split(
                "train",
                reps=cache_reps,
                train_aug=cache_train_aug,
            )
        )

        # CLAP-aligned hook:
        # - CrossModal concatenates text prompt features into train features.
        # - CLAP constraint initializes class-wise multipliers from zero-shot logits.
        if hasattr(trainer.method, "prepare_cache_data"):
            trainer.method.prepare_cache_data(trainer)

        trainer.cache_train_loader = self._make_feature_loader(
            trainer,
            trainer.features_train,
            trainer.labels_train,
            trainer.logits_train,
            shuffle=True,
        )

        trainer.method.on_cache_ready(trainer)

    def _epoch_loader(self, trainer):
        """
        CLAP CrossModal samples half of the concatenated feature pool each epoch.
        This preserves the original modular data loader while matching that semantic.
        """
        adapter = getattr(trainer.method.model, "adapter", None)
        is_cross_modal = bool(getattr(adapter, "uses_cross_modal", False))
        do_subsample = bool(
            getattr(trainer.cfg.CLIP_ADAPTERS, "CROSS_MODAL_EPOCH_SUBSAMPLE", True)
        )

        if not (is_cross_modal and do_subsample):
            return trainer.cache_train_loader

        dataset = trainer.cache_train_loader.dataset
        n = max(1, len(dataset) // 2)
        indices = torch.randperm(len(dataset))[:n].tolist()

        return DataLoader(
            Subset(dataset, indices),
            batch_size=trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def run_epoch(self, trainer):
        trainer.set_model_mode("eval")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        epoch_loader = self._epoch_loader(trainer)

        trainer.num_batches = len(epoch_loader)
        end = time.time()
        loss_summary = None

        for trainer.batch_idx, batch in enumerate(epoch_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(trainer, batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (trainer.batch_idx + 1) % trainer.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = trainer.num_batches < trainer.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = trainer.num_batches - trainer.batch_idx - 1
                nb_remain += (trainer.max_epoch - trainer.epoch - 1) * trainer.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"epoch [{trainer.epoch + 1}/{trainer.max_epoch}] "
                    f"batch [{trainer.batch_idx + 1}/{trainer.num_batches}] "
                    f"{losses} eta {eta}"
                )

            n_iter = trainer.epoch * trainer.num_batches + trainer.batch_idx
            for name, meter in losses.meters.items():
                trainer.write_scalar("train/" + name, meter.avg, n_iter)
            trainer.write_scalar("train/lr", trainer.get_current_lr(), n_iter)

            end = time.time()

        if loss_summary is None:
            raise RuntimeError("No batches were found in cache training epoch.")

        return loss_summary

    def forward_backward(self, trainer, batch):
        labels = batch["label"].to(trainer.device)
        features = batch["features"].to(trainer.device)
        prec = self.method.get_precision()

        if hasattr(self.method, "set_kl_normalizer"):
            self.method.set_kl_normalizer(getattr(trainer, "num_batches", 1))

        if hasattr(self.method, "set_kl_beta"):
            warmup_epochs = int(getattr(self.method, "kl_warmup_epochs", 0))
            if warmup_epochs > 0:
                kl_beta = min(1.0, float(trainer.epoch) / float(warmup_epochs))
            else:
                kl_beta = 1.0
            self.method.set_kl_beta(kl_beta)

        if hasattr(self.method, "set_epoch_context"):
            self.method.set_epoch_context(trainer.epoch, trainer.max_epoch)

        payload = {"features": features, "label": labels}

        if prec == "amp":
            with autocast():
                outputs = self.method.forward_train(payload)
                loss = self.method.loss(outputs)

            trainer.optim.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(trainer.optim)
            trainer.scaler.update()
        else:
            outputs = self.method.forward_train(payload)
            loss = self.method.loss(outputs)

            trainer.optim.zero_grad()
            loss.backward()
            trainer.optim.step()

        train_logits = self.method.select_train_logits(outputs)
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(train_logits, outputs.labels)[0].item(),
        }

        if hasattr(outputs, "losses") and outputs.losses is not None:
            for key in [
                "loss_ce",
                "loss_constraint",
                "data_term",
                "raw_kl_rep",
                "raw_kl_proj_rep",
                "kl_rep_term",
                "kl_proj_rep_term",
                "kl_term",
                "kl_normalizer",
                "kl_beta",
            ]:
                if key in outputs.losses:
                    value = outputs.losses[key]
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    loss_summary[key] = float(value)

        if (trainer.batch_idx + 1) == trainer.num_batches:
            trainer.update_lr()

        return loss_summary