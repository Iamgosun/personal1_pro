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

    def _base_train_size(self, trainer) -> int:
        try:
            return int(len(trainer.dm.dataset.train_x))
        except Exception:
            return int(len(trainer.cache_train_loader.dataset))

    def _epoch_loader(self, trainer):
        """
        CLAP-style feature-pool training.

        If CACHE_REPS=20 and CACHE_AGGREGATION=pool, the cached train set has
        roughly 20 * N features. Official CLAP does not consume all 20N features
        every epoch; it uses the feature pool but keeps the epoch size tied to
        the original training loader.

        Therefore, each epoch samples N items from the cached pool, where N is
        len(train_x). This also handles CrossModal after text features are added.
        """
        dataset = trainer.cache_train_loader.dataset
        dataset_size = int(len(dataset))
        base_train_size = self._base_train_size(trainer)

        do_subsample = bool(
            getattr(trainer.cfg.CLIP_ADAPTERS, "CACHE_POOL_EPOCH_SUBSAMPLE", True)
        )

        # Backward-compatible path: train over the full cached dataset.
        if not do_subsample:
            return trainer.cache_train_loader

        # If cache is already N, no need to create a subset.
        if dataset_size <= base_train_size:
            return trainer.cache_train_loader

        n = max(1, base_train_size)
        indices = torch.randperm(dataset_size)[:n].tolist()

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


        adapter = getattr(getattr(self.method, "model", None), "adapter", None)
        is_closed_form = bool(getattr(adapter, "closed_form_adapter", False))

        if is_closed_form:
            with torch.no_grad():
                outputs = self.method.forward_train(payload)
                loss = self.method.loss(outputs)

            train_logits = self.method.select_train_logits(outputs)
            loss_summary = {
                "loss": float(loss.detach().item()),
                "acc": compute_accuracy(train_logits, outputs.labels)[0].item(),
            }

            if hasattr(outputs, "losses") and outputs.losses is not None:
                for key in [
                    "loss_ce",
                    "loss_constraint",
                    "loss_kl",
                    "kl_term",
                ]:
                    if key in outputs.losses:
                        value = outputs.losses[key]
                        if torch.is_tensor(value):
                            value = value.detach().item()
                        loss_summary[key] = float(value)

            # 不 backward，不 step，不 update_lr。
            return loss_summary



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
                "loss_pc",
                "loss_capel_pc",
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




        # CAPEL prompt-gate debug: print once every N epochs, at the last batch.
        if (
            (trainer.batch_idx + 1) == trainer.num_batches
            and hasattr(self.method.model.adapter, "prompt_weight_stats")
        ):
            interval = int(getattr(trainer.cfg.CLIP_ADAPTERS, "CAPEL_GATE_PRINT_EVERY", 10))
            if interval > 0 and ((trainer.epoch + 1) % interval == 0 or trainer.epoch == 0):
                stats = self.method.model.adapter.prompt_weight_stats()
                print(
                    "[CAPEL] "
                    f"epoch={trainer.epoch + 1}/{trainer.max_epoch} "
                    f"prompt_gate min={stats['min']:.6f} "
                    f"max={stats['max']:.6f} "
                    f"mean={stats['mean']:.6f} "
                    f"std={stats['std']:.6f}",
                    flush=True,
                )


        if (trainer.batch_idx + 1) == trainer.num_batches:
            trainer.update_lr()

        return loss_summary
