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
        Official BayesAdapter-style feature-view sampling.

        In pool cache mode, train features are laid out as:

            [rep0 sample0..N-1,
            rep1 sample0..N-1,
            ...
            repR sample0..N-1]

        Official BayesAdapter does NOT sample N items uniformly from the 20N pool.
        Instead, for every original sample i, each epoch randomly selects exactly
        one augmented view among the R cached views.

        Therefore each epoch contains exactly N items:
            index_i = sampled_rep_i * N + i

        This guarantees:
        - every original sample appears exactly once per epoch;
        - each sample uses one randomly selected augmentation view;
        - no original sample is duplicated or missing within the epoch.
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

        # If cache is already N, no view sampling is needed.
        if dataset_size <= base_train_size:
            return trainer.cache_train_loader

        n = max(1, int(base_train_size))

        # Strict official-style path only applies when the pool is exactly
        # an integer number of full augmented passes over the original train set.
        #
        # For plain BayesAdapter with CACHE_REPS=20 and pool aggregation:
        #     dataset_size == 20 * n
        if dataset_size % n == 0:
            num_reps = dataset_size // n

            # sample_ids: [0, 1, ..., N-1]
            sample_ids = torch.arange(n, dtype=torch.long)

            # For every original sample, choose one augmentation rep.
            # rep_ids[i] is the selected view for original sample i.
            rep_ids = torch.randint(
                low=0,
                high=num_reps,
                size=(n,),
                dtype=torch.long,
            )

            indices = (rep_ids * n + sample_ids).tolist()

            # Official BayesAdapter iterates over saved feature batches without
            # shuffling. Keep shuffle=False below for the closest reproduction.
            #
            # If you prefer your existing random batch order while still preserving
            # "one view per original sample", uncomment these two lines:
            #
            # perm = torch.randperm(n).tolist()
            # indices = [indices[i] for i in perm]

        else:
            # Fallback for adapters that modify the cache size after extraction
            # e.g. CrossModal appending text features. This preserves the old
            # behavior instead of silently producing wrong indexing.
            indices = torch.randperm(dataset_size)[:n].tolist()

        return DataLoader(
            Subset(dataset, indices),
            batch_size=trainer.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            shuffle=False,
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
