import copy
import datetime
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import AverageMeter, MetricMeter, load_checkpoint, load_pretrained_weights

from .clip_adapter_stage2 import (
    CustomCLIP,
    FeatureCacheManager,
    FeaturePipeline,
    load_clip_to_cpu,
    save_confidence_coverage_stats,
    save_final_reports,
)

torch.backends.cuda.matmul.allow_tf32 = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True


class TrainerXCostume(SimpleTrainer):
    """Base trainer for feature-based training used by clip_adapters."""

    def run_epoch(self):
        self.set_model_mode("eval")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size

        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        if "CrossModal" in self.model.adapter.initialization:
            idx = np.random.choice(list(np.arange(0, features.shape[0])), features.shape[0] // 2)
            features = features[idx, :]
            labels = labels[idx]

        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(features[batch_init:batch_end], labels[batch_init:batch_end])
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = [
                    f"epoch [{self.epoch + 1}/{self.max_epoch}]",
                    f"batch [{self.batch_idx + 1}/{self.num_batches}]",
                    f"{losses}",
                    f"eta {eta}",
                ]
                print(" ".join(info))
                print("**********************************************************")

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
        return loss_summary


@TRAINER_REGISTRY.register()
class ClipADAPTER(TrainerXCostume):
    """Stage-2 refactor of clip_adapters.

    What changed relative to the analysis doc:
    - Kept the external Dassl trainer entrypoint `ClipADAPTER` for drop-in compatibility.
    - Extracted text encoding, adapter variants, custom CLIP wrapper, feature cache,
      and reporting into `trainers/clip_adapter_stage2/`.
    - Did *not* yet replace Dassl trainer with the full `Method + Executor` architecture.
      That larger change would touch entrypoints and experiment scripts more broadly.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ClipADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.ClipADAPTER.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)
            else:
                print(name)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter", self.model.adapter, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.ClipADAPTER.PREC == "amp" else None

        self.cache_manager = FeatureCacheManager(cfg, self.model.adapter.initialization)
        self.feature_pipeline = FeaturePipeline(self, self.cache_manager)

    def reset_hyperparams(self, params):
        if "ClipA" in self.model.adapter.initialization:
            self.model.adapter.reset_hparams({"ratio": params["ratio"]})
        if "TipA" in self.model.adapter.initialization:
            self.model.adapter.reset_hparams({"alpha": params["alpha"], "beta": params["beta"]})
        if "TR" in self.model.adapter.initialization:
            self.model.adapter.reset_hparams({"alpha": params["alpha"]})

        if "TipA" in self.model.adapter.initialization:
            self.model.adapter.init_tipadapter(self.features_train, self.labels_train)
            if "-f-" in self.model.adapter.initialization:
                self.max_epoch = 20

        self.model.to(self.device)
        self.model = self.model.float()
        if "lr" in params:
            self.cfg.OPTIM["LR"] = params["lr"]
        self.optim = build_optimizer(self.model.adapter, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self._models.popitem(), self._optims.popitem(), self._scheds.popitem()
        self.register_model("adapter" + str(random.random()), self.model.adapter, self.optim, self.sched)
        return 1

    def train(self):
        self.set_model_mode("eval")
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print(
            f"Zero-Shot accuracy on test: {round(compute_accuracy(output_test.to(self.device), self.labels_test.to(self.device))[0].item(), 2)}"
        )

        self.labels_train, self.logits_zs, self.features_train = self.extract_features(
            partition="train",
            reps=self.model.adapter.epochs_aumentation,
            train_is_augf=self.model.adapter.augmentations,
        )

        if "CrossModal" in self.model.adapter.initialization:
            print("Preparing cross-modal dataset... resampling text prompts")
            zs_prototypes = self.model.text_embeddings_all.cpu().numpy()
            zs_labels = np.repeat(np.expand_dims(np.arange(0, zs_prototypes.shape[0]), 0), zs_prototypes.shape[1], 0)
            zs_prototypes = np.reshape(
                np.transpose(zs_prototypes, (2, 1, 0)),
                (zs_prototypes.shape[-1], zs_prototypes.shape[0] * zs_prototypes.shape[1]),
            ).transpose()
            zs_labels = np.transpose(zs_labels, (1, 0)).flatten()
            idx = np.random.choice(list(np.arange(0, len(zs_labels))), self.features_train.shape[0])
            zs_labels = zs_labels[idx]
            zs_prototypes = zs_prototypes[idx, :]
            self.features_train = torch.cat([self.features_train, torch.tensor(zs_prototypes).to(self.device)], dim=0)
            self.labels_train = torch.cat([self.labels_train, torch.tensor(zs_labels).to(self.device)])

        if self.model.adapter.apply_constraint not in ["none", "KL"]:
            print("Getting initial lagrangian multipliers for constraint formulation")
            self.model.adapter.device = self.device
            self.model.adapter.init_lagrangian_multipliers(self.labels_train, self.logits_zs)
            print("Lagrangian multipliers:")
            print(list(torch.round(self.model.adapter.alpha_constraint.detach(), decimals=3).cpu().numpy()))

        if "TipA" in self.model.adapter.initialization:
            self.model.adapter.init_tipadapter(self.features_train, self.labels_train)
            self.optim = build_optimizer(self.model.adapter, self.cfg.OPTIM)
            self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
            self.register_model("adapter_tipa-f-", self.model.adapter, self.optim, self.sched)
            if "-f-" not in self.model.adapter.initialization:
                self.max_epoch = 1

        summary_grid = []
        if "grid_search" in self.model.adapter.initialization:
            from sklearn.model_selection import ParameterGrid
            import pandas as pd

            best_acc = 0.0
            best_setting = []
            grid = ParameterGrid(self.model.adapter.grid_search_param)
            for params in grid:
                print("Iteration grid hyperparameters search:")
                print(params)
                self.reset_hyperparams(params)
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    self.before_epoch()
                    loss_summary = self.run_epoch()
                    if loss_summary["acc_test"] > best_acc:
                        best_acc = loss_summary["acc_test"]
                        best_setting = params
                    self.epoch = -1
                    self.after_epoch()
                params["acc_test"] = loss_summary["acc_test"]
                summary_grid.append(params)
                print("Current configuration:")
                print(params)
                print("A on test:")
                print(loss_summary["acc_test"])

            print("Best configuration:")
            print(best_setting)
            print("Best accuracy on test:")
            print(best_acc)
            pd.DataFrame(summary_grid).to_csv(self.cfg.OUTPUT_DIR + "/grid_search.csv")
        else:
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                if "adaptative" in self.model.adapter.apply_constraint and self.model.adapter.apply_constraint != "KL":
                    self.model.adapter.outer_step()
                if self.epoch >= self.model.adapter.anneal_start_epoch:
                    self.model.adapter.kl_weight += self.model.adapter.anneal_rate
                    self.model.adapter.kl_weight = min(self.model.adapter.kl_weight, 1.0)
                print(f"Current KL weight: {self.model.adapter.kl_weight:.4f}")
                self.after_epoch()

        self.after_train()

    def _save_confidence_coverage(self):
        logits = self.model.forward_features(self.features_test.clone().detach().to(self.device))
        return save_confidence_coverage_stats(self.cfg.OUTPUT_DIR, self.labels_test.to(self.device), logits)

    def after_train(self):
        super().after_train()
        self._save_confidence_coverage()
        save_final_reports(
            model=self.model,
            cfg=self.cfg,
            device=self.device,
            features_test=self.features_test,
            labels_test=self.labels_test,
            features_ood=getattr(self, "features_ood", None),
        )

    def forward_backward(self, features, labels):
        prec = self.cfg.TRAINER.ClipADAPTER.PREC
        features = torch.as_tensor(features, device=self.device)
        labels = labels.to(self.device)

        if prec == "amp":
            with autocast():
                output = self.model.forward_features(features)
                loss_ce = F.cross_entropy(output, labels)
                if self.model.adapter.apply_constraint != "none":
                    loss_constraint = self.model.adapter.zero_shot_constraint()
                    loss = loss_ce + loss_constraint
                else:
                    loss = loss_ce
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model.forward_features(features)
            loss_ce = F.cross_entropy(output, labels)
            loss = loss_ce
            if self.model.adapter.apply_constraint != "none" and self.model.adapter.apply_constraint == "l2":
                loss = loss_ce + self.model.adapter.zero_shot_constraint()
            elif self.model.adapter.apply_constraint != "none" and self.model.adapter.apply_constraint == "KL":
                loss_kl = self.model.adapter.kl_divergence()
                loss = loss_ce + self.model.adapter.kl_weight * loss_kl
                print(f"loss_ce: {loss_ce.item()}")
                print(f"KL divergence: {loss_kl.item()}, weight: {self.model.adapter.kl_weight}")
            self.model_backward_and_update(loss)

        with torch.no_grad():
            output_test = self.model.forward_features(self.features_test.clone().detach().to(self.device))
            save_confidence_coverage_stats(self.cfg.OUTPUT_DIR, self.labels_test.to(self.device), output_test)

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test.to(self.device))[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss_summary

    def load_model(self, directory, cfg, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        print("Pretrained model given")
        if self.model.adapter.initialization == "TipA":
            epoch = 1

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')
            print(f'Model found at "{model_path}"')
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            if "TipA" in self.model.adapter.initialization:
                self.model.adapter.cache_keys = nn.Parameter(state_dict["cache_keys"].clone())
                self.model.adapter.cache_values = nn.Parameter(state_dict["cache_values"].clone())

            if self.cfg.DATASET.NAME in ["ImageNetA", "ImageNetR"]:
                if self.cfg.DATASET.NAME == "ImageNetA":
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
                else:
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
                imageneta_indexes = find_indexes()
                state_dict["base_text_features"] = state_dict["base_text_features"][imageneta_indexes]
                state_dict["prototypes"] = state_dict["prototypes"][imageneta_indexes]
                if "TipA" in self.model.adapter.initialization:
                    state_dict["cache_values"] = state_dict["cache_values"][:, imageneta_indexes]
                    self.model.adapter.cache_keys = nn.Parameter(state_dict["cache_keys"].clone())
                    self.model.adapter.cache_values = nn.Parameter(state_dict["cache_values"].clone())
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
            self.model.float()

    def _get_feature_cache_path(self, partition, reps):
        spec = self.cache_manager.build_spec(partition, reps, train_is_augf=(partition == "train" and self.model.adapter.augmentations))
        return spec.tensor_path

    def parse_batch_test(self, batch):
        return self.feature_pipeline.parse_batch_test(batch)

    def extract_features(self, partition, reps=1, train_is_augf=False):
        return self.feature_pipeline.extract_features(partition=partition, reps=reps, train_is_augf=train_is_augf)
