from __future__ import annotations

import os
import os.path as osp
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from trainers.shared.protocol_router import select_eval_output


class BaseMMRLTrainer(TrainerX):
    trainer_cfg_name: str = ""
    clip_loader = None
    custom_clip_cls = None
    text_encoder_clip_cls = None
    text_feature_extractor = None
    loss_cls = None
    register_name: str = "model"
    trainable_substrings: Tuple[str, ...] = ()
    design_model_name: str = "CLIP"


    def get_trainer_cfg(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        return getattr(cfg.TRAINER, self.trainer_cfg_name)

    def check_cfg(self, cfg):
        assert self.get_trainer_cfg(cfg).PREC in ["fp16", "fp32", "amp"]


    def _freeze_parameters(self):
        enabled = set()
        for name, param in self.model.named_parameters():
            update = any(key in name for key in self.trainable_substrings)
            param.requires_grad_(update)
            if update:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = self.get_trainer_cfg()
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = self.clip_loader(cfg, self.design_model_name)
        clip_model_zero_shot = self.clip_loader(cfg)

        if trainer_cfg.PREC in ["fp32", "amp"]:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype

        with torch.no_grad():
            self.text_encoder_clip = self.text_encoder_clip_cls(clip_model_zero_shot)
            text_features_clip = self.text_feature_extractor(
                cfg, classnames, clip_model_zero_shot, self.text_encoder_clip
            )
            self.text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

        self.image_encoder_clip = clip_model_zero_shot.visual

        print("Building custom CLIP")
        self.model = self.custom_clip_cls(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        self._freeze_parameters()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.image_encoder_clip.to(self.device)

        self.criterion = self.loss_cls(
            reg_weight=trainer_cfg.REG_WEIGHT,
            alpha=trainer_cfg.ALPHA,
        )

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(self.register_name, self.model, self.optim, self.sched)
        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.image_encoder_clip = nn.DataParallel(self.image_encoder_clip)

    def parse_batch_train(self, batch):
        images = batch["img"].to(self.device)
        labels = batch["label"].to(self.device)
        return images, labels

    def _forward_loss(self, image, label):
        with torch.no_grad():
            image_features_clip = self.image_encoder_clip(image.type(self.dtype))
            image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

        logits, logits_rep, logits_fusion, image_features, text_features = self.model(image)
        text_features = text_features[0:self.num_classes]
        loss = self.criterion(
            logits,
            logits_rep,
            image_features,
            text_features,
            image_features_clip,
            self.text_features_clip,
            label,
        )
        return loss, logits_fusion

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.get_trainer_cfg().PREC

        if prec == "amp":
            with autocast():
                loss, output = self._forward_loss(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, output = self._forward_loss(image, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        task = self.cfg.TASK
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        for batch in tqdm(data_loader):
            image, label = self.parse_batch_test(batch)
            logits, _, logits_fusion, _, _ = self.model(image)
            output = select_eval_output(task, dataset, sub_cls, logits, logits_fusion)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)
        return list(results.values())[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print('Note that load_model() is skipped as no pretrained model is given')
            return

        names = self.get_model_names()
        for name in names:
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(f'Model not found at "{model_path_prefix}"')

            model_path = None
            for file in os.listdir(model_path_prefix):
                if "model-best.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
            if model_path is None or not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path_prefix}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}
            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
