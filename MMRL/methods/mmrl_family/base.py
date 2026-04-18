from __future__ import annotations

import torch

from backbones.clip_loader import load_mmrl_clip_to_cpu
from backbones.freeze import freeze_all_but
from core.types import MethodOutputs
from methods.base import BaseMethod
from .modules import (
    CLIPTextEncoderPlain,
    MMRLFamilyModel,
    build_zero_shot_text_features,
)


class BaseMMRLFamilyMethod(BaseMethod):
    method_name = "BaseMMRLFamily"
    cfg_section_name = "MMRL"
    clip_loader_name = "MMRL"
    model_cls = MMRLFamilyModel
    loss_adapter_cls = None
    trainable_substrings = ("representation_learner", "image_encoder.proj_rep")

    def build_loss(self, method_cfg):
        if self.loss_adapter_cls is None:
            raise NotImplementedError(
                f"{self.method_name} must define `loss_adapter_cls` or override `build_loss`."
            )
        return self.loss_adapter_cls(
            reg_weight=float(method_cfg.REG_WEIGHT),
            alpha=float(method_cfg.ALPHA),
        )

    def get_precision(self) -> str:
        return getattr(self.cfg, self.cfg_section_name).PREC

    def build(self):
        cfg = self.cfg
        method_cfg = getattr(cfg, self.cfg_section_name)
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        self.method_cfg = method_cfg

        clip_model = load_mmrl_clip_to_cpu(cfg, self.clip_loader_name)
        clip_model_zero_shot = load_mmrl_clip_to_cpu(cfg, "CLIP")

        if method_cfg.PREC in {"fp32", "amp"}:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype

        self.text_encoder_clip = CLIPTextEncoderPlain(clip_model_zero_shot).to(
            self.device
        )
        with torch.no_grad():
            text_features_clip = build_zero_shot_text_features(
                cfg,
                classnames,
                clip_model_zero_shot,
                self.text_encoder_clip,
            )
            self.text_features_clip = (
                text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
            ).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)
        self.model = self.model_cls(cfg, method_cfg, classnames, clip_model).to(
            self.device
        )

        enabled = freeze_all_but(self.model, list(self.trainable_substrings))
        print(f"[{self.method_name}] trainable params: {enabled}")

        self.loss = self.build_loss(method_cfg)
        return self

    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = img_ref / img_ref.norm(dim=-1, keepdim=True)

        logits, logits_rep, logits_fusion, image_features, text_features = self.model(
            image
        )
        text_features = text_features[: self.num_classes]

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={"rep": logits_rep, "fusion": logits_fusion},
            features={
                "img": image_features,
                "text": text_features,
                "img_ref": img_ref,
                "text_ref": self.text_features_clip,
            },
        )

    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        logits, logits_rep, logits_fusion, image_features, text_features = self.model(
            image
        )
        text_features = text_features[: self.num_classes]

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={"rep": logits_rep, "fusion": logits_fusion},
            features={"img": image_features, "text": text_features},
        )

    def select_train_logits(self, outputs):
        return outputs.aux_logits.get("fusion", outputs.logits)

    def select_eval_logits(self, outputs, eval_ctx):
        logits = outputs.logits
        logits_fusion = outputs.aux_logits.get("fusion")

        if logits_fusion is None:
            return logits

        protocol = eval_ctx.protocol
        dataset = eval_ctx.dataset_name
        sub_cls = eval_ctx.subsample_classes or "all"

        if protocol == "B2N":
            if sub_cls == "base":
                return logits_fusion
            return logits

        if protocol == "FS":
            return logits_fusion

        if protocol == "CD":
            if dataset == "ImageNet":
                return logits_fusion
            return logits

        return logits