from __future__ import annotations

import torch

from backbones.freeze import freeze_all_but
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.base import BaseMethod
from .loss import MMRLppLossAdapter


@METHOD_REGISTRY.register('MMRLpp')
@METHOD_REGISTRY.register('MMRLPP')
class MMRLppMethod(BaseMethod):
    method_name = 'MMRLpp'

    def build(self):
        from trainers import mmrlpp as legacy

        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = legacy.load_clip_to_cpu(cfg, 'MMRLpp')
        clip_model_zero_shot = legacy.load_clip_to_cpu(cfg)
        if cfg.MMRLPP.PREC in {'fp32', 'amp'}:
            clip_model.float()
            clip_model_zero_shot.float()
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

        with torch.no_grad():
            self.text_encoder_clip = legacy.TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = legacy._get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip)
            self.text_features_clip = (text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)
        self.model = legacy.CustomCLIP(cfg, classnames, clip_model).to(self.device)
        enabled = freeze_all_but(self.model, ['representation_learner', 'image_encoder.proj_rep', 'image_encoder.A', 'image_encoder.B'])
        print(f'[MMRLppMethod] trainable params: {enabled}')
        self.loss = MMRLppLossAdapter(legacy.MMRLpp_Loss(reg_weight=cfg.MMRLPP.REG_WEIGHT, alpha=cfg.MMRLPP.ALPHA))
        return self

    def forward_train(self, batch):
        image = batch['img'].to(self.device)
        label = batch['label'].to(self.device)
        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = img_ref / img_ref.norm(dim=-1, keepdim=True)
        logits, logits_rep, logits_fusion, image_features, text_features = self.model(image)
        text_features = text_features[0:self.num_classes]
        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={'rep': logits_rep, 'fusion': logits_fusion},
            features={
                'img': image_features,
                'text': text_features,
                'img_ref': img_ref,
                'text_ref': self.text_features_clip,
            },
        )

    def forward_eval(self, batch, eval_ctx):
        image = batch['img'].to(self.device)
        label = batch.get('label')
        if label is not None:
            label = label.to(self.device)
        logits, logits_rep, logits_fusion, image_features, text_features = self.model(image)
        text_features = text_features[0:self.num_classes]
        return MethodOutputs(logits=logits, labels=label, aux_logits={'rep': logits_rep, 'fusion': logits_fusion}, features={'img': image_features, 'text': text_features})
