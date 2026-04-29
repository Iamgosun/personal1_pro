from __future__ import annotations

import torch

from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from methods.mmrl_family.base import BaseMMRLFamilyMethod

from .loss import VCRMMMRLLossAdapter
from .modules import VCRMMMRLModel


@METHOD_REGISTRY.register("VCRMMMRL")
class VCRMMMRLMethod(BaseMMRLFamilyMethod):
    """
    Visual-Context-Regulated MMRL.

    Train:
        Use conditioned text prototypes w_c(x).

    Eval:
        - B2N base / FS / ImageNet source:
            use conditioned text prototypes and fusion logits.
        - B2N novel / cross-dataset target / domain-shift target:
            disable VCRM and use zero-shot text prototypes w_c^0 with f_c.
    """

    method_name = "VCRMMMRL"
    cfg_section_name = "VCRM_MMRL"
    clip_loader_name = "MMRL"
    model_cls = VCRMMMRLModel
    loss_adapter_cls = VCRMMMRLLossAdapter

    # Let BaseMMRLFamilyMethod freeze everything first.
    # We then manually enable the exact parameter set.
    trainable_substrings = ()

    def build_loss(self, method_cfg):
        return self.loss_adapter_cls(
            reg_weight=float(method_cfg.REG_WEIGHT),
            alpha=float(method_cfg.ALPHA),
            mod_weight=float(getattr(method_cfg, "VCRM_MOD_WEIGHT", 0.0)),
        )

    def build(self):
        super().build()

        enabled = set()

        for name, param in self.model.named_parameters():
            trainable = (
                name == "representation_learner.compound_rep_tokens"
                or name.startswith(
                    "representation_learner.compound_rep_tokens_r2tproj."
                )
                or name.startswith(
                    "representation_learner.visual_context_modulators."
                )
                or name == "image_encoder.proj_rep"
            )

            param.requires_grad_(trainable)

            if trainable:
                enabled.add(name)

        print(f"[{self.method_name}] exact trainable params: {sorted(enabled)}")
        return self

    def _use_conditioned_text_for_eval(self, eval_ctx) -> bool:
        protocol = eval_ctx.protocol
        dataset = eval_ctx.dataset_name
        sub_cls = eval_ctx.subsample_classes or "all"

        if protocol == "B2N":
            return sub_cls == "base"

        if protocol == "FS":
            return True

        if protocol == "CD":
            return dataset == "ImageNet"

        return False

    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = img_ref / img_ref.norm(dim=-1, keepdim=True)

        logits, logits_rep, logits_fusion, image_features, text_features = self.model(
            image,
            use_conditioned_text=True,
        )

        if text_features.dim() == 3:
            text_features_for_reg = text_features[:, : self.num_classes, :]
        else:
            text_features_for_reg = text_features[: self.num_classes]

        mod_loss = self.model.representation_learner.last_modulation_loss

        losses = {}
        if mod_loss is not None:
            losses["mod_loss"] = mod_loss

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": image_features,
                "text": text_features_for_reg,
                "img_ref": img_ref,
                "text_ref": self.text_features_clip,
            },
            losses=losses,
        )

    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")

        if label is not None:
            label = label.to(self.device)

        use_conditioned_text = self._use_conditioned_text_for_eval(eval_ctx)

        logits, logits_rep, logits_fusion, image_features, text_features = self.model(
            image,
            use_conditioned_text=use_conditioned_text,
        )

        if use_conditioned_text:
            if text_features.dim() == 3:
                text_features_out = text_features[:, : self.num_classes, :]
            else:
                text_features_out = text_features[: self.num_classes]

            return MethodOutputs(
                logits=logits,
                labels=label,
                aux_logits={
                    "rep": logits_rep,
                    "fusion": logits_fusion,
                },
                features={
                    "img": image_features,
                    "text": text_features_out,
                },
            )

        # Novel / new-dataset path:
        # implement p(y=c|x)=p(y=c|f_c,w_c^0)
        logits_zs = 100.0 * image_features @ self.text_features_clip.t()

        return MethodOutputs(
            logits=logits_zs,
            labels=label,
            aux_logits={
                "rep": logits_zs,
                "fusion": logits_zs,
            },
            features={
                "img": image_features,
                "text": self.text_features_clip,
            },
        )

    def select_eval_logits(self, outputs, eval_ctx):
        if self._use_conditioned_text_for_eval(eval_ctx):
            return outputs.aux_logits.get("fusion", outputs.logits)

        return outputs.logits