from __future__ import annotations

import contextlib
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from methods.mmrl_family.modules import MMRLTextEncoder, MMRLFamilyRepresentationLearner


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class VisualContextModulator(nn.Module):
    """
    Generate a per-instance channel gate g(x) in the representation space.

    The last linear layer is zero-initialized so that at initialization:
        g(x) = 0
        R_tilde(x) = R

    This makes VCRM start exactly from the original MMRL behavior.
    """

    def __init__(self, visual_dim: int, rep_dim: int, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rep_dim),
        )

        # Zero-init only the last layer. Do not zero-init the whole MLP.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        return self.net(z_v)


class VCRMRepresentationLearner(MMRLFamilyRepresentationLearner):
    """
    MMRL representation learner with visual-context residual modulation.

    Original MMRL:
        R_t^{(i)} = W_t^{(i)}(R)

    VCRM:
        g^{(i)}(x) = phi^{(i)}(z_v)
        R_tilde^{(i)}(x) = R * (1 + eta * g^{(i)}(x))
        R_t^{(i)}(x) = W_t^{(i)}(R_tilde^{(i)}(x))

    Visual-side representation tokens are kept static, matching your current
    design that only conditions the text-side realization of R.
    """

    def __init__(self, cfg, method_cfg, classnames, clip_model):
        super().__init__(cfg, method_cfg, classnames, clip_model)

        rep_dim = int(method_cfg.REP_DIM)
        visual_dim = clip_model.visual.ln_post.weight.shape[0]

        hidden_dim = int(getattr(method_cfg, "VCRM_HIDDEN_DIM", rep_dim))
        self.vcrm_eta = float(getattr(method_cfg, "VCRM_ETA", 0.1))

        base_modulator = VisualContextModulator(
            visual_dim=visual_dim,
            rep_dim=rep_dim,
            hidden_dim=hidden_dim,
        )

        self.visual_context_modulators = _get_clones(
            base_modulator,
            self.rep_layers_length,
        )

        self.last_modulation_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        z_v: Optional[torch.Tensor] = None,
        disable_vcrm: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            z_v:
                None or [B, d_v].
                If None, this method behaves like original MMRL.
            disable_vcrm:
                If True, force static MMRL tokens.

        Returns:
            compound_rep_tokens_text:
                If VCRM disabled: list of [K, d_t].
                If VCRM enabled:  list of [B, K, d_t].
            compound_rep_tokens_visual:
                Always list of [K, d_v].
        """
        use_vcrm = (z_v is not None) and (not disable_vcrm)

        compound_rep_tokens_text = []
        compound_rep_tokens_visual = []
        modulation_losses = []

        base_rep = self.compound_rep_tokens

        for index in range(self.rep_layers_length):
            # Visual-side tokens stay static MMRL tokens.
            visual_tokens = self.compound_rep_tokens_r2vproj[index](base_rep)
            compound_rep_tokens_visual.append(visual_tokens.type(self.dtype))

            if use_vcrm:
                gate = self.visual_context_modulators[index](z_v)
                gate = gate.type(base_rep.dtype)

                # base_rep: [K, d_r]
                # gate:     [B, d_r]
                # rep:      [B, K, d_r]
                conditioned_rep = base_rep.unsqueeze(0) * (
                    1.0 + self.vcrm_eta * gate.unsqueeze(1)
                )

                text_tokens = self.compound_rep_tokens_r2tproj[index](
                    conditioned_rep
                )
                compound_rep_tokens_text.append(text_tokens.type(self.dtype))

                modulation_losses.append(gate.float().pow(2).mean())
            else:
                text_tokens = self.compound_rep_tokens_r2tproj[index](base_rep)
                compound_rep_tokens_text.append(text_tokens.type(self.dtype))

        if modulation_losses:
            self.last_modulation_loss = torch.stack(modulation_losses).mean()
        else:
            self.last_modulation_loss = None

        return compound_rep_tokens_text, compound_rep_tokens_visual


class VCRMMMRLModel(nn.Module):
    """
    VCRM-MMRL model.

    Training / base evaluation:
        use image-conditioned text prototypes w_c(x)

    Novel / new-dataset evaluation:
        caller can disable VCRM and use zero-shot prototypes externally.
    """

    def __init__(self, cfg, method_cfg, classnames, clip_model):
        super().__init__()

        self.alpha = float(method_cfg.ALPHA)
        self.vcrm_context_layer = int(getattr(method_cfg, "VCRM_CONTEXT_LAYER", 3))
        self.vcrm_detach_context = bool(getattr(method_cfg, "VCRM_DETACH_CONTEXT", True))

        self.representation_learner = VCRMRepresentationLearner(
            cfg,
            method_cfg,
            classnames,
            clip_model,
        ).type(clip_model.dtype)

        self.register_buffer(
            "tokenized_prompts",
            self.representation_learner.tokenized_prompts.clone(),
        )
        self.register_buffer(
            "prompt_embeddings",
            self.representation_learner.prompt_embeddings.clone(),
        )

        self.image_encoder = clip_model.visual
        self.text_encoder = MMRLTextEncoder(clip_model)
        self.dtype = clip_model.dtype

        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def clear_inference_cache(self):
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _extract_visual_context(
        self,
        image: torch.Tensor,
        layer_idx: int,
        detach: bool = True,
    ) -> torch.Tensor:
        """
        Extract z_v from the frozen ViT image encoder after `layer_idx` blocks.

        This intentionally does not modify clip/model_mmrl.py.

        Args:
            image:
                [B, 3, H, W]
            layer_idx:
                1-based transformer block index.
                Example: 3 means after visual transformer block 3.
            detach:
                If True, stop gradient from the modulator to the image encoder.

        Returns:
            z_v:
                [B, d_v], global-average-pooled patch tokens.
        """
        if layer_idx < 0:
            raise ValueError(f"VCRM_CONTEXT_LAYER must be >= 0, got {layer_idx}")

        grad_ctx = torch.no_grad() if detach else contextlib.nullcontext()

        with grad_ctx:
            x = image.type(self.image_encoder.conv1.weight.dtype)

            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            cls = self.image_encoder.class_embedding.to(x.dtype)
            cls = cls + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([cls, x], dim=1)
            x = x + self.image_encoder.positional_embedding.to(x.dtype)
            x = self.image_encoder.ln_pre(x)

            # [B, N, D] -> [N, B, D]
            x = x.permute(1, 0, 2)

            # model_mmrl.ResidualAttentionBlock expects [x, compound_tokens, counter]
            # under model == "MMRL". Empty token list means no rep-token insertion.
            state = [x, [], 0]

            max_layer = min(
                int(layer_idx),
                len(self.image_encoder.transformer.resblocks),
            )

            for idx, block in enumerate(self.image_encoder.transformer.resblocks):
                if idx >= max_layer:
                    break
                state = block(state)

            x = state[0]
            x = x.permute(1, 0, 2)

            patch_tokens = x[:, 1:, :]
            z_v = patch_tokens.mean(dim=1)

        if detach:
            z_v = z_v.detach()

        return z_v

    def _encode_dynamic_text_features(
        self,
        compound_rep_tokens_text: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode image-conditioned text prototypes.

        Args:
            compound_rep_tokens_text:
                list of [B, K, d_t]

        Returns:
            text_features:
                [B, C, d]
        """
        batch_size = compound_rep_tokens_text[0].shape[0]
        text_features_all = []

        for b in range(batch_size):
            tokens_b = [tokens[b] for tokens in compound_rep_tokens_text]
            text_features_b = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                tokens_b,
            )
            text_features_all.append(text_features_b)

        return torch.stack(text_features_all, dim=0)

    def _encode_static_text_features(self):
        """
        Encode static MMRL text features and cache them for inference.
        """
        if self.text_features_for_inference is None:
            rep_text, rep_visual = self.representation_learner(disable_vcrm=True)

            self.compound_rep_tokens_text_for_inference = rep_text
            self.compound_rep_tokens_visual_for_inference = rep_visual

            self.text_features_for_inference = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                rep_text,
            )

        return (
            self.text_features_for_inference,
            self.compound_rep_tokens_visual_for_inference,
        )

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True)

    @staticmethod
    def _compute_logits(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features:
                [B, d]
            text_features:
                [C, d] or [B, C, d]
        """
        if text_features.dim() == 3:
            return 100.0 * torch.einsum("bd,bcd->bc", image_features, text_features)

        return 100.0 * image_features @ text_features.t()

    def forward(
        self,
        image: torch.Tensor,
        use_conditioned_text: bool = True,
    ):
        image = image.type(self.dtype)

        if use_conditioned_text:
            z_v = self._extract_visual_context(
                image=image,
                layer_idx=self.vcrm_context_layer,
                detach=self.vcrm_detach_context,
            )

            compound_rep_tokens_text, compound_rep_tokens_visual = (
                self.representation_learner(
                    z_v=z_v,
                    disable_vcrm=False,
                )
            )

            text_features = self._encode_dynamic_text_features(
                compound_rep_tokens_text
            )
        else:
            text_features, compound_rep_tokens_visual = self._encode_static_text_features()

        image_features, image_features_rep = self.image_encoder(
            [image, compound_rep_tokens_visual]
        )

        image_features = self._normalize(image_features)
        image_features_rep = self._normalize(image_features_rep)
        text_features = self._normalize(text_features)

        logits = self._compute_logits(image_features, text_features)
        logits_rep = self._compute_logits(image_features_rep, text_features)
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features