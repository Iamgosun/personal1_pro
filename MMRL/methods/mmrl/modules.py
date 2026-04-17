from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from backbones.prompt_builder import CUSTOM_TEMPLATES


def _get_clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class MMRLTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):
        prompts = prompts.to(self.positional_embedding.device)
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        eot_index = tokenized_prompts.argmax(dim=-1)
        outputs = self.transformer([x, compound_rep_tokens_text, 0, eot_index])
        x = outputs[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0], device=x.device), eot_index + n_rep_tokens] @ self.text_projection


class CLIPTextEncoderPlain(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        prompts = prompts.to(self.positional_embedding.device)
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_index = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot_index] @ self.text_projection


@torch.no_grad()
def build_zero_shot_text_features(cfg, classnames, clip_model, text_encoder):
    text_device = next(text_encoder.parameters()).device
    token_device = clip_model.token_embedding.weight.device
    text_encoder = text_encoder.to(text_device)
    template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

    tokenized_prompts = []
    for text in classnames:
        tokens = clip.tokenize(template.format(text.replace('_', ' ')))
        tokenized_prompts.append(tokens.to(token_device))
    tokenized_prompts = torch.cat(tokenized_prompts, dim=0)
    embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
    text_embeddings = text_encoder(embeddings.to(text_device), tokenized_prompts.to(text_device))
    return text_embeddings


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_rep_tokens = cfg.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.MMRL.REP_DIM
        self.rep_layers_length = len(cfg.MMRL.REP_LAYERS)
        assert cfg_imsize == clip_imsize, f'cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})'

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        token_device = clip_model.token_embedding.weight.device
        tokenized_prompts = [clip.tokenize(template.format(text.replace('_', ' '))).to(token_device) for text in classnames]
        tokenized_prompts = torch.cat(tokenized_prompts, dim=0)
        self.register_buffer('tokenized_prompts', tokenized_prompts)
        with torch.no_grad():
            prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)
        self.register_buffer('prompt_embeddings', prompt_embeddings)

        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)
        self.compound_rep_tokens_r2vproj = _get_clones(nn.Linear(rep_dim, visual_dim), self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(nn.Linear(rep_dim, text_dim), self.rep_layers_length)

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            compound_rep_tokens_text.append(self.compound_rep_tokens_r2tproj[index](rep_tokens).type(self.dtype))
            compound_rep_tokens_visual.append(self.compound_rep_tokens_r2vproj[index](rep_tokens).type(self.dtype))
        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomMMRLModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.MMRL.ALPHA
        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(clip_model.dtype)
        self.register_buffer('tokenized_prompts', self.representation_learner.tokenized_prompts.clone())
        self.register_buffer('prompt_embeddings', self.representation_learner.prompt_embeddings.clone())
        self.image_encoder = clip_model.visual
        self.text_encoder = MMRLTextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def forward(self, image):
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
        else:
            if self.text_features_for_inference is None:
                rep_text, rep_visual = self.representation_learner()
                self.compound_rep_tokens_text_for_inference = rep_text
                self.compound_rep_tokens_visual_for_inference = rep_visual
                self.text_features_for_inference = self.text_encoder(
                    self.prompt_embeddings,
                    self.tokenized_prompts,
                    self.compound_rep_tokens_text_for_inference,
                )
            compound_rep_tokens_visual = self.compound_rep_tokens_visual_for_inference
            text_features = self.text_features_for_inference

        image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1 - self.alpha) * logits_rep
        return logits, logits_rep, logits_fusion, image_features, text_features


class MMRLLoss(nn.Module):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super().__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def forward(self, logits, logits_rep, image_features, text_features, image_features_clip, text_features_clip, label):
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)
        cossim_reg_img = 1 - torch.mean(F.cosine_similarity(image_features, image_features_clip, dim=1))
        cossim_reg_text = 1 - torch.mean(F.cosine_similarity(text_features, text_features_clip, dim=1))
        return self.alpha * xe_loss1 + (1 - self.alpha) * xe_loss2 + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text
