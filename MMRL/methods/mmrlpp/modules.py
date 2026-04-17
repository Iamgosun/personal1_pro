from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from backbones.prompt_builder import CUSTOM_TEMPLATES


class MMRLppTextEncoder(nn.Module):
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
        outputs = self.transformer([x, compound_rep_tokens_text, 0])
        x = outputs[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_index = tokenized_prompts.argmax(dim=-1)
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
    return text_encoder(embeddings.to(text_device), tokenized_prompts.to(text_device))


class ResidualAligner(nn.Module):
    def __init__(self, weight, bias, rank):
        super().__init__()
        self.weight_shape = weight.shape
        self.rank = rank
        self.A = nn.Parameter(torch.zeros(self.weight_shape[0], self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.rank, self.weight_shape[-1]))
        self.bias = nn.Parameter(bias.clone().detach()) if bias is not None else None

    def forward(self, x, weight):
        return F.linear(x, weight + (self.A @ self.B), self.bias)


class SharedResidualRepresentationAligner(nn.Module):
    def __init__(self, base_linear, num_layers, rank):
        super().__init__()
        self.weight = nn.Parameter(base_linear.weight.clone().detach())
        self.srra = nn.ModuleList([
            ResidualAligner(weight=self.weight, bias=base_linear.bias, rank=rank)
            for _ in range(num_layers)
        ])

    def forward(self, x, idx):
        return self.srra[idx](x, self.weight)


class MultiModalRepresentationLearnerPP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_rep_tokens = cfg.MMRLPP.N_REP_TOKENS
        self.dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.MMRLPP.REP_DIM
        self.rep_layers_length = len(cfg.MMRLPP.REP_LAYERS)
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

        shared_layer_r2v = nn.Linear(rep_dim, visual_dim)
        shared_layer_r2t = nn.Linear(rep_dim, text_dim)
        res_lora_dim = cfg.MMRLPP.RES_LORA_DIM
        self.srra_r2vproj = SharedResidualRepresentationAligner(shared_layer_r2v, self.rep_layers_length, res_lora_dim)
        self.srra_r2tproj = SharedResidualRepresentationAligner(shared_layer_r2t, self.rep_layers_length, res_lora_dim)

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            compound_rep_tokens_text.append(self.srra_r2tproj(rep_tokens, index).type(self.dtype))
            compound_rep_tokens_visual.append(self.srra_r2vproj(rep_tokens, index).type(self.dtype))
        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomMMRLPPModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.MMRLPP.ALPHA
        self.representation_learner = MultiModalRepresentationLearnerPP(cfg, classnames, clip_model).type(clip_model.dtype)
        self.register_buffer('tokenized_prompts', self.representation_learner.tokenized_prompts.clone())
        self.register_buffer('prompt_embeddings', self.representation_learner.prompt_embeddings.clone())
        self.image_encoder = clip_model.visual
        self.text_encoder = MMRLppTextEncoder(clip_model)
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


class MMRLppLoss(nn.Module):
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
