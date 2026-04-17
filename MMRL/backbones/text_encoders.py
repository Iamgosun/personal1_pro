from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from clip import clip
from backbones.prompt_builder import resolve_templates


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @property
    def device(self):
        return self.positional_embedding.device

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        prompts = prompts.to(self.device)
        tokenized_prompts = tokenized_prompts.to(self.device)
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        return x[torch.arange(x.shape[0], device=x.device), eot_indices] @ self.text_projection


def build_base_text_features(cfg, classnames: List[str], clip_model, text_encoder: CLIPTextEncoder, pretrained_projection: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    text_encoder = text_encoder.to(text_encoder.device)
    if pretrained_projection is not None:
        pretrained = torch.load(pretrained_projection, map_location='cpu')
        state_dict = text_encoder.state_dict()
        state_dict['text_projection'] = pretrained['state_dict']['weight'].t().to(text_encoder.device)
        text_encoder.load_state_dict(state_dict)

    templates = resolve_templates(cfg.DATASET.NAME)
    with torch.no_grad():
        all_embeddings = []
        token_embedding_device = clip_model.token_embedding.weight.device
        for name in classnames:
            tokens = clip.tokenize([template.format(name) for template in templates]).to(token_embedding_device)
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            prototype = text_encoder(embeddings.to(text_encoder.device), tokens.to(text_encoder.device))
            all_embeddings.append(prototype)
    all_embeddings = torch.stack(all_embeddings)
    return all_embeddings.mean(1), all_embeddings
