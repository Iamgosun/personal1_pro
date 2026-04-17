import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


class TextEncoder(nn.Module):
    """Thin wrapper around CLIP text branch with explicit device handling."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @property
    def device(self) -> torch.device:
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


def _resolve_templates(dataset_name: str) -> List[str]:
    templates: List[str] = []
    if dataset_name == "ImageNet":
        templates.extend(IMAGENET_TEMPLATES_SELECT)
    templates.append(CUSTOM_TEMPLATES[dataset_name])
    return templates


def get_base_text_features(
    cfg,
    classnames: List[str],
    clip_model,
    text_encoder: TextEncoder,
    pretrained_projection: Optional[str] = None,
    debug_save_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build zero-shot class text features for clip_adapters."""

    device = text_encoder.device
    text_encoder = text_encoder.to(device)

    if pretrained_projection is not None:
        pretrained_text_projection = torch.load(pretrained_projection, map_location="cpu")
        state_dict = text_encoder.state_dict()
        state_dict["text_projection"] = pretrained_text_projection["state_dict"]["weight"].t().to(device)
        text_encoder.load_state_dict(state_dict)
        print(">> Pretrained text encoder loaded!")
        params = (
            pretrained_text_projection["state_dict"]["weight"].size(0)
            * pretrained_text_projection["state_dict"]["weight"].size(1)
        )
        print(">> Text projection parameters:", params)

    dataset = cfg.DATASET.NAME
    templates = _resolve_templates(dataset)

    with torch.no_grad():
        text_embeddings = []
        token_embedding_device = clip_model.token_embedding.weight.device
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in templates]).to(token_embedding_device)
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            prototype = text_encoder(embeddings.to(device), tokens.to(device))
            text_embeddings.append(prototype)

    text_embeddings = torch.stack(text_embeddings)
    text_embeddings_avg = text_embeddings.mean(1)

    if debug_save_path:
        save_dir = os.path.dirname(debug_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "base_text_features": text_embeddings_avg.cpu(),
                "classnames": classnames,
                "dataset": dataset,
            },
            debug_save_path,
        )

    return text_embeddings_avg, text_embeddings
