import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.shared.base_mmrl_trainer import BaseMMRLTrainer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
}


def load_clip_to_cpu(cfg, model_name="CLIP"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "model": model_name,
        "rep_tokens_layers": cfg.TRAINER.MMRLpp.REP_LAYERS,
        "n_rep_tokens": cfg.TRAINER.MMRLpp.N_REP_TOKENS,
        "proj_lora_dim": cfg.TRAINER.MMRLpp.PROJ_LORA_DIM,
        "beta": cfg.TRAINER.MMRLpp.BETA,
    }
    model = clip.build_model_MMRLpp(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder_MMRLpp(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):
        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_rep_tokens_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eot_index = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection
        return x


class TextEncoder_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    text_encoder = text_encoder.cuda()
    template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

    with torch.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))
            tokenized_prompts.append(tokens.to(device))
        tokenized_prompts = torch.cat(tokenized_prompts)
        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
        text_embeddings = text_encoder(embeddings.cuda(), tokenized_prompts.cuda())

    text_encoder = text_encoder.to(device)
    return text_embeddings


class Residual_Aligner(nn.Module):
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


class Shared_Residual_Representation_Aligner(nn.Module):
    def __init__(self, base_linear, num_layers, rank):
        super().__init__()
        self.weight = nn.Parameter(base_linear.weight.clone().detach())
        self.srra = nn.ModuleList([
            Residual_Aligner(weight=self.weight, bias=base_linear.bias, rank=rank)
            for _ in range(num_layers)
        ])

    def forward(self, x, idx):
        return self.srra[idx](x, self.weight)


class MultiModalRepresentationLearner_pp(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_rep_tokens = cfg.TRAINER.MMRLpp.N_REP_TOKENS
        self.dtype = clip_model.dtype
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRLpp.REP_DIM
        self.rep_layers_length = len(cfg.TRAINER.MMRLpp.REP_LAYERS)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        tokenized_prompts = [clip.tokenize(template.format(text.replace('_', ' '))) for text in classnames]
        self.tokenized_prompts = torch.cat(tokenized_prompts)
        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)

        shared_layer_r2v = nn.Linear(rep_dim, visual_dim)
        shared_layer_r2t = nn.Linear(rep_dim, text_dim)
        res_lora_dim = cfg.TRAINER.MMRLpp.RES_LORA_DIM
        self.srra_r2vproj = Shared_Residual_Representation_Aligner(shared_layer_r2v, self.rep_layers_length, res_lora_dim)
        self.srra_r2tproj = Shared_Residual_Representation_Aligner(shared_layer_r2t, self.rep_layers_length, res_lora_dim)

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            compound_rep_tokens_text.append(self.srra_r2tproj(rep_tokens, index).type(self.dtype))
            compound_rep_tokens_visual.append(self.srra_r2vproj(rep_tokens, index).type(self.dtype))
        return compound_rep_tokens_text, compound_rep_tokens_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRLpp.ALPHA
        self.representation_learner = MultiModalRepresentationLearner_pp(cfg, classnames, clip_model).type(clip_model.dtype)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRLpp(clip_model)
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
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference = self.representation_learner()
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


class MMRLpp_Loss(_Loss):
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


@TRAINER_REGISTRY.register()
class MMRLpp(BaseMMRLTrainer):
    trainer_cfg_name = "MMRLpp"
    clip_loader = staticmethod(load_clip_to_cpu)
    custom_clip_cls = CustomCLIP
    text_encoder_clip_cls = TextEncoder_CLIP
    text_feature_extractor = staticmethod(_get_text_base_features_zero_shot)
    loss_cls = MMRLpp_Loss
    register_name = "MultiModalRepresentationLearner"
    trainable_substrings = ("representation_learner", "image_encoder.proj_rep", "image_encoder.A", "image_encoder.B")
    design_model_name = "MMRLpp"
