from __future__ import annotations

import torch

from clip import clip


def _load_state_dict_from_backbone(backbone_name: str):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    return state_dict or model.state_dict()


def load_raw_clip_to_cpu(cfg):
    state_dict = _load_state_dict_from_backbone(cfg.MODEL.BACKBONE.NAME)
    return clip.build_model(state_dict)


def load_mmrl_clip_to_cpu(cfg, model_name: str = 'CLIP'):
    state_dict = _load_state_dict_from_backbone(cfg.MODEL.BACKBONE.NAME)
    design_details = {
        'model': model_name,
        'rep_tokens_layers': list(cfg.MMRL.REP_LAYERS),
        'n_rep_tokens': int(cfg.MMRL.N_REP_TOKENS),
    }
    return clip.build_model_MMRL(state_dict, design_details)


def load_mmrlpp_clip_to_cpu(cfg, model_name: str = 'CLIP'):
    state_dict = _load_state_dict_from_backbone(cfg.MODEL.BACKBONE.NAME)
    design_details = {
        'model': model_name,
        'rep_tokens_layers': list(cfg.MMRLPP.REP_LAYERS),
        'n_rep_tokens': int(cfg.MMRLPP.N_REP_TOKENS),
        'proj_lora_dim': int(cfg.MMRLPP.PROJ_LORA_DIM),
        'beta': float(cfg.MMRLPP.BETA),
    }
    return clip.build_model_MMRLpp(state_dict, design_details)
