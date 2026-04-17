from __future__ import annotations

import torch

from clip import clip


def load_raw_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    return clip.build_model(state_dict or model.state_dict())


def load_mmrl_clip_to_cpu(cfg, model_name='CLIP'):
    from trainers.mmrl import load_clip_to_cpu
    return load_clip_to_cpu(cfg, model_name)


def load_mmrlpp_clip_to_cpu(cfg, model_name='CLIP'):
    from trainers.mmrlpp import load_clip_to_cpu
    return load_clip_to_cpu(cfg, model_name)
