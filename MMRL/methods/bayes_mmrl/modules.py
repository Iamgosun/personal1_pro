from __future__ import annotations

import copy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from backbones.prompt_builder import CUSTOM_TEMPLATES
from methods.mmrl.modules import (
    CLIPTextEncoderPlain,
    MMRLTextEncoder,
    build_zero_shot_text_features,
)


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class BayesianMultiModalRepresentationLearner(nn.Module):
    """
    Bayesian version of the shared representation learner.

    Only the shared token matrix R is randomized:
        q(R) = N(mu, sigma^2 I)

    where sigma is a single global scalar (shared across all entries),
    following your latest requirement of using a unified covariance scale.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        bayes_cfg = cfg.BAYES_MMRL

        n_rep_tokens = bayes_cfg.N_REP_TOKENS
        rep_dim = bayes_cfg.REP_DIM
        self.dtype = clip_model.dtype
        self.rep_layers_length = len(bayes_cfg.REP_LAYERS)

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, (
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        )

        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        token_device = clip_model.token_embedding.weight.device
        tokenized_prompts = [
            clip.tokenize(template.format(text.replace("_", " "))).to(token_device)
            for text in classnames
        ]
        tokenized_prompts = torch.cat(tokenized_prompts, dim=0)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(
                self.dtype
            )
        self.register_buffer("prompt_embeddings", prompt_embeddings)

        # Variational parameters for the shared representation R.
        # Keep them in float32 for numerical stability even if the backbone uses fp16.
        self.posterior_mean = nn.Parameter(
            torch.empty(n_rep_tokens, rep_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.posterior_mean, std=float(bayes_cfg.PRIOR_STD))

        self.posterior_rho = nn.Parameter(
            torch.tensor(float(bayes_cfg.POSTERIOR_RHO_INIT), dtype=torch.float32)
        )

        # Deterministic trainable projection layers, unchanged from original MMRL.
        self.compound_rep_tokens_r2vproj = _get_clones(
            nn.Linear(rep_dim, visual_dim), self.rep_layers_length
        )
        self.compound_rep_tokens_r2tproj = _get_clones(
            nn.Linear(rep_dim, text_dim), self.rep_layers_length
        )

        self.prior_std = float(bayes_cfg.PRIOR_STD)
        self.min_sigma = 1e-6

    def posterior_sigma(self) -> torch.Tensor:
        return F.softplus(self.posterior_rho.float()) + self.min_sigma

    def kl_divergence(self) -> torch.Tensor:
        """
        KL[q(r) || p(r)] with:
            q(r) = N(mu, sigma^2 I)
            p(r) = N(0, sigma_p^2 I)
        where sigma is a single scalar and mu is a full matrix.
        """
        sigma = self.posterior_sigma()
        prior_var = sigma.new_tensor(self.prior_std ** 2)
        sigma2 = sigma.pow(2)

        d = self.posterior_mean.numel()
        mu_sq_sum = self.posterior_mean.float().pow(2).sum()

        return 0.5 * (
            d * sigma2 / prior_var
            + mu_sq_sum / prior_var
            - d
            - d * torch.log(sigma2 / prior_var)
        )

    def _project_rep_tokens(
        self, rep_tokens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        compound_rep_tokens_text: List[torch.Tensor] = []
        compound_rep_tokens_visual: List[torch.Tensor] = []

        rep_tokens = rep_tokens.to(self.compound_rep_tokens_r2tproj[0].weight.dtype)

        for index in range(self.rep_layers_length):
            rep_text = self.compound_rep_tokens_r2tproj[index](rep_tokens).type(self.dtype)
            rep_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens).type(self.dtype)
            compound_rep_tokens_text.append(rep_text)
            compound_rep_tokens_visual.append(rep_visual)

        return compound_rep_tokens_text, compound_rep_tokens_visual

    def project_mean_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._project_rep_tokens(self.posterior_mean)

    def project_sample_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        mu = self.posterior_mean.float()
        sigma = self.posterior_sigma()
        eps = torch.randn_like(mu)
        sampled_rep_tokens = mu + sigma * eps
        return self._project_rep_tokens(sampled_rep_tokens)

    def forward(self, use_posterior_mean: bool = False):
        if use_posterior_mean:
            return self.project_mean_tokens()
        return self.project_sample_tokens()


class BayesianCustomMMRLModel(nn.Module):
    """
    Bayesian MMRL model:
    - training: Monte Carlo samples over shared representation R
    - evaluation: posterior mean or MC averaging
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        bayes_cfg = cfg.BAYES_MMRL

        self.alpha = bayes_cfg.ALPHA
        self.eval_use_posterior_mean = bool(bayes_cfg.EVAL_USE_POSTERIOR_MEAN)
        self.n_mc_test = max(1, int(bayes_cfg.N_MC_TEST))

        # Do NOT cast the representation learner wholesale to clip_model.dtype.
        # We keep variational params in fp32 for stability.
        self.representation_learner = BayesianMultiModalRepresentationLearner(
            cfg, classnames, clip_model
        )

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

        self._cached_text_features = None
        self._cached_rep_visual = None
        self._cached_mode = None

    def clear_inference_cache(self):
        self._cached_text_features = None
        self._cached_rep_visual = None
        self._cached_mode = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _encode_with_tokens(
        self,
        image: torch.Tensor,
        compound_rep_tokens_text: Sequence[torch.Tensor],
        compound_rep_tokens_visual: Sequence[torch.Tensor],
    ):
        text_features = self.text_encoder(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features, image_features_rep = self.image_encoder(
            [image.type(self.dtype), list(compound_rep_tokens_visual)]
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _forward_with_cached_text(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        text_features: torch.Tensor,
    ):
        image_features, image_features_rep = self.image_encoder(
            [image.type(self.dtype), list(compound_rep_tokens_visual)]
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _aggregate_outputs(self, sample_outputs):
        logits = torch.stack([out[0] for out in sample_outputs], dim=0).mean(dim=0)
        logits_rep = torch.stack([out[1] for out in sample_outputs], dim=0).mean(dim=0)
        logits_fusion = torch.stack([out[2] for out in sample_outputs], dim=0).mean(dim=0)
        image_features = torch.stack([out[3] for out in sample_outputs], dim=0).mean(dim=0)
        text_features = torch.stack([out[4] for out in sample_outputs], dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return logits, logits_rep, logits_fusion, image_features, text_features

    @torch.no_grad()
    def _get_cached_mean_eval_state(self):
        cache_mode = "posterior_mean"

        if (
            self._cached_text_features is None
            or self._cached_rep_visual is None
            or self._cached_mode != cache_mode
        ):
            rep_text, rep_visual = self.representation_learner.project_mean_tokens()
            text_features = self.text_encoder(
                self.prompt_embeddings,
                self.tokenized_prompts,
                rep_text,
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self._cached_text_features = text_features
            self._cached_rep_visual = rep_visual
            self._cached_mode = cache_mode

        return self._cached_text_features, self._cached_rep_visual

    def forward_train_samples(self, image: torch.Tensor, num_samples: int):
        num_samples = max(1, int(num_samples))
        outputs = []
        for _ in range(num_samples):
            rep_text, rep_visual = self.representation_learner.project_sample_tokens()
            outputs.append(self._encode_with_tokens(image, rep_text, rep_visual))
        return outputs

    @torch.no_grad()
    def forward_eval(
        self,
        image: torch.Tensor,
        num_samples: int | None = None,
        use_posterior_mean: bool | None = None,
    ):
        if num_samples is None:
            num_samples = self.n_mc_test
        num_samples = max(1, int(num_samples))

        if use_posterior_mean is None:
            use_posterior_mean = self.eval_use_posterior_mean

        # Fast deterministic inference with cache.
        if use_posterior_mean and num_samples == 1:
            text_features, rep_visual = self._get_cached_mean_eval_state()
            return self._forward_with_cached_text(image, rep_visual, text_features)

        # Otherwise do MC averaging. If use_posterior_mean=True and num_samples>1,
        # include the posterior mean as the first member in the ensemble.
        sample_outputs = []

        if use_posterior_mean:
            rep_text, rep_visual = self.representation_learner.project_mean_tokens()
            sample_outputs.append(self._encode_with_tokens(image, rep_text, rep_visual))
            remaining = num_samples - 1
        else:
            remaining = num_samples

        for _ in range(remaining):
            rep_text, rep_visual = self.representation_learner.project_sample_tokens()
            sample_outputs.append(self._encode_with_tokens(image, rep_text, rep_visual))

        return self._aggregate_outputs(sample_outputs)