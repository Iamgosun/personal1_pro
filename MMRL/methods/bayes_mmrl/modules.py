from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from backbones.prompt_builder import CUSTOM_TEMPLATES
from methods.mmrl_family.modules import (
    CLIPTextEncoderPlain,
    MMRLFamilyRepresentationLearner,
    MMRLTextEncoder,
    build_zero_shot_text_features,
)


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _get_with_fallback(obj, primary: str, fallback: str, default):
    if hasattr(obj, primary):
        return getattr(obj, primary)
    if hasattr(obj, fallback):
        return getattr(obj, fallback)
    return default


def _canonical_eval_mode(mode: str | None) -> str:
    mode = str(mode or "mc_predictive")
    if mode == "mean_only":
        return "posterior_mean"
    if mode == "mc_only":
        return "mc_predictive"
    if mode in {"posterior_mean", "mc_predictive", "mean_plus_mc"}:
        return mode
    raise ValueError(f"Unsupported EVAL_MODE: {mode}")


class BayesianTensorParameter(nn.Module):
    """
    Generic Bayesian tensor parameter with diagonal Gaussian posterior:

        q(W) = N(mu, sigma^2)
        p(W) = N(prior_mean, prior_std^2 I)

    Supported sigma modes for 2D tensors:
        - global
        - per_token / row   -> [shape[0], 1]
        - per_dim   / col   -> [1, shape[1]]
        - full              -> shape
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        sigma_mode: str,
        prior_std: float,
        posterior_rho_init: float,
        init_std: float = 0.02,
        min_sigma: float = 1e-6,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.sigma_mode = str(sigma_mode)
        self.prior_std = float(prior_std)
        self.min_sigma = float(min_sigma)

        self.posterior_mean = nn.Parameter(
            torch.empty(*self.shape, dtype=torch.float32)
        )
        nn.init.normal_(self.posterior_mean, std=float(init_std))

        rho_shape = self._resolve_rho_shape(self.shape, self.sigma_mode)
        self.posterior_rho = nn.Parameter(
            torch.full(rho_shape, float(posterior_rho_init), dtype=torch.float32)
        )

        self.register_buffer(
            "prior_mean",
            torch.zeros(*self.shape, dtype=torch.float32),
        )

    @staticmethod
    def _resolve_rho_shape(shape: Tuple[int, int], sigma_mode: str):
        if sigma_mode == "global":
            return ()
        if sigma_mode in {"per_token", "row"}:
            return (shape[0], 1)
        if sigma_mode in {"per_dim", "col"}:
            return (1, shape[1])
        if sigma_mode == "full":
            return shape
        raise ValueError(f"Unsupported sigma mode: {sigma_mode}")

    def posterior_sigma(self) -> torch.Tensor:
        return F.softplus(self.posterior_rho.float()) + self.min_sigma

    def expanded_sigma(self) -> torch.Tensor:
        return self.posterior_sigma().expand_as(self.posterior_mean)

    def reset_posterior_random(self, init_std: float):
        with torch.no_grad():
            nn.init.normal_(self.posterior_mean, std=float(init_std))

    def set_posterior_from_tensor(self, tensor: torch.Tensor, noise_std: float = 0.0):
        tensor = tensor.detach().float()
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            self.posterior_mean.copy_(tensor)
            if float(noise_std) > 0:
                self.posterior_mean.add_(
                    torch.randn_like(self.posterior_mean) * float(noise_std)
                )

    def set_prior_mean(self, tensor: torch.Tensor):
        tensor = tensor.detach().float()
        if tuple(tensor.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            self.prior_mean.copy_(tensor)

    def sample_tensor(self, use_posterior_mean: bool = False) -> torch.Tensor:
        if use_posterior_mean:
            return self.posterior_mean.float()
        eps = torch.randn_like(self.posterior_mean)
        return self.posterior_mean.float() + self.expanded_sigma() * eps

    def kl_divergence(self) -> torch.Tensor:
        mu = self.posterior_mean.float()
        sigma2 = self.expanded_sigma().pow(2)
        prior_var = mu.new_tensor(self.prior_std ** 2)
        prior_mean = self.prior_mean.float()

        kl = 0.5 * (
            sigma2 / prior_var
            + (mu - prior_mean).pow(2) / prior_var
            - 1.0
            - torch.log(sigma2 / prior_var)
        )
        return kl.sum()


class DeterministicRepresentationLearnerAdapter(nn.Module):
    """
    Wrap original deterministic MMRL representation learner with a compatible API.
    Used when Bayes is moved from R to proj_rep.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.base = MMRLFamilyRepresentationLearner(
            cfg,
            cfg.BAYES_MMRL,
            classnames,
            clip_model,
        )

    @property
    def tokenized_prompts(self):
        return self.base.tokenized_prompts

    @property
    def prompt_embeddings(self):
        return self.base.prompt_embeddings

    def project_mean_tokens(self):
        return self.base()

    def project_sample_tokens(self):
        return self.base()

    def posterior_sigma(self):
        device = self.base.compound_rep_tokens.device
        return torch.zeros(
            self.base.compound_rep_tokens.shape[0],
            1,
            device=device,
            dtype=torch.float32,
        )

    def kl_divergence(self):
        return self.base.compound_rep_tokens.new_zeros(())

    def forward(self):
        return self.base()


class BayesianMultiModalRepresentationLearner(nn.Module):
    """
    Bayesian version of shared representation learner.

    Supported modes:
        - Scheme A: zero isotropic prior over R
        - Scheme B: CLIP-informed prior over R

    The trainable modality projection layers remain deterministic,
    same as original MMRL.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        bayes_cfg = cfg.BAYES_MMRL

        n_rep_tokens = int(bayes_cfg.N_REP_TOKENS)
        rep_dim = int(bayes_cfg.REP_DIM)
        self.dtype = clip_model.dtype
        self.rep_layers_length = len(bayes_cfg.REP_LAYERS)

        rep_sigma_mode = str(
            _get_with_fallback(bayes_cfg, "REP_SIGMA_MODE", "SIGMA_MODE", "per_token")
        )
        rep_prior_std = float(
            _get_with_fallback(bayes_cfg, "REP_PRIOR_STD", "PRIOR_STD", 0.05)
        )
        rep_init_std = float(
            getattr(bayes_cfg, "REP_INIT_STD", rep_prior_std)
        )
        rep_rho_init = float(
            _get_with_fallback(
                bayes_cfg,
                "REP_POSTERIOR_RHO_INIT",
                "POSTERIOR_RHO_INIT",
                -3.9,
            )
        )

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

        self.rep_posterior = BayesianTensorParameter(
            shape=(n_rep_tokens, rep_dim),
            sigma_mode=rep_sigma_mode,
            prior_std=rep_prior_std,
            posterior_rho_init=rep_rho_init,
            init_std=rep_init_std,
        )

        self.compound_rep_tokens_r2vproj = _get_clones(
            nn.Linear(rep_dim, visual_dim),
            self.rep_layers_length,
        )
        self.compound_rep_tokens_r2tproj = _get_clones(
            nn.Linear(rep_dim, text_dim),
            self.rep_layers_length,
        )

    @property
    def posterior_mean(self):
        return self.rep_posterior.posterior_mean

    def posterior_sigma(self):
        return self.rep_posterior.posterior_sigma()

    def kl_divergence(self):
        return self.rep_posterior.kl_divergence()

    def apply_rep_prior(
        self,
        prior_mean: torch.Tensor,
        init_mode: str = "prior_mean_noise",
        init_std: float = 0.01,
        prior_std: float | None = None,
    ):
        if prior_std is not None:
            self.rep_posterior.prior_std = float(prior_std)

        self.rep_posterior.set_prior_mean(prior_mean)

        init_mode = str(init_mode)
        if init_mode == "prior_mean":
            self.rep_posterior.set_posterior_from_tensor(prior_mean, noise_std=0.0)
        elif init_mode == "prior_mean_noise":
            self.rep_posterior.set_posterior_from_tensor(
                prior_mean,
                noise_std=float(init_std),
            )
        elif init_mode == "normal":
            self.rep_posterior.reset_posterior_random(init_std)
        else:
            raise ValueError(f"Unsupported REP_INIT_MODE: {init_mode}")

    def _project_rep_tokens(
        self,
        rep_tokens: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        compound_rep_tokens_text: List[torch.Tensor] = []
        compound_rep_tokens_visual: List[torch.Tensor] = []

        rep_tokens = rep_tokens.to(self.compound_rep_tokens_r2tproj[0].weight.dtype)

        for index in range(self.rep_layers_length):
            rep_text = self.compound_rep_tokens_r2tproj[index](rep_tokens).type(
                self.dtype
            )
            rep_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens).type(
                self.dtype
            )
            compound_rep_tokens_text.append(rep_text)
            compound_rep_tokens_visual.append(rep_visual)

        return compound_rep_tokens_text, compound_rep_tokens_visual

    def project_mean_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._project_rep_tokens(
            self.rep_posterior.sample_tensor(use_posterior_mean=True)
        )

    def project_sample_tokens(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._project_rep_tokens(
            self.rep_posterior.sample_tensor(use_posterior_mean=False)
        )

    def forward(self, use_posterior_mean: bool = False):
        if use_posterior_mean:
            return self.project_mean_tokens()
        return self.project_sample_tokens()


class BayesianVisualEncoderWrapper(nn.Module):
    """
    Bayesian wrapper for the representation visual projection P_v^r.

    This avoids modifying clip/model_mmrl.py directly.
    """

    def __init__(self, base_visual: nn.Module, cfg):
        super().__init__()
        if not hasattr(base_visual, "proj_rep"):
            raise ValueError("Bayesian proj_rep requires a ViT visual backbone with proj_rep")

        self.base = base_visual
        bayes_cfg = cfg.BAYES_MMRL

        sigma_mode = str(getattr(bayes_cfg, "PROJ_REP_SIGMA_MODE", "row"))
        prior_std = float(getattr(bayes_cfg, "PROJ_REP_PRIOR_STD", 0.01))
        rho_init = float(getattr(bayes_cfg, "PROJ_REP_POSTERIOR_RHO_INIT", -5.5))
        init_mode = str(getattr(bayes_cfg, "PROJ_REP_INIT_MODE", "pretrained_mean"))
        init_std = float(getattr(bayes_cfg, "PROJ_REP_INIT_STD", 0.0))

        pretrained_proj_rep = self.base.proj_rep.detach().float()

        self.bayes_proj_rep = BayesianTensorParameter(
            shape=tuple(pretrained_proj_rep.shape),
            sigma_mode=sigma_mode,
            prior_std=prior_std,
            posterior_rho_init=rho_init,
            init_std=max(init_std, 1e-6),
        )

        # prior stays zero-mean isotropic by default; only posterior init changes
        if init_mode == "pretrained_mean":
            self.bayes_proj_rep.set_posterior_from_tensor(
                pretrained_proj_rep,
                noise_std=0.0,
            )
        elif init_mode == "pretrained_mean_noise":
            self.bayes_proj_rep.set_posterior_from_tensor(
                pretrained_proj_rep,
                noise_std=init_std,
            )
        elif init_mode == "normal":
            self.bayes_proj_rep.reset_posterior_random(init_std)
        else:
            raise ValueError(f"Unsupported PROJ_REP_INIT_MODE: {init_mode}")

    def posterior_sigma(self):
        return self.bayes_proj_rep.posterior_sigma()

    def kl_divergence(self):
        return self.bayes_proj_rep.kl_divergence()

    def _resolve_proj_rep(self, use_posterior_mean: bool) -> torch.Tensor:
        return self.bayes_proj_rep.sample_tensor(use_posterior_mean=use_posterior_mean)

    def _forward_vit_with_proj(
        self,
        inputs,
        proj_rep: torch.Tensor,
    ):
        x = inputs[0]
        compound_rep_tokens = list(inputs[1])

        x = self.base.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.base.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.base.positional_embedding.to(x.dtype)
        x = self.base.ln_pre(x)

        x = x.permute(1, 0, 2)
        outputs = self.base.transformer([x, compound_rep_tokens, 0])
        x = outputs[0]
        x = x.permute(1, 0, 2)

        n_tokens = compound_rep_tokens[0].shape[0]
        x_rep = self.base.ln_post(x[:, 1 : 1 + n_tokens, :]).mean(dim=1)
        x_rep = x_rep @ proj_rep.to(x_rep.dtype)

        x_cls = self.base.ln_post(x[:, 0, :])
        x_cls = x_cls @ self.base.proj

        return x_cls, x_rep

    def forward(self, inputs, use_posterior_mean: bool = False):
        proj_rep = self._resolve_proj_rep(use_posterior_mean=use_posterior_mean)
        return self._forward_vit_with_proj(inputs, proj_rep)


class BayesianCustomMMRLModel(nn.Module):
    """
    Supports:
        - Scheme A: Bayes on R with zero prior
        - Scheme B: Bayes on R with CLIP prior
        - Scheme C: Bayes on P_v^r with deterministic R

    Evaluation:
        - posterior_mean: deterministic posterior mean
        - mc_predictive : average predictive probabilities
        - mean_plus_mc  : include posterior mean sample + MC samples
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        bayes_cfg = cfg.BAYES_MMRL

        self.alpha = float(bayes_cfg.ALPHA)
        self.bayes_target = str(getattr(bayes_cfg, "BAYES_TARGET", "rep_tokens"))
        self.eval_mode = _canonical_eval_mode(getattr(bayes_cfg, "EVAL_MODE", "mc_predictive"))
        self.eval_use_posterior_mean = bool(
            getattr(bayes_cfg, "EVAL_USE_POSTERIOR_MEAN", False)
        )
        self.n_mc_test = max(1, int(bayes_cfg.N_MC_TEST))

        if self.bayes_target == "rep_tokens":
            self.representation_learner = BayesianMultiModalRepresentationLearner(
                cfg,
                classnames,
                clip_model,
            )
        elif self.bayes_target == "proj_rep":
            self.representation_learner = DeterministicRepresentationLearnerAdapter(
                cfg,
                classnames,
                clip_model,
            )
        else:
            raise ValueError(f"Unsupported BAYES_TARGET: {self.bayes_target}")

        self.register_buffer(
            "tokenized_prompts",
            self.representation_learner.tokenized_prompts.clone(),
        )
        self.register_buffer(
            "prompt_embeddings",
            self.representation_learner.prompt_embeddings.clone(),
        )

        if self.bayes_target == "proj_rep":
            self.image_encoder = BayesianVisualEncoderWrapper(clip_model.visual, cfg)
        else:
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

    def kl_terms(self) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=self.prompt_embeddings.device, dtype=torch.float32)
        rep_kl = (
            self.representation_learner.kl_divergence()
            if hasattr(self.representation_learner, "kl_divergence")
            else zero
        )
        proj_kl = (
            self.image_encoder.kl_divergence()
            if isinstance(self.image_encoder, BayesianVisualEncoderWrapper)
            else zero
        )
        return {
            "rep_tokens": rep_kl,
            "proj_rep": proj_kl,
        }

    def posterior_stats(self) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}
        if hasattr(self.representation_learner, "posterior_sigma"):
            stats["rep_posterior_sigma"] = (
                self.representation_learner.posterior_sigma().detach()
            )
        if isinstance(self.image_encoder, BayesianVisualEncoderWrapper):
            stats["proj_rep_posterior_sigma"] = (
                self.image_encoder.posterior_sigma().detach()
            )
        return stats

    def _encode_image(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        use_posterior_mean_proj: bool = False,
    ):
        if isinstance(self.image_encoder, BayesianVisualEncoderWrapper):
            return self.image_encoder(
                [image.type(self.dtype), list(compound_rep_tokens_visual)],
                use_posterior_mean=use_posterior_mean_proj,
            )
        return self.image_encoder([image.type(self.dtype), list(compound_rep_tokens_visual)])

    def _encode_with_tokens(
        self,
        image: torch.Tensor,
        compound_rep_tokens_text: Sequence[torch.Tensor],
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        use_posterior_mean_proj: bool = False,
    ):
        text_features = self.text_encoder(
            self.prompt_embeddings,
            self.tokenized_prompts,
            compound_rep_tokens_text,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features, image_features_rep = self._encode_image(
            image,
            compound_rep_tokens_visual,
            use_posterior_mean_proj=use_posterior_mean_proj,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(
            dim=-1, keepdim=True
        )

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _forward_with_cached_text(
        self,
        image: torch.Tensor,
        compound_rep_tokens_visual: Sequence[torch.Tensor],
        text_features: torch.Tensor,
        use_posterior_mean_proj: bool = False,
    ):
        image_features, image_features_rep = self._encode_image(
            image,
            compound_rep_tokens_visual,
            use_posterior_mean_proj=use_posterior_mean_proj,
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(
            dim=-1, keepdim=True
        )

        logits = 100.0 * image_features @ text_features.t()
        logits_rep = 100.0 * image_features_rep @ text_features.t()
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        return logits, logits_rep, logits_fusion, image_features, text_features

    def _aggregate_train_outputs(self, sample_outputs):
        logits = torch.stack([out[0] for out in sample_outputs], dim=0).mean(dim=0)
        logits_rep = torch.stack([out[1] for out in sample_outputs], dim=0).mean(dim=0)
        logits_fusion = torch.stack([out[2] for out in sample_outputs], dim=0).mean(dim=0)
        image_features = torch.stack([out[3] for out in sample_outputs], dim=0).mean(dim=0)
        text_features = torch.stack([out[4] for out in sample_outputs], dim=0).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return logits, logits_rep, logits_fusion, image_features, text_features

    def _aggregate_eval_outputs(self, sample_outputs):
        eps = 1e-8

        probs = torch.stack(
            [torch.softmax(out[0], dim=-1) for out in sample_outputs],
            dim=0,
        ).mean(dim=0)
        probs_rep = torch.stack(
            [torch.softmax(out[1], dim=-1) for out in sample_outputs],
            dim=0,
        ).mean(dim=0)
        probs_fusion = torch.stack(
            [torch.softmax(out[2], dim=-1) for out in sample_outputs],
            dim=0,
        ).mean(dim=0)

        logits = torch.log(probs.clamp_min(eps))
        logits_rep = torch.log(probs_rep.clamp_min(eps))
        logits_fusion = torch.log(probs_fusion.clamp_min(eps))

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
            outputs.append(
                self._encode_with_tokens(
                    image,
                    rep_text,
                    rep_visual,
                    use_posterior_mean_proj=False,
                )
            )
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

        if use_posterior_mean is not None:
            if use_posterior_mean and num_samples == 1:
                eval_mode = "posterior_mean"
            elif use_posterior_mean and num_samples > 1:
                eval_mode = "mean_plus_mc"
            else:
                eval_mode = "mc_predictive"
        else:
            eval_mode = self.eval_mode
            if eval_mode is None:
                eval_mode = (
                    "posterior_mean"
                    if self.eval_use_posterior_mean
                    else "mc_predictive"
                )

        eval_mode = _canonical_eval_mode(eval_mode)

        if eval_mode == "posterior_mean":
            text_features, rep_visual = self._get_cached_mean_eval_state()
            return self._forward_with_cached_text(
                image,
                rep_visual,
                text_features,
                use_posterior_mean_proj=True,
            )

        if eval_mode == "mc_predictive":
            sample_outputs = []
            for _ in range(num_samples):
                rep_text, rep_visual = self.representation_learner.project_sample_tokens()
                sample_outputs.append(
                    self._encode_with_tokens(
                        image,
                        rep_text,
                        rep_visual,
                        use_posterior_mean_proj=False,
                    )
                )
            return self._aggregate_eval_outputs(sample_outputs)

        if eval_mode == "mean_plus_mc":
            sample_outputs = []

            rep_text, rep_visual = self.representation_learner.project_mean_tokens()
            sample_outputs.append(
                self._encode_with_tokens(
                    image,
                    rep_text,
                    rep_visual,
                    use_posterior_mean_proj=True,
                )
            )

            for _ in range(max(0, num_samples - 1)):
                rep_text, rep_visual = self.representation_learner.project_sample_tokens()
                sample_outputs.append(
                    self._encode_with_tokens(
                        image,
                        rep_text,
                        rep_visual,
                        use_posterior_mean_proj=False,
                    )
                )

            return self._aggregate_eval_outputs(sample_outputs)

        raise ValueError(f"Unsupported EVAL_MODE: {eval_mode}")