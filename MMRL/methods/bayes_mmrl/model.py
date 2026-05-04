from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from backbones.clip_loader import load_mmrl_clip_to_cpu
from backbones.freeze import freeze_all_but
from core.registry import METHOD_REGISTRY
from core.types import MethodOutputs
from data.build import build_split_dataset
from methods.base import BaseMethod
from methods.mmrl.loss import MMRLLoss

from .loss import BayesMMRLLossAdapter
from .modules import (
    CLIPTextEncoderPlain,
    BayesianCustomMMRLModel,
    build_zero_shot_text_features,
)


@METHOD_REGISTRY.register("BayesMMRL")
class BayesMMRLMethod(BaseMethod):
    method_name = "BayesMMRL"

    def _resolve_trainable_substrings(self):
        substrings = ["representation_learner"]
        if self.bayes_target == "proj_rep":
            substrings.append("image_encoder.bayes_proj_rep")
        else:
            substrings.append("image_encoder.proj_rep")
        return substrings

    def _build_support_loader(self):
        dataset = build_split_dataset(
            self.cfg,
            self.dm.dataset.train_x,
            is_train=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE,
            sampler=SequentialSampler(dataset),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    @torch.no_grad()
    def _compute_support_image_class_prototypes(self):
        loader = self._build_support_loader()
        feat_dim = self.text_features_clip.shape[-1]

        feat_sums = torch.zeros(
            self.num_classes,
            feat_dim,
            device=self.device,
            dtype=torch.float32,
        )
        counts = torch.zeros(
            self.num_classes,
            device=self.device,
            dtype=torch.float32,
        )

        for batch in loader:
            images = batch["img"].to(self.device)
            labels = batch["label"].to(self.device)

            img_features = self.image_encoder_clip(images.type(self.dtype))
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            feat_sums.index_add_(0, labels, img_features.float())
            counts.index_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=torch.float32),
            )

        counts = counts.clamp_min(1.0).unsqueeze(-1)
        protos = feat_sums / counts
        protos = protos / protos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return protos

    @torch.no_grad()
    def _build_rep_prior_mean(self):
        bayes_cfg = self.cfg.BAYES_MMRL
        rep_prior_mode = str(getattr(bayes_cfg, "REP_PRIOR_MODE", "zero"))
        rep_dim = int(bayes_cfg.REP_DIM)
        n_rep_tokens = int(bayes_cfg.N_REP_TOKENS)

        if rep_prior_mode == "zero":
            return torch.zeros(
                n_rep_tokens,
                rep_dim,
                device=self.device,
                dtype=torch.float32,
            )

        if rep_prior_mode != "clip_joint":
            raise ValueError(
                f"Unsupported REP_PRIOR_MODE: {rep_prior_mode}. "
                "Expected one of {'zero', 'clip_joint'}."
            )

        text_basis = self.text_features_clip[: self.num_classes].float()
        if text_basis.shape[-1] != rep_dim:
            raise ValueError(
                f"CLIP prior requires REP_DIM == CLIP embed dim, got {rep_dim} vs {text_basis.shape[-1]}"
            )

        image_basis = self._compute_support_image_class_prototypes()
        blend = float(getattr(bayes_cfg, "CLIP_PRIOR_BLEND", 0.5))
        centers = (1.0 - blend) * text_basis + blend * image_basis

        centers = centers - centers.mean(dim=0, keepdim=True)

        if torch.allclose(centers, torch.zeros_like(centers)):
            centers = text_basis

        _, _, vh = torch.linalg.svd(centers, full_matrices=False)
        basis = vh[: min(n_rep_tokens, vh.shape[0])]

        if basis.shape[0] < n_rep_tokens:
            extra = torch.randn(
                n_rep_tokens - basis.shape[0],
                basis.shape[1],
                device=basis.device,
                dtype=basis.dtype,
            )
            basis = torch.cat([basis, extra], dim=0)

        basis = F.normalize(basis, dim=-1)
        scale = float(getattr(bayes_cfg, "CLIP_PRIOR_SCALE", 0.05))
        return (scale * basis).detach()

    def _assert_initial_kl_zero(
        self,
        kl_value: torch.Tensor,
        name: str,
        atol: float = 1e-6,
    ):
        kl_scalar = float(kl_value.detach().cpu().item())
        if abs(kl_scalar) > atol:
            raise RuntimeError(
                f"Expected initial KL({name}) ~= 0 because q0=p, "
                f"but got {kl_scalar:.8e}"
            )

    def build(self):
        cfg = self.cfg
        bayes_cfg = cfg.BAYES_MMRL
        self.kl_warmup_epochs = int(getattr(bayes_cfg, "KL_WARMUP_EPOCHS", 0))

        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)
        self.bayes_target = str(getattr(bayes_cfg, "BAYES_TARGET", "rep_tokens"))

        clip_model = load_mmrl_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_mmrl_clip_to_cpu(cfg, "CLIP")

        if bayes_cfg.PREC in {"fp32", "amp"}:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype
        self.n_mc_train = max(1, int(bayes_cfg.N_MC_TRAIN))
        self.n_mc_test = max(1, int(bayes_cfg.N_MC_TEST))
        self.eval_use_posterior_mean = bool(bayes_cfg.EVAL_USE_POSTERIOR_MEAN)

        self.rep_kl_weight = float(getattr(bayes_cfg, "REP_KL_WEIGHT", 1e-4))
        self.proj_rep_kl_weight = float(getattr(bayes_cfg, "PROJ_REP_KL_WEIGHT", 1e-6))

        self.text_encoder_clip = CLIPTextEncoderPlain(clip_model_zero_shot).to(
            self.device
        )

        with torch.no_grad():
            text_features_clip = build_zero_shot_text_features(
                cfg,
                classnames,
                clip_model_zero_shot,
                self.text_encoder_clip,
            )
            self.text_features_clip = (
                text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
            ).to(self.device)

        self.image_encoder_clip = clip_model_zero_shot.visual.to(self.device)

        self.model = BayesianCustomMMRLModel(cfg, classnames, clip_model).to(
            self.device
        )

        # ------------------------------------------------------------------
        # unify initialization:
        #   A/B: random variable = R
        #        p(R)=N(m0, Sigma), q0(R)=p(R)
        #   C  : random variable = proj_rep
        #        p(W)=N(W_pretrained, Sigma), q0(W)=p(W)
        # ------------------------------------------------------------------
        if self.bayes_target == "rep_tokens":
            prior_mean = self._build_rep_prior_mean()
            prior_std = float(getattr(bayes_cfg, "REP_PRIOR_STD", 0.05))

            self.model.representation_learner.configure_rep_prior_and_initialize(
                prior_mean=prior_mean.to(self.device),
                prior_std=prior_std,
            )

            init_kl = self.model.kl_terms()["rep_tokens"]
            self._assert_initial_kl_zero(init_kl, "rep_tokens")

        elif self.bayes_target == "proj_rep":
            init_kl = self.model.kl_terms()["proj_rep"]
            self._assert_initial_kl_zero(init_kl, "proj_rep")

        else:
            raise ValueError(f"Unsupported BAYES_TARGET: {self.bayes_target}")

        enabled = freeze_all_but(
            self.model,
            self._resolve_trainable_substrings(),
        )
        print(f"[BayesMMRLMethod] trainable params: {enabled}")

        # Kept for full-MC ablation/backward compatibility.
        self.sample_loss = MMRLLoss(
            reg_weight=bayes_cfg.REG_WEIGHT,
            alpha=bayes_cfg.ALPHA,
        )
        self.loss = BayesMMRLLossAdapter()
        return self

    def set_kl_normalizer(self, normalizer):
        try:
            normalizer = float(normalizer)
        except (TypeError, ValueError):
            normalizer = 1.0

        self.kl_normalizer = max(1.0, normalizer)

    def set_kl_beta(self, beta):
        try:
            beta = float(beta)
        except (TypeError, ValueError):
            beta = 1.0

        self.kl_beta = min(1.0, max(0.0, beta))

    def get_precision(self) -> str:
        return self.cfg.BAYES_MMRL.PREC

    def select_train_logits(self, outputs):
        return outputs.aux_logits.get("fusion", outputs.logits)

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Normalized predictive entropy in [0, 1].

        Args:
            logits: [B, C]

        Returns:
            entropy: [B]
        """
        probs = torch.softmax(logits.float(), dim=-1).clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)

        num_classes = logits.shape[-1]
        if num_classes > 1:
            entropy = entropy / torch.log(
                logits.new_tensor(float(num_classes), dtype=torch.float32)
            )

        return entropy

    @staticmethod
    def _margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Top-1 minus top-2 probability margin.

        Args:
            logits: [B, C]

        Returns:
            margin: [B]
        """
        probs = torch.softmax(logits.float(), dim=-1)

        if probs.shape[-1] == 1:
            return probs[..., 0]

        top2 = torch.topk(probs, k=2, dim=-1).values
        return top2[..., 0] - top2[..., 1]

    def _dynamic_no_beta_fusion(
        self,
        logits_c: torch.Tensor,
        logits_r: torch.Tensor,
    ):
        """
        Dynamic no-beta fusion.

        No learned gate and no beta search:
            prior_r = 1 - alpha
            score = logit(prior_r)
                    + (H(C) - H(R))
                    + (margin(R) - margin(C))
            omega_r = sigmoid(score)
            logits_dyn = (1 - omega_r) * logits_c + omega_r * logits_r

        Args:
            logits_c: [B, C], main/CLIP branch logits.
            logits_r: [B, C], representation branch logits.

        Returns:
            logits_dynamic: [B, C]
            extras: detached diagnostics.
        """
        alpha = float(self.cfg.BAYES_MMRL.ALPHA)
        prior_r = 1.0 - alpha
        prior_r = min(max(prior_r, 1e-6), 1.0 - 1e-6)

        base = torch.logit(
            logits_c.new_tensor(prior_r, dtype=torch.float32)
        )

        entropy_c = self._entropy_from_logits(logits_c)
        entropy_r = self._entropy_from_logits(logits_r)
        u_gap = entropy_c - entropy_r

        margin_c = self._margin_from_logits(logits_c)
        margin_r = self._margin_from_logits(logits_r)
        margin_gain = margin_r - margin_c

        score = base + u_gap + margin_gain
        omega_r = torch.sigmoid(score).to(logits_c.dtype)

        logits_dynamic = (
            (1.0 - omega_r.unsqueeze(-1)) * logits_c
            + omega_r.unsqueeze(-1) * logits_r
        )

        extras = {
            "dynamic_no_beta_weight_r": omega_r.detach(),
            "dynamic_no_beta_u_gap": u_gap.detach(),
            "dynamic_no_beta_margin_gain": margin_gain.detach(),
        }

        return logits_dynamic, extras

    def _eval_fusion_variant(self) -> str:
        """
        Which fusion variant should be used as aux_logits['fusion'].

        Valid values:
            - static
            - dynamic_no_beta
        """
        variant = str(
            getattr(
                self.cfg.BAYES_MMRL,
                "EVAL_FUSION_VARIANT",
                "static",
            )
        ).lower()

        if variant not in {"static", "dynamic_no_beta"}:
            raise ValueError(
                "BAYES_MMRL.EVAL_FUSION_VARIANT must be one of "
                "{'static', 'dynamic_no_beta'}, "
                f"got {variant}"
            )

        return variant

    def _build_fusion_aux_logits(
        self,
        logits_main: torch.Tensor,
        logits_rep: torch.Tensor,
        logits_static: torch.Tensor,
    ):
        """
        Build all fusion variants for reporting.

        Returns:
            aux_logits:
                rep
                fusion
                fusion_static
                fusion_dynamic_no_beta

            extras:
                eval_fusion_variant
                dynamic no-beta diagnostics
        """
        logits_dynamic, dyn_extras = self._dynamic_no_beta_fusion(
            logits_c=logits_main,
            logits_r=logits_rep,
        )

        variant = self._eval_fusion_variant()

        if variant == "dynamic_no_beta":
            logits_selected = logits_dynamic
        else:
            logits_selected = logits_static

        aux_logits = {
            "rep": logits_rep,
            "fusion": logits_selected,
            "fusion_static": logits_static,
            "fusion_dynamic_no_beta": logits_dynamic,
        }

        extras = {
            "eval_fusion_variant": variant,
            **dyn_extras,
        }

        return aux_logits, extras

    def select_eval_logits(self, outputs, eval_ctx):
        logits = outputs.logits
        logits_fusion = outputs.aux_logits.get("fusion")
        if logits_fusion is None:
            return logits

        protocol = eval_ctx.protocol
        dataset = eval_ctx.dataset_name
        sub_cls = eval_ctx.subsample_classes or "all"

        if protocol == "B2N":
            if sub_cls == "base":
                return logits_fusion
            return logits

        if protocol == "FS":
            return logits_fusion

        if protocol == "CD":
            if dataset == "ImageNet":
                return logits_fusion
            return logits

        return logits


    def _maybe_posterior_stats(self):
        """
        Matrix-normal posterior_stats is not free: it recomputes posterior_sigma,
        which calls token Cholesky and feature covariance stats. Keep it off by
        default during every forward; set BAYES_MMRL.LOG_POSTERIOR_STATS_EVERY_FORWARD
        to True if you need the old behavior.
        """
        enabled = bool(
            getattr(
                self.cfg.BAYES_MMRL,
                "LOG_POSTERIOR_STATS_EVERY_FORWARD",
                False,
            )
        )

        if not enabled:
            return {}

        return dict(self.model.posterior_stats())

    def _skip_kl_when_beta_zero(self) -> bool:
        return bool(
            getattr(
                self.cfg.BAYES_MMRL,
                "SKIP_KL_WHEN_BETA_ZERO",
                True,
            )
        )

    def _kl_terms_for_loss(self, data_term: torch.Tensor):
        kl_normalizer = float(getattr(self, "kl_normalizer", 1.0))
        kl_normalizer = max(1.0, kl_normalizer)

        kl_beta = float(getattr(self, "kl_beta", 1.0))
        kl_beta = min(1.0, max(0.0, kl_beta))

        if kl_beta == 0.0 and self._skip_kl_when_beta_zero():
            raw_kl_rep = data_term.detach().new_zeros(())
            raw_kl_proj_rep = data_term.detach().new_zeros(())
        else:
            raw_kl = self.model.kl_terms()
            raw_kl_rep = raw_kl["rep_tokens"]
            raw_kl_proj_rep = raw_kl["proj_rep"]

        kl_rep_term = kl_beta * self.rep_kl_weight * raw_kl_rep / kl_normalizer
        kl_proj_rep_term = (
            kl_beta * self.proj_rep_kl_weight * raw_kl_proj_rep / kl_normalizer
        )
        kl_term = kl_rep_term + kl_proj_rep_term

        return {
            "raw_kl_rep": raw_kl_rep,
            "raw_kl_proj_rep": raw_kl_proj_rep,
            "kl_rep_term": kl_rep_term,
            "kl_proj_rep_term": kl_proj_rep_term,
            "kl_term": kl_term,
            "kl_normalizer": data_term.detach().new_tensor(kl_normalizer),
            "kl_beta": data_term.detach().new_tensor(kl_beta),
        }


    def _main_consistency_penalty(self, logits_stack: torch.Tensor):
        """
        Penalize sample-to-sample variation of main-branch predictions.

        Args:
            logits_stack: [S, B, C], main logits from full-MC samples.

        Returns:
            scalar tensor.

        Modes:
            - "prob":  variance of softmax probabilities. More scale-stable.
            - "logit": variance of raw logits. More direct but affected by 100.0 scale.
        """
        if logits_stack.shape[0] <= 1:
            return logits_stack.detach().new_zeros(())

        mode = str(
            getattr(
                self.cfg.BAYES_MMRL,
                "MAIN_CONSISTENCY_MODE",
                "prob",
            )
        ).lower()

        if mode == "prob":
            probs_stack = torch.softmax(logits_stack, dim=-1)
            return probs_stack.var(dim=0, unbiased=False).mean()

        if mode == "logit":
            return logits_stack.var(dim=0, unbiased=False).mean()

        raise ValueError(
            f"Unsupported MAIN_CONSISTENCY_MODE: {mode}. "
            "Expected one of {'prob', 'logit'}."
        )



    def _build_outputs_from_samples(self, label, img_ref, sample_outputs):
        """
        Full-MC VI objective with optional main-consistency regularization.

        Base VI term:
            E_q[L_MMRL(r)] + KL(q||p)

        Optional main consistency:
            + lambda_main_cons * Var_s[p_main^{(s)}]
              or
            + lambda_main_cons * Var_s[z_main^{(s)}]

        This does not replace the VI objective. It only constrains the main
        branch to be stable under posterior samples, while the rep branch can
        still carry useful MC uncertainty.
        """
        per_sample_losses = []
        logits_list = []
        logits_rep_list = []
        logits_fusion_list = []
        image_features_list = []
        text_features_list = []

        for logits, logits_rep, logits_fusion, image_features, text_features in sample_outputs:
            text_features = text_features[: self.num_classes]

            loss_s = self.sample_loss(
                logits,
                logits_rep,
                image_features,
                text_features,
                img_ref,
                self.text_features_clip,
                label,
            )
            per_sample_losses.append(loss_s)

            logits_list.append(logits)
            logits_rep_list.append(logits_rep)
            logits_fusion_list.append(logits_fusion)
            image_features_list.append(image_features)
            text_features_list.append(text_features)

        # ------------------------------------------------------------------
        # 1) Original VI data term: E_q[L_MMRL(r)]
        # ------------------------------------------------------------------
        data_term = torch.stack(per_sample_losses, dim=0).mean(dim=0)

        # ------------------------------------------------------------------
        # 2) KL term
        # ------------------------------------------------------------------
        kl_terms = self._kl_terms_for_loss(data_term)

        # ------------------------------------------------------------------
        # 3) Main-consistency penalty
        #    This is the new mechanism diagnostic/regularizer.
        # ------------------------------------------------------------------
        logits_stack = torch.stack(logits_list, dim=0)  # [S, B, C]
        main_consistency = self._main_consistency_penalty(logits_stack)

        main_consistency_weight = float(
            getattr(
                self.cfg.BAYES_MMRL,
                "MAIN_CONSISTENCY_WEIGHT",
                0.0,
            )
        )
        main_consistency_term = main_consistency_weight * main_consistency

        total_loss = data_term + kl_terms["kl_term"] + main_consistency_term

        # ------------------------------------------------------------------
        # 4) Output aggregation, same as original full-MC path
        # ------------------------------------------------------------------
        logits_mean = logits_stack.mean(dim=0)
        logits_rep_mean = torch.stack(logits_rep_list, dim=0).mean(dim=0)
        logits_fusion_mean = torch.stack(logits_fusion_list, dim=0).mean(dim=0)
        image_features_mean = torch.stack(image_features_list, dim=0).mean(dim=0)
        text_features_mean = torch.stack(text_features_list, dim=0).mean(dim=0)
        text_features_mean = text_features_mean / text_features_mean.norm(
            dim=-1,
            keepdim=True,
        )

        losses = {
            "data_term": data_term,
            "main_consistency": main_consistency,
            "main_consistency_term": main_consistency_term,
            "main_consistency_weight": data_term.detach().new_tensor(
                main_consistency_weight
            ),
            "total": total_loss,
            **kl_terms,
        }

        return MethodOutputs(
            logits=logits_mean,
            labels=label,
            aux_logits={
                "rep": logits_rep_mean,
                "fusion": logits_fusion_mean,
            },
            features={
                "img": image_features_mean,
                "text": text_features_mean,
                "img_ref": img_ref,
                "text_ref": self.text_features_clip,
            },
            losses=losses,
            extras=self._maybe_posterior_stats(),
        )



    def _build_outputs_mean_main_mc_rep(self, label, img_ref, out):
        """
        Recommended objective: clean posterior-mean main/text branch + MC rep branch.

        Dependency structure:
          - loss_main, loss_cos_img, loss_cos_text depend only on R_mu / clean path.
          - loss_rep depends on sampled R_s or sampled proj_rep.
          - fusion uses clean main logits and posterior-predictive rep logits.
        """
        text_features = out["text_features"][: self.num_classes]
        logits_main = out["logits_main"]
        logits_rep = out["logits_rep"]
        logits_fusion = out["logits_fusion"]
        image_features_main = out["image_features_main"]

        loss_main = F.cross_entropy(logits_main, label)

        logits_rep_stack = out.get("logits_rep_stack")
        if logits_rep_stack is not None:
            loss_rep = torch.stack(
                [
                    F.cross_entropy(logits_rep_stack[s], label)
                    for s in range(logits_rep_stack.shape[0])
                ],
                dim=0,
            ).mean()
        else:
            loss_rep = F.cross_entropy(logits_rep, label)

        loss_cos_img = 1.0 - torch.mean(
            F.cosine_similarity(image_features_main, img_ref, dim=1)
        )
        loss_cos_text = 1.0 - torch.mean(
            F.cosine_similarity(text_features, self.text_features_clip, dim=1)
        )

        alpha = float(self.cfg.BAYES_MMRL.ALPHA)
        reg_weight = float(self.cfg.BAYES_MMRL.REG_WEIGHT)
        data_term = (
            alpha * loss_main
            + (1.0 - alpha) * loss_rep
            + reg_weight * loss_cos_img
            + reg_weight * loss_cos_text
        )

        kl_terms = self._kl_terms_for_loss(data_term)
        total_loss = data_term + kl_terms["kl_term"]

        losses = {
            "loss_main": loss_main,
            "loss_rep": loss_rep,
            "loss_cos_img": loss_cos_img,
            "loss_cos_text": loss_cos_text,
            "data_term": data_term,
            "total": total_loss,
            **kl_terms,
        }

        return MethodOutputs(
            logits=logits_main,
            labels=label,
            aux_logits={
                "rep": logits_rep,
                "fusion": logits_fusion,
            },
            features={
                "img": image_features_main,
                "text": text_features,
                "img_ref": img_ref,
                "text_ref": self.text_features_clip,
            },
            losses=losses,
            extras=self._maybe_posterior_stats(),
        )

    def _use_mean_main_mc_rep(self) -> bool:
        """
        Default to the recommended decoupled objective without requiring a new
        YACS config key. To run the old full-MC ablation, set an existing/allowed
        field in code or add BAYES_MMRL.USE_MEAN_MAIN_MC_REP=False to your config.
        """
        return bool(
            getattr(
                self.cfg.BAYES_MMRL,
                "USE_MEAN_MAIN_MC_REP",
                True,
            )
        )

    def forward_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)

        with torch.no_grad():
            img_ref = self.image_encoder_clip(image.type(self.dtype))
            img_ref = img_ref / img_ref.norm(dim=-1, keepdim=True)

        if self._use_mean_main_mc_rep():
            out = self.model.forward_mean_main_mc_rep(
                image,
                num_samples=self.n_mc_train,
                use_posterior_mean_for_rep=False,
                aggregation=str(
                    getattr(self.cfg.BAYES_MMRL, "EVAL_AGGREGATION", "prob_mean")
                ),
            )
            return self._build_outputs_mean_main_mc_rep(label, img_ref, out)

        sample_outputs = self.model.forward_train_samples(image, self.n_mc_train)
        return self._build_outputs_from_samples(label, img_ref, sample_outputs)


    def forward_eval(self, batch, eval_ctx):
        image = batch["img"].to(self.device)
        label = batch.get("label")
        if label is not None:
            label = label.to(self.device)

        if self._use_mean_main_mc_rep():
            out = self.model.forward_mean_main_mc_rep(
                image,
                num_samples=self.n_mc_test,
                use_posterior_mean_for_rep=self.eval_use_posterior_mean,
                aggregation=str(
                    getattr(self.cfg.BAYES_MMRL, "EVAL_AGGREGATION", "prob_mean")
                ),
            )
            text_features = out["text_features"][: self.num_classes]

            logits_main = out["logits_main"]
            logits_rep = out["logits_rep"]
            logits_static = out["logits_fusion"]

            aux_logits, fusion_extras = self._build_fusion_aux_logits(
                logits_main=logits_main,
                logits_rep=logits_rep,
                logits_static=logits_static,
            )

            extras = self._maybe_posterior_stats()
            extras.update(fusion_extras)

            return MethodOutputs(
                logits=logits_main,
                labels=label,
                aux_logits=aux_logits,
                features={
                    "img": out["image_features_main"],
                    "text": text_features,
                },
                extras=extras,
            )

        logits, logits_rep, logits_static, image_features, text_features = (
            self.model.forward_eval(
                image,
                num_samples=self.n_mc_test,
                use_posterior_mean=self.eval_use_posterior_mean,
            )
        )
        text_features = text_features[: self.num_classes]

        aux_logits, fusion_extras = self._build_fusion_aux_logits(
            logits_main=logits,
            logits_rep=logits_rep,
            logits_static=logits_static,
        )

        extras = self._maybe_posterior_stats()
        extras.update(fusion_extras)

        return MethodOutputs(
            logits=logits,
            labels=label,
            aux_logits=aux_logits,
            features={
                "img": image_features,
                "text": text_features,
            },
            extras=extras,
        )