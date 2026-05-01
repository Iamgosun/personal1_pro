from __future__ import annotations

import torch
import torch.nn.functional as F


class ClipAdaptersLoss:
    def __init__(self, cfg, adapter):
        self.cfg = cfg
        self.adapter = adapter
        self.base_kl_weight = float(cfg.CLIP_ADAPTERS.KL_WEIGHT)

    def _mc_supervised_ce(self, logits_all: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits_all.ndim != 3:
            raise ValueError(
                f"_mc_supervised_ce expects logits_all [S, B, C], got {tuple(logits_all.shape)}"
            )

        num_samples = int(logits_all.shape[0])
        flat_logits = logits_all.reshape(
            num_samples * logits_all.shape[1],
            logits_all.shape[2],
        )
        flat_labels = labels.unsqueeze(0).expand(num_samples, -1).reshape(-1)

        return F.cross_entropy(flat_logits, flat_labels, reduction="none").mean()


    def _btr_loss(self, outputs):
        logits_all = outputs.aux_logits["bayes_logits_all"]
        labels = outputs.labels

        if not hasattr(self.adapter, "btr_data_term"):
            raise AttributeError(
                "BTR loss requires adapter.btr_data_term(); "
                "check CLIP_ADAPTERS.INIT=BTR and adapter_router.py."
            )

        data_term, extras = self.adapter.btr_data_term(logits_all, labels)
        loss = data_term

        extras = {
            "loss_ce": data_term.detach(),
            "data_term": data_term.detach(),
            **{
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in extras.items()
            },
        }

        # KL(q(R) || p(R))
        kl = self.adapter.kl_divergence()
        kl_weight = float(
            outputs.extras.get(
                "bayes_kl_weight",
                self.adapter.bayes_kl_base_weight()
                if hasattr(self.adapter, "bayes_kl_base_weight")
                else self.base_kl_weight,
            )
        )
        kl_term = kl * kl_weight
        loss = loss + kl_term

        extras["loss_kl"] = kl_term.detach()
        extras["kl_term"] = kl_term.detach()
        extras["kl_raw"] = kl.detach()
        extras["kl_weight"] = data_term.detach().new_tensor(kl_weight)

        # Optional Brier calibration regularization
        brier_weight = float(getattr(self.adapter, "brier_weight", 0.0))
        if brier_weight > 0 and hasattr(self.adapter, "brier_loss"):
            brier_raw = self.adapter.brier_loss(logits_all, labels)
            brier_term = brier_weight * brier_raw
            loss = loss + brier_term

            extras["loss_brier"] = brier_term.detach()
            extras["brier_raw"] = brier_raw.detach()

        outputs.losses.update(extras)
        return loss


    def _hba_lr_loss(self, outputs):
        logits_all = outputs.aux_logits["bayes_logits_all"]
        labels = outputs.labels

        data_term = self._mc_supervised_ce(logits_all, labels)
        loss = data_term

        extras = {
            "loss_ce": data_term.detach(),
            "data_term": data_term.detach(),
        }

        # KL
        kl = self.adapter.kl_divergence()
        kl_weight = float(
            outputs.extras.get(
                "bayes_kl_weight",
                self.adapter.bayes_kl_base_weight()
                if hasattr(self.adapter, "bayes_kl_base_weight")
                else self.base_kl_weight,
            )
        )
        kl_term = kl * kl_weight
        loss = loss + kl_term

        extras["loss_kl"] = kl_term.detach()
        extras["kl_term"] = kl_term.detach()
        extras["kl_raw"] = kl.detach()
        extras["kl_weight"] = data_term.detach().new_tensor(kl_weight)

        # Basis regularization
        if hasattr(self.adapter, "basis_regularization"):
            basis_reg = self.adapter.basis_regularization()
            loss = loss + basis_reg
            extras["loss_basis_reg"] = basis_reg.detach()

        # Prototype anchor: keep mu_c close to original text prototype t_c
        lambda_proto = float(getattr(self.adapter, "lambda_proto_anchor", 0.0))
        if lambda_proto > 0 and hasattr(self.adapter, "prototype_anchor_regularization"):
            proto_anchor_raw = self.adapter.prototype_anchor_regularization()
            proto_anchor = lambda_proto * proto_anchor_raw
            loss = loss + proto_anchor

            extras["loss_proto_anchor"] = proto_anchor.detach()
            extras["proto_anchor_raw"] = proto_anchor_raw.detach()

        # Visual anchor: keep adapted image feature close to raw CLIP image feature
        lambda_visual = float(getattr(self.adapter, "lambda_visual_anchor", 0.0))
        raw_features = outputs.features.get("img_raw", None)
        adapted_features = outputs.features.get("img_adapted", None)

        if (
            lambda_visual > 0
            and raw_features is not None
            and adapted_features is not None
            and hasattr(self.adapter, "visual_anchor_regularization")
        ):
            visual_anchor_raw = self.adapter.visual_anchor_regularization(
                raw_features,
                adapted_features,
            )
            visual_anchor = lambda_visual * visual_anchor_raw
            loss = loss + visual_anchor

            extras["loss_visual_anchor"] = visual_anchor.detach()
            extras["visual_anchor_raw"] = visual_anchor_raw.detach()

        outputs.losses.update(extras)
        return loss

    def _bayes_adapter_loss(self, outputs):
        logits_all = outputs.aux_logits["bayes_logits_all"]
        labels = outputs.labels

        data_term = self._mc_supervised_ce(logits_all, labels)
        loss = data_term

        extras = {
            "loss_ce": data_term.detach(),
            "data_term": data_term.detach(),
        }

        if getattr(self.adapter, "apply_constraint", "none") != "none":
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras["loss_constraint"] = constraint.detach()

        kl = self.adapter.kl_divergence()
        kl_weight = float(outputs.extras.get("bayes_kl_weight", self.base_kl_weight))
        kl_term = kl * kl_weight

        loss = loss + kl_term
        extras["loss_kl"] = kl_term.detach()
        extras["kl_term"] = kl_term.detach()
        extras["kl_weight"] = data_term.detach().new_tensor(kl_weight)

        outputs.losses.update(extras)
        return loss

    def _deba_loss(self, outputs):
        """
        DEBA loss is intentionally isolated from the old BayesAdapter branch.

        The adapter may expose initialization_name="BAYES_ADAPTER" only to reuse
        the existing MC logits path in model.py. This loss branch is selected by
        deba_initialization_name == "DEBA", not by initialization_name.
        """
        logits_all = outputs.aux_logits["bayes_logits_all"]
        labels = outputs.labels

        if not hasattr(self.adapter, "deba_data_term"):
            raise AttributeError(
                "DEBA loss requires adapter.deba_data_term(); "
                "check CLIP_ADAPTERS.INIT=DEBA and adapter_router.py."
            )

        data_term, extras = self.adapter.deba_data_term(logits_all, labels)
        loss = data_term

        if getattr(self.adapter, "apply_constraint", "none") != "none":
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras["loss_constraint"] = constraint.detach()

        kl = self.adapter.kl_divergence()
        kl_weight = float(
            outputs.extras.get(
                "bayes_kl_weight",
                self.adapter.bayes_kl_base_weight()
                if hasattr(self.adapter, "bayes_kl_base_weight")
                else self.base_kl_weight,
            )
        )
        kl_term = kl * kl_weight

        loss = loss + kl_term
        extras["loss_kl"] = kl_term.detach()
        extras["kl_term"] = kl_term.detach()
        extras["kl_raw"] = kl.detach()
        extras["kl_weight"] = data_term.detach().new_tensor(kl_weight)

        outputs.losses.update(extras)
        return loss

    def _capel_loss(self, outputs):
        loss_ce = F.cross_entropy(outputs.logits, outputs.labels)

        if "capel_sub_logits" not in outputs.aux_logits:
            raise KeyError(
                "CAPEL loss requires outputs.aux_logits['capel_sub_logits']; "
                "check ClipAdaptersModel.forward_features()/forward_train()."
            )

        sub_logits = outputs.aux_logits["capel_sub_logits"]
        loss_pc = self.adapter.pc_loss(sub_logits, outputs.labels)

        lam = float(getattr(self.cfg.CLIP_ADAPTERS, "CAPEL_PC_LAMBDA", 3.0))
        loss = loss_ce + lam * loss_pc

        extras = {
            "loss_ce": loss_ce.detach(),
            "loss_pc": loss_pc.detach(),
            "loss_capel_pc": (lam * loss_pc).detach(),
        }

        if getattr(self.adapter, "apply_constraint", "none") != "none":
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras["loss_constraint"] = constraint.detach()

        outputs.losses.update(extras)
        return loss

    def _sbea_loss(self, outputs):
        logits_all = outputs.aux_logits["bayes_logits_all"]
        labels = outputs.labels

        if not hasattr(self.adapter, "mc_predictive_log_probs"):
            raise AttributeError(
                "SBEA loss requires adapter.mc_predictive_log_probs()."
            )

        log_probs = self.adapter.mc_predictive_log_probs(logits_all)
        data_term = F.nll_loss(log_probs, labels, reduction="mean")

        loss = data_term

        extras = {
            "loss_ce": data_term.detach(),
            "data_term": data_term.detach(),
        }

        kl = self.adapter.kl_divergence()

        kl_weight = float(
            outputs.extras.get(
                "bayes_kl_weight",
                self.adapter.bayes_kl_base_weight()
                if hasattr(self.adapter, "bayes_kl_base_weight")
                else self.base_kl_weight,
            )
        )

        kl_term = kl * kl_weight
        loss = loss + kl_term

        extras["loss_kl"] = kl_term.detach()
        extras["kl_term"] = kl_term.detach()
        extras["kl_raw"] = kl.detach()
        extras["kl_weight"] = data_term.detach().new_tensor(kl_weight)

        hyper_weight = float(
            getattr(self.cfg.CLIP_ADAPTERS, "SBEA_ARD_HYPER_WEIGHT", 1.0e-4)
        )
        if hyper_weight > 0 and hasattr(self.adapter, "ard_hyper_regularization"):
            hyper_raw = self.adapter.ard_hyper_regularization()
            hyper_term = hyper_weight * hyper_raw
            loss = loss + hyper_term

            extras["sbea_ard_hyper_raw"] = hyper_raw.detach()
            extras["sbea_ard_hyper_term"] = hyper_term.detach()

        proto_anchor_weight = float(
            getattr(self.cfg.CLIP_ADAPTERS, "SBEA_PROTO_ANCHOR_WEIGHT", 0.0)
        )
        if proto_anchor_weight > 0 and hasattr(
            self.adapter, "prototype_anchor_regularization"
        ):
            proto_anchor_raw = self.adapter.prototype_anchor_regularization()
            proto_anchor_term = proto_anchor_weight * proto_anchor_raw
            loss = loss + proto_anchor_term

            extras["sbea_proto_anchor_raw"] = proto_anchor_raw.detach()
            extras["sbea_proto_anchor_term"] = proto_anchor_term.detach()

        if hasattr(self.adapter, "sbea_uncertainty"):
            with torch.no_grad():
                unc = self.adapter.sbea_uncertainty(logits_all)

            for key, value in unc.items():
                if torch.is_tensor(value):
                    extras[f"sbea_{key}_mean"] = value.float().mean().detach()

        outputs.losses.update(extras)
        return loss


    def _vnccapel_loss(self, outputs):
        loss_ce = F.cross_entropy(outputs.logits, outputs.labels)

        if "capel_sub_logits" not in outputs.aux_logits:
            raise KeyError(
                "VNC-CAPEL loss requires outputs.aux_logits['capel_sub_logits']; "
                "check ClipAdaptersModel.forward_features()."
            )

        if "vnc_assignment_logits" not in outputs.aux_logits:
            raise KeyError(
                "VNC-CAPEL loss requires outputs.aux_logits['vnc_assignment_logits']; "
                "check ClipAdaptersModel.forward_features()."
            )

        if "img" not in outputs.features:
            raise KeyError(
                "VNC-CAPEL loss requires outputs.features['img'] as CLIP image features."
            )

        assignment_logits = outputs.aux_logits["vnc_assignment_logits"]
        image_features = outputs.features["img"]

        loss_pc = self.adapter.pc_loss(assignment_logits, outputs.labels)
        loss_vnc = self.adapter.vnc_loss(
            image_features=image_features,
            labels=outputs.labels,
            assignment_logits=assignment_logits,
        )

        lambda_pc = float(getattr(self.cfg.CLIP_ADAPTERS, "CAPEL_PC_LAMBDA", 3.0))
        lambda_vnc = float(getattr(self.cfg.CLIP_ADAPTERS, "VNC_CAPEL_VNC_LAMBDA", 0.2))

        loss = loss_ce + lambda_pc * loss_pc + lambda_vnc * loss_vnc

        extras = {
            "loss_ce": loss_ce.detach(),
            "loss_pc": loss_pc.detach(),
            "loss_capel_pc": (lambda_pc * loss_pc).detach(),
            "loss_vnc": loss_vnc.detach(),
            "loss_vnc_weighted": (lambda_vnc * loss_vnc).detach(),
        }

        if getattr(self.adapter, "apply_constraint", "none") != "none":
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras["loss_constraint"] = constraint.detach()

        outputs.losses.update(extras)
        return loss

    def __call__(self, outputs):
        if (
            str(getattr(self.adapter, "initialization_name", "")).upper() == "CAPEL"
            and "capel_sub_logits" in outputs.aux_logits
        ):
            return self._capel_loss(outputs)

        if (
            str(getattr(self.adapter, "btr_initialization_name", "")).upper() == "BTR"
            and "bayes_logits_all" in outputs.aux_logits
        ):
            return self._btr_loss(outputs)


        if (
            str(getattr(self.adapter, "hba_initialization_name", "")).upper() == "HBA_LR"
            and "bayes_logits_all" in outputs.aux_logits
        ):
            return self._hba_lr_loss(outputs)

        # DEBA is a separate method branch. It is checked before the generic
        # BayesAdapter branch because DEBAAdapter reuses initialization_name
        # "BAYES_ADAPTER" only for model.py MC routing compatibility.
        if (
            str(getattr(self.adapter, "deba_initialization_name", "")).upper() == "DEBA"
            and "bayes_logits_all" in outputs.aux_logits
        ):
            return self._deba_loss(outputs)

        if (
            str(getattr(self.adapter, "sbea_initialization_name", "")).upper() == "SBEA"
            and "bayes_logits_all" in outputs.aux_logits
        ):
            return self._sbea_loss(outputs)


        if (
            str(getattr(self.adapter, "initialization_name", "")).upper() == "BAYES_ADAPTER"
            and "bayes_logits_all" in outputs.aux_logits
        ):
            return self._bayes_adapter_loss(outputs)

        loss = F.cross_entropy(outputs.logits, outputs.labels)
        extras = {"loss_ce": loss.detach()}

        if getattr(self.adapter, "apply_constraint", "none") != "none":
            constraint = self.adapter.zero_shot_constraint()
            loss = loss + constraint
            extras["loss_constraint"] = constraint.detach()

        if hasattr(self.adapter, "kl_divergence"):
            kl_weight = getattr(self.adapter, "kl_weight", self.base_kl_weight)
            kl = self.adapter.kl_divergence() * kl_weight
            loss = loss + kl
            extras["loss_kl"] = kl.detach()

        if hasattr(self.adapter, "extra_loss"):
            extra = self.adapter.extra_loss()
            if extra is not None:
                loss = loss + extra
                extras["loss_extra"] = extra.detach()

        outputs.losses.update(extras)
        return loss