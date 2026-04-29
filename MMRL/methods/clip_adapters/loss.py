from __future__ import annotations

import torch
import torch.nn.functional as F


class ClipAdaptersLoss:
    def __init__(self, cfg, adapter):
        self.cfg = cfg
        self.adapter = adapter
        self.base_kl_weight = float(cfg.CLIP_ADAPTERS.KL_WEIGHT)

    def _bayes_adapter_loss(self, outputs):
        logits_all = outputs.aux_logits["bayes_logits_all"]  # [S, B, C]
        labels = outputs.labels  # [B]

        num_samples = int(logits_all.shape[0])

        # Equivalent to the uploaded code:
        # cross_entropy(logits.permute(1,2,0), labels.repeat(...)).mean()
        flat_logits = logits_all.reshape(num_samples * logits_all.shape[1], logits_all.shape[2])
        flat_labels = labels.unsqueeze(0).expand(num_samples, -1).reshape(-1)

        data_term = F.cross_entropy(flat_logits, flat_labels, reduction="none").mean()
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


    def _capel_loss(self, outputs):
        loss_ce = F.cross_entropy(outputs.logits, outputs.labels)

        if "capel_sub_logits" not in outputs.aux_logits:
            raise KeyError(
                "CAPEL loss requires outputs.aux_logits['capel_sub_logits']; "
                "check ClipAdaptersModel.forward_features()/forward_train()."
            )

        # Paper Eq.10:
        # P(k | x_i, y) corresponds to logits Z_y^k after Softmax.
        # Here capel_sub_logits is exactly Z_yk = tau * cos(x, w_yk).
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


    def _vnccapel_loss(self, outputs):
        loss_ce = F.cross_entropy(outputs.logits, outputs.labels)

        if "capel_sub_logits" not in outputs.aux_logits:
            raise KeyError(
                "VNC-CAPEL loss requires outputs.aux_logits['capel_sub_logits']; "
                "check ClipAdaptersModel.forward_features()/forward_train()."
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

        sub_logits = outputs.aux_logits["capel_sub_logits"]
        assignment_logits = outputs.aux_logits["vnc_assignment_logits"]
        image_features = outputs.features["img"]

        # CAPEL CE 仍然来自 scaled logits。
        # L_pc 和 L_vnc 使用 unscaled assignment logits，更稳定。
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

        outputs.losses.update(extras)
        return loss
