from __future__ import annotations

import torch
import torch.nn.functional as F

from .capel_adapter import CapelAdapter


class VncCapelAdapter(CapelAdapter):
    """
    VNC-CAPEL: CAPEL + training-only Visual Neighborhood Consistency.

    注意：
    - 不修改原始 CapelAdapter。
    - 继承 CAPEL 的 prompt bank、prototype cache、prompt weights。
    - 推理阶段仍然走 CAPEL logit-space prompt ensemble。
    - VNC loss 只在训练 loss.py 中启用。
    """

    initialization_name = "VNC_CAPEL"
    adapter_kind = "vnccapel_prototype"

    def __init__(self, cfg, clip_model, base_text_features: torch.Tensor, classnames):
        super().__init__(cfg, clip_model, base_text_features, classnames)

        cad = cfg.CLIP_ADAPTERS
        self.vnc_lambda = float(getattr(cad, "VNC_CAPEL_VNC_LAMBDA", 0.2))

        print(
            "[VNC-CAPEL] initialized: "
            f"lambda_vnc={self.vnc_lambda}, "
            "assignment_logits=unscaled_cosine",
            flush=True,
        )

    def vnc_loss(
        self,
        image_features: torch.Tensor,
        labels: torch.Tensor,
        assignment_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Visual Neighborhood Consistency loss.

        image_features:
            [B, D], frozen CLIP visual features.

        labels:
            [B].

        assignment_logits:
            [B, C, K], 建议使用未乘 CLIP logit_scale 的 cosine logits。
            对真实类别 y_i 的 K 个 prompt 做 softmax 得到 q_i。

        返回:
            scalar tensor.
            如果 batch 内没有有效同类正相似邻居，则返回 differentiable zero。
        """
        if image_features.ndim != 2:
            raise ValueError(
                f"VNC image_features must be [B, D], got {tuple(image_features.shape)}"
            )

        if assignment_logits.ndim != 3:
            raise ValueError(
                "VNC assignment_logits must be [B, C, K], "
                f"got {tuple(assignment_logits.shape)}"
            )

        labels = labels.reshape(-1).to(
            device=assignment_logits.device,
            dtype=torch.long,
        )

        if image_features.shape[0] != labels.shape[0]:
            raise ValueError(
                "VNC image/label batch mismatch: "
                f"features={tuple(image_features.shape)}, labels={tuple(labels.shape)}"
            )

        if assignment_logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "VNC assignment/label batch mismatch: "
                f"assignment={tuple(assignment_logits.shape)}, labels={tuple(labels.shape)}"
            )

        if self.vnc_lambda <= 0.0:
            return assignment_logits.sum() * 0.0

        eps = 1e-8

        image_features = F.normalize(image_features.float(), dim=-1)
        q_all = F.softmax(assignment_logits.float(), dim=-1)

        valid_kl = []

        for y in labels.unique(sorted=False):
            idx = torch.where(labels == y)[0]

            if idx.numel() <= 1:
                continue

            y_int = int(y.item())

            x_y = image_features.index_select(0, idx)      # [n, D]
            q_y = q_all.index_select(0, idx)[:, y_int, :]  # [n, K]

            sim = torch.clamp(x_y @ x_y.t(), min=0.0)      # [n, n]

            n = sim.shape[0]
            sim = sim * (
                1.0 - torch.eye(n, device=sim.device, dtype=sim.dtype)
            )

            denom = sim.sum(dim=-1, keepdim=True)
            valid = denom.squeeze(-1) > eps

            if not bool(valid.any()):
                continue

            pi = sim / denom.clamp_min(eps)                # [n, n]

            # stop-gradient teacher
            q_teacher = pi @ q_y.detach()                  # [n, K]

            kl = (
                q_teacher
                * (
                    torch.log(q_teacher.clamp_min(eps))
                    - torch.log(q_y.clamp_min(eps))
                )
            ).sum(dim=-1)

            valid_kl.append(kl[valid])

        if not valid_kl:
            return assignment_logits.sum() * 0.0

        return torch.cat(valid_kl, dim=0).mean()