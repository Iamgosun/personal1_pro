from __future__ import annotations

import math

import torch

from .bayes_adapter import BayesAdapter


class RCVBayesAdapter(BayesAdapter):
    """
    Reliability-Calibrated Variance BayesAdapter.

    Key idea:
        Keep CLIP text prototype direction fixed:
            p(w_c) = N(t_c, sigma_c^2 I)

        Only calibrate class-wise prior std sigma_c using support-set reliability.

    This avoids moving prototype means, which was unstable in HBA/ECKA-style variants.
    """

    # Keep BayesAdapter routing, MC logits, and loss behavior.
    initialization_name = "BAYES_ADAPTER"
    rcv_initialization_name = "RCV_BAYES_ADAPTER"
    adapter_kind = "stochastic_prototype"

    # This makes ClipAdaptersMethod collect support features and call build_cache().
    needs_support_features = True

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @torch.no_grad()
    def build_cache(self, features_train: torch.Tensor, labels_train: torch.Tensor) -> None:
        cad = self.cfg.CLIP_ADAPTERS
        device = self.prior_mean.device
        dtype = self.prior_mean.dtype

        x = features_train.detach().to(device=device, dtype=torch.float32)
        y = labels_train.detach().to(device=device, dtype=torch.long).flatten()

        x = self._normalize(x)
        text = self._normalize(self.prior_mean.detach().float().to(device))

        valid = (y >= 0) & (y < self.num_classes)
        x = x[valid]
        y = y[valid]

        if x.numel() == 0:
            print("[RCV-BayesAdapter] no valid support features; keep original prior std")
            return None

        base_prior_std = float(getattr(cad, "BAYES_PRIOR_STD", 0.01))

        min_std = float(getattr(cad, "RCV_PRIOR_MIN_STD", -1.0))
        max_std = float(getattr(cad, "RCV_PRIOR_MAX_STD", -1.0))

        # Conservative default:
        #   reliable classes keep BayesAdapter std;
        #   unreliable-but-visually-stable classes get weaker prior.
        if min_std <= 0:
            min_std = base_prior_std
        if max_std <= 0:
            max_std = 5.0 * base_prior_std

        if min_std <= 0 or max_std <= 0 or max_std < min_std:
            raise ValueError(
                f"Invalid RCV prior std range: min={min_std}, max={max_std}"
            )

        shot_kappa = float(getattr(cad, "RCV_SHOT_KAPPA", 4.0))
        compact_tau = float(getattr(cad, "RCV_COMPACT_TAU", 0.05))
        margin_tau_cfg = float(getattr(cad, "RCV_MARGIN_TAU", -1.0))
        debug = bool(getattr(cad, "RCV_DEBUG", True))

        logits = x @ text.t()

        row = torch.arange(y.numel(), device=device)
        true_logits = logits[row, y]
        masked_logits = logits.clone()
        masked_logits[row, y] = -1.0e30
        margins = true_logits - masked_logits.max(dim=1).values

        if margin_tau_cfg > 0:
            margin_tau = torch.tensor(margin_tau_cfg, device=device, dtype=torch.float32)
        else:
            # Robust scale; fallback avoids extreme sigmoid saturation.
            med = margins.median()
            mad = (margins - med).abs().median()
            margin_tau = (1.4826 * mad).clamp_min(1.0e-3)

        prior_std = torch.full(
            (self.num_classes,),
            base_prior_std,
            device=device,
            dtype=torch.float32,
        )

        text_rel_all = torch.zeros(self.num_classes, device=device, dtype=torch.float32)
        vis_rel_all = torch.zeros(self.num_classes, device=device, dtype=torch.float32)
        shot_rel_all = torch.zeros(self.num_classes, device=device, dtype=torch.float32)
        unreliability_all = torch.zeros(self.num_classes, device=device, dtype=torch.float32)
        counts_all = torch.zeros(self.num_classes, device=device, dtype=torch.float32)

        for c in range(self.num_classes):
            idx = (y == c).nonzero(as_tuple=False).flatten()
            n_c = int(idx.numel())
            counts_all[c] = float(n_c)

            if n_c <= 0:
                continue

            x_c = x.index_select(0, idx)

            margin_c = margins.index_select(0, idx).mean()
            text_rel = torch.sigmoid(margin_c / margin_tau)

            v_c = self._normalize(x_c.mean(dim=0, keepdim=True)).squeeze(0)
            compact = (1.0 - (x_c @ v_c).clamp(-1.0, 1.0)).mean()
            vis_rel = torch.exp(-compact / max(compact_tau, 1.0e-6))

            shot_rel = torch.tensor(
                float(n_c) / (float(n_c) + max(shot_kappa, 1.0e-6)),
                device=device,
                dtype=torch.float32,
            )

            # Only relax the prior when:
            #   text prototype is unreliable,
            #   visual support is compact,
            #   and there is enough support evidence.
            unreliability = (1.0 - text_rel) * vis_rel * shot_rel
            unreliability = unreliability.clamp(0.0, 1.0)

            prior_std[c] = min_std + (max_std - min_std) * unreliability

            text_rel_all[c] = text_rel
            vis_rel_all[c] = vis_rel
            shot_rel_all[c] = shot_rel
            unreliability_all[c] = unreliability

        prior_logstd = prior_std.clamp_min(1.0e-8).log().to(dtype=dtype)

        # Update p(W). prior_mean stays text prototype.
        self.prior_logstd.data.copy_(prior_logstd)

        # Keep q0 approximately p after support reliability fitting.
        # Otherwise build_cache would create a nonzero initial KL.
        self.text_features_unnorm_logstd.data.copy_(prior_logstd)

        # Optional debug buffers.
        self.register_buffer("rcv_prior_std", prior_std.detach().to(dtype=dtype), persistent=False)
        self.register_buffer("rcv_text_reliability", text_rel_all.detach().to(dtype=dtype), persistent=False)
        self.register_buffer("rcv_visual_reliability", vis_rel_all.detach().to(dtype=dtype), persistent=False)
        self.register_buffer("rcv_shot_reliability", shot_rel_all.detach().to(dtype=dtype), persistent=False)
        self.register_buffer("rcv_unreliability", unreliability_all.detach().to(dtype=dtype), persistent=False)
        self.register_buffer("rcv_counts", counts_all.detach().to(dtype=dtype), persistent=False)

        if debug:
            print(
                "[RCV-BayesAdapter] fitted class-wise prior std: "
                f"support={int(x.shape[0])}, "
                f"std_min={float(prior_std.min().cpu()):.6g}, "
                f"std_med={float(prior_std.median().cpu()):.6g}, "
                f"std_max={float(prior_std.max().cpu()):.6g}, "
                f"margin_tau={float(margin_tau.cpu()):.6g}, "
                f"base_prior_std={base_prior_std:.6g}"
            )

        return None