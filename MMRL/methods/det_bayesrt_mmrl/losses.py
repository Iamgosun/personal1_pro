from __future__ import annotations

import torch


def det_jensen_ce(
    mu: torch.Tensor,
    var: torch.Tensor,
    label: torch.Tensor,
    var_scale: float = 0.5,
    var_clamp: float = 20.0,
) -> torch.Tensor:
    """
    Deterministic Jensen CE:
        -mu_y + logsumexp(mu_c + 0.5 * eta * var_c)

    This is the sampling-free replacement for MC expected CE:
        mean_s CE(z^(s), y)

    Args:
        mu:
            [B, C] posterior mean logits.
        var:
            [B, C] approximate posterior logit variances.
        label:
            [B] integer labels.
        var_scale:
            eta, scales the variance correction term.
        var_clamp:
            upper clamp for logit variance. Use <= 0 to disable.

    Returns:
        Scalar loss.
    """
    var = var.float().clamp_min(0.0)

    if float(var_clamp) > 0:
        var = var.clamp_max(float(var_clamp))

    psi = mu.float() + 0.5 * float(var_scale) * var
    target_mu = mu.float().gather(1, label[:, None]).squeeze(1)

    return (-target_mu + torch.logsumexp(psi, dim=1)).mean()
