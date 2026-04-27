"""
Drifting field with optional Analytical Bias Correction (ABC).

Computes V(x) = ybar^+(x) - ybar^-(x) using bidirectional kernel
normalization (the form used in the official Drifting demo).  When ``abc``
is True, both centroids are replaced by their ABC-corrected counterparts
``T_abc = (1 - H) * T + sum(alpha^2 * y)`` with H = sum(alpha^2).
"""

import torch


def _abc_centroid(alpha: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """ABC-corrected centroid using the Herfindahl form (no extra memory)."""
    alpha_sq = alpha ** 2
    H = alpha_sq.sum(dim=1, keepdim=True)
    T_n = torch.mm(alpha, features)
    T_sq = torch.mm(alpha_sq, features)
    return (1.0 - H) * T_n + T_sq


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperature: float,
    mask_self: bool = True,
    abc: bool = False,
) -> torch.Tensor:
    """
    Compute the drifting field V at query points x using bidirectional
    kernel normalization sqrt(row_sum * col_sum).  Returns a tensor of
    shape (N, D) where N = x.shape[0] and D = y_pos.shape[1].
    """
    N = x.shape[0]
    N_neg = y_neg.shape[0]
    device = x.device

    targets = torch.cat([y_neg, y_pos], dim=0)
    dist = torch.cdist(x, targets, p=2)

    if mask_self and N == N_neg:
        eye = torch.eye(N, N_neg, dtype=torch.bool, device=device)
        N_pos = dist.shape[1] - N_neg
        mask = torch.cat(
            [eye, torch.zeros(N, N_pos, dtype=torch.bool, device=device)], dim=1
        )
        dist = dist.masked_fill(mask, 1e6)

    kernel = (-dist / temperature).exp()
    row_sum = kernel.sum(dim=-1, keepdim=True)
    col_sum = kernel.sum(dim=-2, keepdim=True)
    K_norm = kernel / (row_sum * col_sum).clamp_min(1e-12).sqrt()

    S_neg = K_norm[:, :N_neg].sum(dim=-1, keepdim=True)
    S_pos = K_norm[:, N_neg:].sum(dim=-1, keepdim=True)

    if abc:
        alpha_pos = K_norm[:, N_neg:] / S_pos.clamp_min(1e-12)
        alpha_neg = K_norm[:, :N_neg] / S_neg.clamp_min(1e-12)

        pos_centroid = _abc_centroid(alpha_pos, y_pos)
        neg_centroid = _abc_centroid(alpha_neg, y_neg)

        pos_V = S_pos * S_neg * pos_centroid
        neg_V = S_neg * S_pos * neg_centroid
    else:
        # Product-form V matching the official demo.
        pos_V = (K_norm[:, N_neg:] * S_neg) @ y_pos
        neg_V = (K_norm[:, :N_neg] * S_pos) @ y_neg

    return pos_V - neg_V
