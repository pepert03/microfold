"""FAPE loss (Frame-Aligned Point Error), backbone-only.

Following AF2 conventions: clamp at 10 A, scale by 10.
"""

from __future__ import annotations

import torch

from microfold.stamping import stamp_backbone


def fape(
    pred_R: torch.Tensor,           # [B, N, 3, 3]
    pred_t: torch.Tensor,           # [B, N, 3]
    true_R: torch.Tensor,           # [B, N, 3, 3]
    true_t: torch.Tensor,           # [B, N, 3]
    pred_points: torch.Tensor,      # [B, P, 3]
    true_points: torch.Tensor,      # [B, P, 3]
    frame_mask: torch.Tensor,       # [B, N]
    point_mask: torch.Tensor,       # [B, P]
    clamp: float = 10.0,
    scale: float = 10.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Mean clamped FAPE distance over all (frame i, point j) pairs."""
    # Local coords: R^T (x - t)
    # pred_local[b, i, j] = pred_R[b,i]^T @ (pred_points[b,j] - pred_t[b,i])
    pred_diff = pred_points.unsqueeze(1) - pred_t.unsqueeze(2)              # [B, N, P, 3]
    true_diff = true_points.unsqueeze(1) - true_t.unsqueeze(2)              # [B, N, P, 3]
    pred_local = torch.einsum("bnji,bnpj->bnpi", pred_R, pred_diff)         # R^T applied
    true_local = torch.einsum("bnji,bnpj->bnpi", true_R, true_diff)

    dist = (pred_local - true_local).pow(2).sum(dim=-1).clamp_min(eps).sqrt()  # [B, N, P]
    dist = dist.clamp_max(clamp)

    pair_mask = frame_mask.unsqueeze(-1) * point_mask.unsqueeze(-2)         # [B, N, P]
    num = (dist * pair_mask).sum()
    denom = pair_mask.sum().clamp_min(1.0)
    return (num / denom) / scale


def total_loss(
    intermediate: list[tuple[torch.Tensor, torch.Tensor]],
    final_R: torch.Tensor,
    final_t: torch.Tensor,
    true_R: torch.Tensor,
    true_t: torch.Tensor,
    true_all_backbone: torch.Tensor,    # [B, N, 3 atoms, 3]
    mask: torch.Tensor,                 # [B, N]
    clamp: float = 10.0,
    scale: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Average intermediate Cα FAPE + final all-atom FAPE."""
    # Intermediate: Cα only (point set == translation vectors)
    inter_losses = []
    for R_i, t_i in intermediate:
        L = fape(R_i, t_i, true_R, true_t, t_i, true_t, mask, mask, clamp, scale)
        inter_losses.append(L)
    inter = torch.stack(inter_losses).mean()

    # Final: stamp full backbone -> [B, N, 3 atoms, 3] -> [B, N*3, 3]
    pred_all = stamp_backbone(final_R, final_t)
    B, N, A, _ = pred_all.shape
    pred_pts = pred_all.reshape(B, N * A, 3)
    true_pts = true_all_backbone.reshape(B, N * A, 3)
    pt_mask = mask.unsqueeze(-1).expand(B, N, A).reshape(B, N * A)
    final = fape(final_R, final_t, true_R, true_t, pred_pts, true_pts, mask, pt_mask, clamp, scale)

    return {"intermediate": inter, "final": final, "total": inter + final}
