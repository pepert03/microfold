"""FAPE loss + structural-violation auxiliaries (peptide bond, steric clash).

Following AF2 conventions: clamp at 10 A, scale by 10. Structural losses use
flat-bottom L1 (zero penalty inside tolerance, linear outside).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from microfold.stamping import stamp_backbone

# Atom indices in stamp_backbone output (LOCAL_BACKBONE order: N, CA, C).
_ATOM_N, _ATOM_CA, _ATOM_C = 0, 1, 2
_ATOMS_PER_RES = 3


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
    pred_diff = pred_points.unsqueeze(1) - pred_t.unsqueeze(2)              # [B, N, P, 3]
    true_diff = true_points.unsqueeze(1) - true_t.unsqueeze(2)              # [B, N, P, 3]
    pred_local = torch.einsum("bnji,bnpj->bnpi", pred_R, pred_diff)
    true_local = torch.einsum("bnji,bnpj->bnpi", true_R, true_diff)

    dist = (pred_local - true_local).pow(2).sum(dim=-1).clamp_min(eps).sqrt()
    dist = dist.clamp_max(clamp)

    pair_mask = frame_mask.unsqueeze(-1) * point_mask.unsqueeze(-2)
    num = (dist * pair_mask).sum()
    denom = pair_mask.sum().clamp_min(1.0)
    return (num / denom) / scale


def peptide_bond_loss(
    pred_atoms: torch.Tensor,   # [B, N, 3 atoms, 3]
    mask: torch.Tensor,         # [B, N]
    d_lit: float = 1.328,
    tau: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Flat-bottom L1 on C_i -- N_{i+1} distance vs literature peptide bond length."""
    c_i = pred_atoms[:, :-1, _ATOM_C, :]    # [B, N-1, 3]
    n_ip1 = pred_atoms[:, 1:, _ATOM_N, :]   # [B, N-1, 3]
    dist = (c_i - n_ip1).pow(2).sum(dim=-1).clamp_min(eps).sqrt()  # [B, N-1]

    bond_mask = mask[:, :-1] * mask[:, 1:]                          # [B, N-1]
    penalty = F.relu((dist - d_lit).abs() - tau) * bond_mask
    denom = bond_mask.sum().clamp_min(1.0)
    return penalty.sum() / denom


def clash_loss(
    pred_atoms: torch.Tensor,   # [B, N, 3 atoms, 3]
    mask: torch.Tensor,         # [B, N]
    clash_limit: float = 2.0,
    neighbor_band: int = 1,
) -> torch.Tensor:
    """Flat-bottom L1 penalty for non-bonded atom pairs closer than clash_limit.

    Excludes atoms in the same residue and within `neighbor_band` residues
    (i, i-1, i+1 by default) — these are bonded or near-bonded and handled by
    the peptide-bond loss / fixed local geometry.
    """
    B, N, A, _ = pred_atoms.shape
    flat = pred_atoms.reshape(B, N * A, 3)                          # [B, L*A, 3]
    dist = torch.cdist(flat, flat)                                   # [B, L*A, L*A]

    res_idx = torch.arange(N, device=pred_atoms.device).repeat_interleave(A)  # [L*A]
    res_diff = (res_idx.unsqueeze(0) - res_idx.unsqueeze(1)).abs()             # [L*A, L*A]
    non_bonded = (res_diff > neighbor_band).to(dist.dtype)                     # [L*A, L*A]

    atom_mask = mask.repeat_interleave(A, dim=1)                               # [B, L*A]
    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)              # [B, L*A, L*A]
    pair_mask = pair_mask * non_bonded.unsqueeze(0)

    penalty = F.relu(clash_limit - dist) * pair_mask
    denom = pair_mask.sum().clamp_min(1.0)
    return penalty.sum() / denom


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
    w_bond: float = 0.1,
    w_clash: float = 0.05,
    bond_tol: float = 0.02,
    clash_limit: float = 2.0,
    use_clash: bool = False,
) -> dict[str, torch.Tensor]:
    """FAPE (intermediate Cα + final all-atom) plus structural violations."""
    inter_losses = []
    for R_i, t_i in intermediate:
        L = fape(R_i, t_i, true_R, true_t, t_i, true_t, mask, mask, clamp, scale)
        inter_losses.append(L)
    inter = torch.stack(inter_losses).mean()

    pred_all = stamp_backbone(final_R, final_t)                    # [B, N, 3, 3]
    B, N, A, _ = pred_all.shape
    pred_pts = pred_all.reshape(B, N * A, 3)
    true_pts = true_all_backbone.reshape(B, N * A, 3)
    pt_mask = mask.unsqueeze(-1).expand(B, N, A).reshape(B, N * A)
    final = fape(final_R, final_t, true_R, true_t, pred_pts, true_pts, mask, pt_mask, clamp, scale)

    bond = peptide_bond_loss(pred_all, mask, tau=bond_tol)
    if use_clash:
        clash = clash_loss(pred_all, mask, clash_limit=clash_limit)
    else:
        clash = torch.zeros((), device=final.device, dtype=final.dtype)

    total = inter + final + w_bond * bond + w_clash * clash
    return {
        "intermediate": inter,
        "final": final,
        "bond": bond,
        "clash": clash,
        "total": total,
    }
