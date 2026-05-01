"""Backbone RMSD (Kabsch) and HTML 3D viewer.

Render style mirrors `code_that_works.py`: truth = green, prediction = red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py3Dmol
import torch

ATOM_NAMES = ("N", "CA", "C")


def _kabsch_align(mobile: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation R and translation t that minimize ||R @ mobile + t - fixed||^2."""
    cm = mobile.mean(axis=0)
    cf = fixed.mean(axis=0)
    M = mobile - cm
    F = fixed - cf
    H = M.T @ F
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cf - R @ cm
    return R, t


def calculate_backbone_rmsd(
    truth: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Kabsch-aligned RMSD over backbone atoms of real (mask=1) residues.

    Args:
        truth, pred: [N_res, 3 atoms, 3] tensors (atoms ordered N, CA, C).
        mask: [N_res] {0,1}.

    Returns:
        RMSD in Angstroms.
    """
    m = mask.bool().cpu().numpy()
    t_atoms = truth.detach().cpu().numpy()[m].reshape(-1, 3)
    p_atoms = pred.detach().cpu().numpy()[m].reshape(-1, 3)
    if len(t_atoms) == 0:
        return 0.0
    R, t = _kabsch_align(p_atoms, t_atoms)
    aligned = p_atoms @ R.T + t
    return float(np.sqrt(((aligned - t_atoms) ** 2).sum(axis=1).mean()))


def kabsch_superimpose(
    mobile: torch.Tensor,
    fixed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rigid-align `mobile` to `fixed` so they overlap maximally on masked positions.

    Computes the rotation/translation from masked atoms only, then applies it to ALL
    atoms (resolved + masked-out) so the returned tensor has the same shape.

    Args:
        mobile, fixed: [N_res, 3 atoms, 3].
        mask: [N_res] {0,1}.

    Returns:
        Aligned mobile tensor, same shape as input.
    """
    m = mask.bool().cpu().numpy()
    mob = mobile.detach().cpu().numpy()
    fix = fixed.detach().cpu().numpy()
    flat_mob = mob[m].reshape(-1, 3)
    flat_fix = fix[m].reshape(-1, 3)
    if len(flat_fix) == 0:
        return mobile
    R, t = _kabsch_align(flat_mob, flat_fix)
    aligned = mob.reshape(-1, 3) @ R.T + t
    return torch.from_numpy(aligned.reshape(mob.shape).astype(mob.dtype))


def _coords_to_pdb(coords: torch.Tensor, sequence: str, mask: torch.Tensor) -> str:
    """Build a minimal PDB string from backbone coords (N, CA, C per residue)."""
    lines: list[str] = []
    serial = 1
    coords = coords.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    for i, aa in enumerate(sequence):
        if i >= len(mask) or mask[i] < 0.5:
            continue
        for a, name in enumerate(ATOM_NAMES):
            x, y, z = coords[i, a]
            lines.append(
                f"ATOM  {serial:>5d}  {name:<3s} {'ALA':<3s} A{i + 1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {name[0]:>1s}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def generate_html_report(
    truth: torch.Tensor,
    pred: torch.Tensor,
    sequence: str,
    mask: torch.Tensor,
    out_path: Path,
    width: int = 800,
    height: int = 600,
) -> Path:
    """Write a standalone HTML viewer comparing truth (green) vs pred (red) backbones."""
    truth_pdb = _coords_to_pdb(truth, sequence, mask)
    pred_pdb = _coords_to_pdb(pred, sequence, mask)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(truth_pdb, "pdb")
    view.setStyle({"model": 0}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}})
    view.addModel(pred_pdb, "pdb")
    view.setStyle({"model": 1}, {"stick": {"colorscheme": "redCarbon", "radius": 0.15, "opacity": 0.7}})
    view.addStyle({"model": 1}, {"sphere": {"radius": 0.3, "opacity": 0.5}})
    view.zoomTo()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(view.write_html())
    return out_path
