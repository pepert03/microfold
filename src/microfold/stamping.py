"""Convert rigid frames (R, t) into N/CA/C atom coordinates via parallel matrix multiply.

Replaces NeRF: every residue's local backbone is the same idealized triangle, so we
just rotate-and-translate it by each residue's frame in one vectorized op.
"""

from __future__ import annotations

import torch

# AF2-style idealized local backbone (Angstroms), with CA at origin.
# Atom order: N, CA, C.
LOCAL_BACKBONE = torch.tensor(
    [
        [-0.525,  1.363,  0.000],   # N
        [ 0.000,  0.000,  0.000],   # CA
        [ 1.526,  0.000,  0.000],   # C
    ],
    dtype=torch.float32,
)


def stamp_backbone(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Stamp the idealized backbone triangle onto each residue frame.

    Args:
        R: rotation tensor [..., N, 3, 3] (columns = local axes).
        t: translation tensor [..., N, 3].

    Returns:
        Tensor [..., N, 3, 3]: per-residue (atom × xyz) world coordinates, atoms ordered N, CA, C.
    """
    local = LOCAL_BACKBONE.to(R.device, R.dtype)            # [3 atoms, 3 xyz]
    # world_xyz = R @ local^T (per-atom column) + t
    # einsum: R[..., N, i, j] * local[a, j] -> world[..., N, a, i]
    world = torch.einsum("...nij,aj->...nai", R, local) + t.unsqueeze(-2)
    return world
