from __future__ import annotations

import torch

from microfold.geometry import cache_frames
from microfold.stamping import stamp_backbone

DBAASP_ID = 11
SEQUENCE = "RVKRVWPLVIRTVIAGYNLYRAIKKK"


def test_stamp_recovers_real_backbone() -> None:
    p = cache_frames(DBAASP_ID, SEQUENCE)
    rec = torch.load(p, weights_only=False)

    n = int(rec["mask"].sum().item())
    R = rec["R"]
    t = rec["t"]

    pred = stamp_backbone(R, t)              # [30, 3 atoms, 3]
    truth = rec["true_all_backbone_coords"]  # [30, 3 atoms, 3]

    diff = (pred[:n] - truth[:n]).abs()
    # Real residues deviate from idealized bond lengths; AF2 uses similar tolerance.
    assert diff.max().item() < 0.2, f"max diff {diff.max().item():.4f} > 0.2 A"
    # CA must be exact (CA == t in our convention).
    ca_diff = (pred[:n, 1] - truth[:n, 1]).abs().max()
    assert ca_diff.item() < 1e-5


def test_stamp_batched_shapes() -> None:
    R = torch.eye(3).expand(2, 5, 3, 3).contiguous()
    t = torch.zeros(2, 5, 3)
    out = stamp_backbone(R, t)
    assert out.shape == (2, 5, 3, 3)
    # Identity frame at origin should reproduce LOCAL_BACKBONE for every residue
    expected = torch.tensor(
        [[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0]]
    )
    assert torch.allclose(out[0, 0], expected, atol=1e-5)
