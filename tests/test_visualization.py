from __future__ import annotations

from pathlib import Path

import torch

from microfold.visualization import calculate_backbone_rmsd, generate_html_report


def test_rmsd_zero_on_identical() -> None:
    coords = torch.randn(30, 3, 3)
    mask = torch.zeros(30)
    mask[:10] = 1.0
    rmsd = calculate_backbone_rmsd(coords, coords, mask)
    assert rmsd < 1e-6


def test_rmsd_invariant_under_rigid_motion() -> None:
    coords = torch.randn(30, 3, 3)
    mask = torch.zeros(30)
    mask[:8] = 1.0
    moved = coords + torch.tensor([5.0, -3.0, 2.0])
    rmsd = calculate_backbone_rmsd(coords, moved, mask)
    assert rmsd < 1e-5


def test_kabsch_superimpose_recovers_rotated_copy() -> None:
    from microfold.visualization import kabsch_superimpose

    truth = torch.randn(30, 3, 3)
    mask = torch.zeros(30); mask[:7] = 1.0
    # Rotate + translate the truth to make a "prediction" then align it back.
    angle = 1.2
    Rz = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
                       [torch.sin(torch.tensor(angle)),  torch.cos(torch.tensor(angle)), 0.0],
                       [0.0, 0.0, 1.0]])
    pred_flat = (truth.reshape(-1, 3) @ Rz.T) + torch.tensor([7.0, -3.0, 2.0])
    pred = pred_flat.reshape(30, 3, 3)
    aligned = kabsch_superimpose(pred, truth, mask)
    # Resolved residues should now match the truth.
    diff = (aligned[:7] - truth[:7]).abs().max()
    assert diff.item() < 1e-4


def test_html_writes_file(tmp_path: Path) -> None:
    coords = torch.randn(30, 3, 3)
    mask = torch.zeros(30)
    mask[:5] = 1.0
    out = generate_html_report(coords, coords, "AAAAA", mask, tmp_path / "v.html")
    assert out.exists() and out.stat().st_size > 1000
