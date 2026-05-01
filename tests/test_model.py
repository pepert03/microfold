from __future__ import annotations

import torch

from microfold.model import BackboneModel, _axis_angle_to_matrix


def test_axis_angle_orthonormal() -> None:
    aa = torch.randn(8, 3) * 0.5
    R = _axis_angle_to_matrix(aa)
    eye = torch.eye(3).expand(8, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8), atol=1e-5)


def test_forward_no_nan_and_orthonormal() -> None:
    model = BackboneModel(n_layers=2)
    seqs = ["ACDEFG", "MMM"]
    mask = torch.zeros(2, 30)
    mask[0, :6] = 1.0
    mask[1, :3] = 1.0
    out = model(seqs, mask)
    R = out["R"]
    t = out["t"]
    assert R.shape == (2, 30, 3, 3)
    assert t.shape == (2, 30, 3)
    assert not torch.isnan(R).any()
    assert not torch.isnan(t).any()
    R_real = torch.cat([R[0, :6], R[1, :3]], dim=0)
    eye = torch.eye(3).expand(R_real.shape[0], 3, 3)
    assert torch.allclose(R_real @ R_real.transpose(-1, -2), eye, atol=1e-4)
    assert len(out["intermediate"]) == 2
