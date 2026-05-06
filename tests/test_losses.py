from __future__ import annotations

import torch

from microfold.losses import fape, total_loss


def _toy_batch(B: int = 2, N: int = 6) -> dict[str, torch.Tensor]:
    R = torch.eye(3).expand(B, N, 3, 3).contiguous()
    t = torch.randn(B, N, 3, requires_grad=False)
    mask = torch.ones(B, N)
    return {"R": R, "t": t, "mask": mask}


def test_fape_zero_on_identical() -> None:
    b = _toy_batch()
    pts = b["t"]
    val = fape(b["R"], b["t"], b["R"], b["t"], pts, pts, b["mask"], b["mask"])
    assert val.item() < 1e-3


def test_total_loss_backward_populates_grads() -> None:
    from microfold.model import BackboneModel

    model = BackboneModel(n_layers=2)
    seqs = ["ACDEFG"]
    mask = torch.zeros(1, 30)
    mask[0, :6] = 1.0
    out = model(seqs, mask)

    true_R = torch.eye(3).expand(1, 30, 3, 3).contiguous()
    true_t = torch.randn(1, 30, 3)
    true_all = torch.randn(1, 30, 3, 3)

    losses = total_loss(out["intermediate"], out["R"], out["t"], true_R, true_t, true_all, mask)
    losses["total"].backward()

    grads = [p.grad for p in model.layers.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any((g.abs().sum() > 0).item() for g in grads)
