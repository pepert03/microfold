"""Phase 1 capstone: stamp noisy frames, compare to truth, write HTML viewer.

Run:
    uv run python -m microfold.simulate --dbaasp-id 11
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from microfold.geometry import cache_frames
from microfold.stamping import stamp_backbone
from microfold.visualization import calculate_backbone_rmsd, generate_html_report, kabsch_superimpose


def _random_small_rotation(n: int, max_angle_rad: float = 0.3) -> torch.Tensor:
    """Per-residue random small rotation via axis-angle, returned as [n, 3, 3]."""
    axis = torch.randn(n, 3)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.empty(n).uniform_(-max_angle_rad, max_angle_rad)
    K = torch.zeros(n, 3, 3)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2]; K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] =  axis[:, 0]
    I = torch.eye(3).expand(n, 3, 3)
    s = angle.view(n, 1, 1)
    return I + torch.sin(s) * K + (1 - torch.cos(s)) * (K @ K)


def simulate(dbaasp_id: int, sigma_t: float = 0.5, sigma_r: float = 0.3) -> Path:
    index = pd.read_csv(Path("data_cache/index.csv"))
    row = index[index["ID"] == dbaasp_id]
    if row.empty:
        raise SystemExit(f"DBAASP id {dbaasp_id} not in data_cache/index.csv")
    seq = str(row.iloc[0]["SEQUENCE"])

    p = cache_frames(dbaasp_id, seq)
    rec = torch.load(p, weights_only=False)
    R, t, mask = rec["R"], rec["t"], rec["mask"]
    truth = rec["true_all_backbone_coords"]

    n = int(mask.sum().item())
    dR = _random_small_rotation(R.shape[0], sigma_r)
    R_noisy = R @ dR
    t_noisy = t + sigma_t * torch.randn_like(t) * mask.unsqueeze(-1)

    pred = stamp_backbone(R_noisy, t_noisy)

    rmsd = calculate_backbone_rmsd(truth, pred, mask)
    aligned_pred = kabsch_superimpose(pred, truth, mask)
    out = Path(f"outputs/sim_{dbaasp_id}.html")
    generate_html_report(truth, aligned_pred, seq[:n], mask, out)
    print(f"DBAASP {dbaasp_id} (len {n}): RMSD = {rmsd:.3f} A   wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbaasp-id", type=int, default=11)
    ap.add_argument("--sigma-t", type=float, default=0.5)
    ap.add_argument("--sigma-r", type=float, default=0.3)
    args = ap.parse_args()
    simulate(args.dbaasp_id, args.sigma_t, args.sigma_r)


if __name__ == "__main__":
    main()
