"""Re-render a validation report from a saved checkpoint without retraining.

Run:
    uv run python -m microfold.report --ckpt models/best.pt --out outputs/val_best
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from microfold.dataset import PeptideDataset
from microfold.evaluate import run_val
from microfold.model import BackboneModel
from microfold.train import SPLIT_SEED, _split_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/best.pt")
    ap.add_argument("--out", type=str, default="outputs/val_best")
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full = PeptideDataset(index_csv=Path("data_cache/index_ok.csv"))
    _, va_idx = _split_indices(len(full), a.val_frac, SPLIT_SEED)
    val_ds = Subset(full, va_idx)

    model = BackboneModel(n_layers=a.n_layers).to(device)
    state = torch.load(a.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded {a.ckpt} (epoch {state.get('epoch')}, val_rmsd_mean {state.get('val_rmsd_mean')})")

    res = run_val(model, val_ds, device, Path(a.out), epoch=state.get("epoch", 0))
    agg = res["aggregate"]
    print(f"n={agg['n']} mean_rmsd {agg['mean_rmsd']:.3f} median {agg['median_rmsd']:.3f} min {agg['min_rmsd']:.3f} max {agg['max_rmsd']:.3f}")
    print(f"report: {Path(a.out) / 'macro.html'}")


if __name__ == "__main__":
    main()
