"""Training loop with optional overfit mode + 85/15 train-val split + periodic val sweeps.

Run full-dataset:
    uv run python -m microfold.train --epochs 20 --val-every 5 --batch 8

Overfit smoke (no split):
    uv run python -m microfold.train --overfit --epochs 500 --batch 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from microfold.dataset import PeptideDataset, collate
from microfold.evaluate import run_val
from microfold.geometry import cache_frames
from microfold.losses import total_loss
from microfold.model import BackboneModel
from microfold.stamping import stamp_backbone
from microfold.visualization import calculate_backbone_rmsd

MODELS_DIR = Path("models")
SPLIT_SEED = 42


def _prepare_overfit_index(n_target: int, src_csv: Path, dst_csv: Path) -> Path:
    src = pd.read_csv(src_csv)
    rows = []
    for _, r in src.iterrows():
        if len(rows) >= n_target:
            break
        try:
            cache_frames(int(r["ID"]), str(r["SEQUENCE"]))
            rows.append({"ID": int(r["ID"]), "SEQUENCE": str(r["SEQUENCE"]), "LENGTH": int(r["LENGTH"])})
        except Exception as e:
            print(f"skip {r['ID']}: {e}")
    if len(rows) < n_target:
        raise SystemExit(f"only {len(rows)} peptides cached, wanted {n_target}")
    Path(dst_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dst_csv, index=False)
    return dst_csv


def _kabsch_batch_rmsd(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> list[float]:
    return [calculate_backbone_rmsd(truth[b], pred[b], mask[b]) for b in range(pred.shape[0])]


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    n_val = max(1, int(round(n * val_frac)))
    val = sorted(idxs[:n_val])
    train = sorted(idxs[n_val:])
    return train, val


def train(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_layers: int = 8,
    overfit: bool = False,
    log_every: int = 1,
    val_every: int = 5,
    val_frac: float = 0.15,
    device: str | None = None,
    index_csv: Path = Path("data_cache/index_ok.csv"),
    out_root: Path = Path("outputs"),
) -> None:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device_t}")

    if overfit:
        idx = _prepare_overfit_index(batch_size, Path("data_cache/index.csv"), Path("data_cache/overfit_index.csv"))
        full = PeptideDataset(index_csv=idx)
        train_ds = full
        val_ds = None
    else:
        if not index_csv.exists():
            raise SystemExit(f"missing {index_csv}; run `uv run python -m microfold.prepare` first")
        full = PeptideDataset(index_csv=index_csv)
        tr_idx, va_idx = _split_indices(len(full), val_frac, SPLIT_SEED)
        train_ds = Subset(full, tr_idx)
        val_ds = Subset(full, va_idx)
        print(f"split: train={len(train_ds)} val={len(val_ds)} (seed={SPLIT_SEED})")

    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate, shuffle=not overfit, drop_last=False)

    model = BackboneModel(n_layers=n_layers).to(device_t)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)

    fixed_batch = next(iter(train_dl)) if overfit else None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_total = 0.0
        ep_inter = 0.0
        ep_final = 0.0
        ep_rmsd_sum = 0.0
        ep_n = 0

        if overfit:
            iters = [fixed_batch]
        else:
            iters = train_dl

        for batch in iters:
            seqs = batch["sequence"]
            mask = batch["mask"].to(device_t)
            true_R = batch["R"].to(device_t)
            true_t = batch["t"].to(device_t)
            true_all = batch["true_all_backbone_coords"].to(device_t)

            out = model(seqs, mask)
            losses = total_loss(out["intermediate"], out["R"], out["t"], true_R, true_t, true_all, mask)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            with torch.no_grad():
                pred_all = stamp_backbone(out["R"], out["t"]).cpu()
                rmsds = _kabsch_batch_rmsd(pred_all, batch["true_all_backbone_coords"], batch["mask"])

            B = mask.shape[0]
            ep_total += losses["total"].item() * B
            ep_inter += losses["intermediate"].item() * B
            ep_final += losses["final"].item() * B
            ep_rmsd_sum += sum(rmsds)
            ep_n += B

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"epoch {epoch:4d}  train: total {ep_total / ep_n:.4f}  "
                f"inter {ep_inter / ep_n:.4f}  final {ep_final / ep_n:.4f}  "
                f"rmsd_mean {ep_rmsd_sum / ep_n:.3f}"
            )

        if val_ds is not None and (epoch % val_every == 0 or epoch == epochs):
            val_dir = out_root / f"val_epoch_{epoch:03d}"
            res = run_val(model, val_ds, device_t, val_dir, epoch)
            agg = res["aggregate"]
            print(
                f"   val epoch {epoch}: n={agg['n']}  mean_total {agg['mean_total']:.4f}  "
                f"mean_rmsd {agg['mean_rmsd']:.3f} A  median {agg['median_rmsd']:.3f}  "
                f"min {agg['min_rmsd']:.3f}  max {agg['max_rmsd']:.3f}"
            )
            print(f"   report: {val_dir / 'macro.html'}")
            if agg["mean_rmsd"] < best_metric:
                best_metric = agg["mean_rmsd"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_rmsd_mean": best_metric},
                    MODELS_DIR / "best.pt",
                )
                print(f"   saved models/best.pt @ val_rmsd_mean {best_metric:.3f}")
        elif overfit:
            cur = ep_rmsd_sum / ep_n
            if cur < best_metric:
                best_metric = cur
                torch.save({"model": model.state_dict(), "epoch": epoch, "rmsd": cur}, MODELS_DIR / "best.pt")

    print(f"done. best metric = {best_metric:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    train(
        epochs=a.epochs,
        batch_size=a.batch,
        lr=a.lr,
        n_layers=a.n_layers,
        overfit=a.overfit,
        log_every=a.log_every,
        val_every=a.val_every,
        val_frac=a.val_frac,
        device=a.device,
    )


if __name__ == "__main__":
    main()
