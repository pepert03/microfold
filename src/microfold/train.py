"""Training loop with optional overfit mode + 85/15 train-val split + periodic val sweeps.

Run full-dataset:
    uv run python -m microfold.train --epochs 20 --val-every 5 --batch 8

Overfit smoke (no split):
    uv run python -m microfold.train --overfit --epochs 500 --batch 4

Each invocation creates a fresh `outputs/run_YYYYMMDD_HHMMSS/` directory containing:
    config.json     - hyperparams + git SHA + device
    history.csv     - every metric per epoch
    train.log       - tee of stdout
    best.pt, last.pt
    val_epoch_NNN/  - per-val-epoch reports + cumulative PDF plots
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from microfold.dataset import PeptideDataset, collate
from microfold.evaluate import run_val
from microfold.geometry import cache_frames
from microfold.losses import total_loss
from microfold.model import BackboneModel
from microfold.plots import render_all
from microfold.stamping import stamp_backbone
from microfold.visualization import calculate_backbone_rmsd

MODELS_DIR = Path("models")
SPLIT_SEED = 42

HISTORY_COLS = [
    "epoch",
    "train_total", "train_inter", "train_final", "train_rmsd",
    "val_total", "val_inter", "val_final",
    "val_rmsd_mean", "val_rmsd_median", "val_rmsd_min", "val_rmsd_max",
    "lr",
]


class _Tee:
    """Mirror writes to stdout and a log file. Drop-in for sys.stdout."""

    def __init__(self, stream: Any, path: Path) -> None:
        self._stream = stream
        self._fh = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> int:
        self._fh.write(s)
        return self._stream.write(s)

    def flush(self) -> None:
        self._fh.flush()
        self._stream.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        # Forward any attribute (isatty, fileno, encoding, ...) to wrapped stream.
        return getattr(self._stream, name)


def _git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
        return r.stdout.strip() or None
    except Exception:
        return None


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


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k) for k in HISTORY_COLS})


def train(
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_layers: int = 8,
    c_s: int | None = None,
    c_hidden: int = 16,
    n_heads: int = 4,
    n_qpoints: int = 4,
    n_vpoints: int = 8,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    max_len: int = 30,
    overfit: bool = False,
    log_every: int = 1,
    val_every: int = 5,
    val_frac: float = 0.15,
    early_stop_patience: int = 6,
    early_stop_min_delta: float = 0.01,
    device: str | None = None,
    index_csv: Path = Path("data_cache/index_ok.csv"),
    out_root: Path = Path("outputs"),
) -> None:
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tee = _Tee(sys.stdout, run_dir / "train.log")
    sys.stdout = tee  # type: ignore[assignment]

    try:
        device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"device: {device_t}")
        print(f"run dir: {run_dir}")

        config = {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "device": str(device_t),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_layers": n_layers,
            "c_s": c_s,
            "c_hidden": c_hidden,
            "n_heads": n_heads,
            "n_qpoints": n_qpoints,
            "n_vpoints": n_vpoints,
            "dropout": dropout,
            "max_len": max_len,
            "overfit": overfit,
            "log_every": log_every,
            "val_every": val_every,
            "val_frac": val_frac,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "split_seed": SPLIT_SEED,
            "index_csv": str(index_csv),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        if overfit:
            idx = _prepare_overfit_index(batch_size, Path("data_cache/index.csv"), Path("data_cache/overfit_index.csv"))
            full = PeptideDataset(index_csv=idx)
            train_ds: Any = full
            val_ds: Any = None
        else:
            if not index_csv.exists():
                raise SystemExit(f"missing {index_csv}; run `uv run python -m microfold.prepare` first")
            full = PeptideDataset(index_csv=index_csv)
            tr_idx, va_idx = _split_indices(len(full), val_frac, SPLIT_SEED)
            train_ds = Subset(full, tr_idx)
            val_ds = Subset(full, va_idx)
            print(f"split: train={len(train_ds)} val={len(val_ds)} (seed={SPLIT_SEED})")

        train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate, shuffle=not overfit, drop_last=False)

        model = BackboneModel(
            n_layers=n_layers,
            c_s=c_s,
            c_hidden=c_hidden,
            n_heads=n_heads,
            n_qpoints=n_qpoints,
            n_vpoints=n_vpoints,
            dropout=dropout,
            max_len=max_len,
        ).to(device_t)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {n_params:,}")
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        fixed_batch = next(iter(train_dl)) if overfit else None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        history: list[dict[str, Any]] = []
        best_metric = float("inf")
        best_epoch = 0
        no_improve = 0  # consecutive val checks with no improvement
        stopped_early = False

        for epoch in range(1, epochs + 1):
            model.train()
            ep_total = 0.0
            ep_inter = 0.0
            ep_final = 0.0
            ep_rmsd_sum = 0.0
            ep_n = 0

            iters = [fixed_batch] if overfit else train_dl

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

            row: dict[str, Any] = {
                "epoch": epoch,
                "train_total": ep_total / ep_n,
                "train_inter": ep_inter / ep_n,
                "train_final": ep_final / ep_n,
                "train_rmsd": ep_rmsd_sum / ep_n,
                "lr": opt.param_groups[0]["lr"],
            }

            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(
                    f"epoch {epoch:4d}  train: total {row['train_total']:.4f}  "
                    f"inter {row['train_inter']:.4f}  final {row['train_final']:.4f}  "
                    f"rmsd_mean {row['train_rmsd']:.3f}"
                )

            improved = False
            if val_ds is not None and (epoch % val_every == 0 or epoch == epochs):
                val_dir = run_dir / f"val_epoch_{epoch:03d}"
                res = run_val(model, val_ds, device_t, val_dir, epoch)
                agg = res["aggregate"]
                row.update({
                    "val_total": agg["mean_total"],
                    "val_inter": agg["mean_intermediate"],
                    "val_final": agg["mean_final"],
                    "val_rmsd_mean": agg["mean_rmsd"],
                    "val_rmsd_median": agg["median_rmsd"],
                    "val_rmsd_min": agg["min_rmsd"],
                    "val_rmsd_max": agg["max_rmsd"],
                })
                print(
                    f"   val epoch {epoch}: n={agg['n']}  mean_total {agg['mean_total']:.4f}  "
                    f"mean_rmsd {agg['mean_rmsd']:.3f} A  median {agg['median_rmsd']:.3f}  "
                    f"min {agg['min_rmsd']:.3f}  max {agg['max_rmsd']:.3f}"
                )
                print(f"   report: {val_dir / 'macro.html'}")

                history.append(row)
                _write_history_csv(history, run_dir / "history.csv")
                render_all(history, val_dir)

                if agg["mean_rmsd"] < best_metric - early_stop_min_delta:
                    best_metric = agg["mean_rmsd"]
                    best_epoch = epoch
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1
                    print(f"   no improvement ({no_improve}/{early_stop_patience}); best {best_metric:.3f} @ epoch {best_epoch}")
            else:
                history.append(row)
                _write_history_csv(history, run_dir / "history.csv")
                if overfit:
                    cur = row["train_rmsd"]
                    if cur < best_metric:
                        best_metric = cur
                        improved = True

            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "metric": row.get("val_rmsd_mean", row["train_rmsd"]),
            }
            torch.save(ckpt, run_dir / "last.pt")
            if improved:
                torch.save(ckpt, run_dir / "best.pt")
                torch.save(ckpt, MODELS_DIR / "best.pt")
                print(f"   saved best.pt @ metric {best_metric:.3f}")

            if val_ds is not None and early_stop_patience > 0 and no_improve >= early_stop_patience:
                stopped_early = True
                print(f"early stop: {no_improve} val checks w/o >{early_stop_min_delta:.3f} A improvement; best {best_metric:.3f} @ epoch {best_epoch}")
                break

        print(f"done. best metric = {best_metric:.3f}{' (early-stopped)' if stopped_early else ''}")
        print(f"run dir: {run_dir}")
    finally:
        sys.stdout = tee._stream  # type: ignore[assignment]
        tee.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--c-s", type=int, default=None, help="trunk dim; default = ESM hidden size")
    ap.add_argument("--c-hidden", type=int, default=16, help="per-head scalar dim in IPA")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-qpoints", type=int, default=4)
    ap.add_argument("--n-vpoints", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--early-stop-patience", type=int, default=6, help="val checks w/o improvement before stop; 0 disables")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.01, help="min RMSD (A) improvement to reset patience")
    ap.add_argument("--device", type=str, default=None)
    a = ap.parse_args()
    train(
        epochs=a.epochs,
        batch_size=a.batch,
        lr=a.lr,
        weight_decay=a.weight_decay,
        n_layers=a.n_layers,
        c_s=a.c_s,
        c_hidden=a.c_hidden,
        n_heads=a.n_heads,
        n_qpoints=a.n_qpoints,
        n_vpoints=a.n_vpoints,
        dropout=a.dropout,
        max_len=a.max_len,
        overfit=a.overfit,
        log_every=a.log_every,
        val_every=a.val_every,
        val_frac=a.val_frac,
        early_stop_patience=a.early_stop_patience,
        early_stop_min_delta=a.early_stop_min_delta,
        device=a.device,
    )


if __name__ == "__main__":
    main()
