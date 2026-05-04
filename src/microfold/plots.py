"""PDF plot rendering for training history.

Pure functions reading a list-of-dicts history; safe to call repeatedly each val epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _epochs_with(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in history:
        v = row.get(key)
        if v is not None:
            xs.append(int(row["epoch"]))
            ys.append(float(v))
    return xs, ys


def plot_total_loss(history: list[dict[str, Any]], out_path: Path) -> None:
    """Train + val total loss vs epoch, aligned on shared x-axis."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tx, ty = _epochs_with(history, "train_total")
    vx, vy = _epochs_with(history, "val_total")
    if tx:
        ax.plot(tx, ty, marker=".", linewidth=1.4, label="train")
    if vx:
        ax.plot(vx, vy, marker="o", linewidth=1.4, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total FAPE loss")
    ax.set_title("Total loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_loss_components(history: list[dict[str, Any]], out_path: Path) -> None:
    """Intermediate vs final FAPE, train+val on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        ("train_inter", "train · intermediate", "-", "."),
        ("train_final", "train · final", "-", "."),
        ("val_inter", "val · intermediate", "--", "o"),
        ("val_final", "val · final", "--", "o"),
    ]
    for key, label, ls, mk in series:
        xs, ys = _epochs_with(history, key)
        if xs:
            ax.plot(xs, ys, linestyle=ls, marker=mk, linewidth=1.3, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("FAPE loss")
    ax.set_title("Loss components")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_rmsd(history: list[dict[str, Any]], out_path: Path) -> None:
    """Train + val RMSD vs epoch. Val plotted as mean line + min/max shaded band."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tx, ty = _epochs_with(history, "train_rmsd")
    if tx:
        ax.plot(tx, ty, marker=".", linewidth=1.4, label="train (mean)", color="#1f77b4")

    vx, vy = _epochs_with(history, "val_rmsd_mean")
    if vx:
        ax.plot(vx, vy, marker="o", linewidth=1.6, label="val (mean)", color="#d62728")
        # min/max band when present
        lo_x, lo = _epochs_with(history, "val_rmsd_min")
        hi_x, hi = _epochs_with(history, "val_rmsd_max")
        if lo_x == vx and hi_x == vx and lo and hi:
            ax.fill_between(vx, lo, hi, alpha=0.15, color="#d62728", label="val (min–max)")
        mx, my = _epochs_with(history, "val_rmsd_median")
        if mx:
            ax.plot(mx, my, marker="x", linewidth=1.0, linestyle=":", color="#d62728", label="val (median)")

    ax.set_xlabel("epoch")
    ax.set_ylabel("backbone RMSD (Å)")
    ax.set_title("RMSD")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def render_all(history: list[dict[str, Any]], out_dir: Path) -> None:
    """Write every plot into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_total_loss(history, out_dir / "loss_total.pdf")
    plot_loss_components(history, out_dir / "loss_components.pdf")
    plot_rmsd(history, out_dir / "rmsd.pdf")
