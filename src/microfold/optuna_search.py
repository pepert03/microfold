"""Optuna hyperparameter search over the microfold training pipeline.

Run:
    uv run python -m microfold.optuna_search --n-trials 30 --epochs 50

Each trial trains a fresh model with sampled hyperparameters, reports
val_rmsd_mean to Optuna at every validation epoch, and minimises the best
val_rmsd_mean reached. Results persist in `optuna_study.db` (SQLite); per-trial
artefacts land in `outputs/optuna_<study>/trial_NNN/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna

from microfold.train import train


def make_objective(study_dir: Path, epochs: int, val_every: int):
    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "lr":           trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
            "n_layers":     trial.suggest_int("n_layers", 4, 12),
            "c_hidden":     trial.suggest_categorical("c_hidden", [8, 16, 32, 64]),
            "n_heads":      trial.suggest_categorical("n_heads", [2, 4, 6, 8]),
            "n_qpoints":    trial.suggest_categorical("n_qpoints", [2, 4, 8]),
            "n_vpoints":    trial.suggest_categorical("n_vpoints", [4, 8, 12]),
        }
        run_dir = study_dir / f"trial_{trial.number:03d}"

        def on_val(epoch: int, agg: dict[str, Any]) -> None:
            trial.report(float(agg["mean_rmsd"]), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = train(
            epochs=epochs,
            val_every=val_every,
            on_validation=on_val,
            run_dir_override=run_dir,
            **params,
        )
        return result["best_rmsd"]

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="microfold")
    ap.add_argument("--storage", default="sqlite:///optuna_study.db")
    ap.add_argument("--n-trials", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=None,
                    help="overall wall-clock budget in seconds")
    a = ap.parse_args()

    study_dir = Path("outputs") / f"optuna_{a.study_name}"
    study_dir.mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=a.study_name,
        storage=a.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        make_objective(study_dir, a.epochs, a.val_every),
        n_trials=a.n_trials,
        timeout=a.timeout,
        gc_after_trial=True,
    )

    best = study.best_trial
    print(f"\nbest trial #{best.number}: rmsd {best.value:.3f}")
    print(json.dumps(best.params, indent=2))
    (study_dir / "best_params.json").write_text(json.dumps(
        {"value": best.value, "params": best.params, "trial": best.number},
        indent=2,
    ))


if __name__ == "__main__":
    main()
