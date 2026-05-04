# microfold

```bash
uv sync
uv run python -m microfold.explore_data
uv run python -m microfold.prepare

uv run python -m microfold.optuna_search --n-trials 2 --epochs 10 