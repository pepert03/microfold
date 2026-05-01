from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from microfold.dataset import PeptideDataset, collate
from microfold.geometry import cache_frames

REPO = Path(__file__).resolve().parents[1]


def _mini_index(tmp_path: Path) -> Path:
    """Single-row index over an already-cached peptide so the test runs offline."""
    cache_frames(11, "RVKRVWPLVIRTVIAGYNLYRAIKKK")
    p = tmp_path / "mini_index.csv"
    pd.DataFrame({"ID": [11], "SEQUENCE": ["RVKRVWPLVIRTVIAGYNLYRAIKKK"], "LENGTH": [26]}).to_csv(p, index=False)
    return p


def test_dataloader_yields_batched_tensors(tmp_path: Path) -> None:
    idx = _mini_index(tmp_path)
    ds = PeptideDataset(index_csv=idx)
    assert len(ds) == 1

    dl = DataLoader(ds, batch_size=1, collate_fn=collate)
    batch = next(iter(dl))
    assert batch["R"].shape == (1, 30, 3, 3)
    assert batch["t"].shape == (1, 30, 3)
    assert batch["mask"].shape == (1, 30)
    assert batch["true_all_backbone_coords"].shape == (1, 30, 3, 3)
    assert isinstance(batch["sequence"], list) and len(batch["sequence"]) == 1
    assert batch["id"] == [11]
    assert torch.is_tensor(batch["R"])
