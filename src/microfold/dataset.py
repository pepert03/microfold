"""PyTorch Dataset over pre-cached DBAASP frame records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from microfold.geometry import CACHE_FRAMES_DIR, cache_frames

DEFAULT_INDEX = Path("data_cache/index.csv")


class PeptideDataset(Dataset):
    """Lazy-but-cached peptide dataset.

    Each item is the dict written by `microfold.geometry.cache_frames`. First-time
    access for an ID downloads PDB + extracts frames; subsequent calls hit the .pt cache.
    """

    def __init__(
        self,
        index_csv: Path = DEFAULT_INDEX,
        cache_dir: Path = CACHE_FRAMES_DIR,
        prebuild: bool = False,
        skip_failed: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        df = pd.read_csv(index_csv)
        rows: list[tuple[int, str]] = []
        for _, r in df.iterrows():
            dbid = int(r["ID"])
            seq = str(r["SEQUENCE"])
            if prebuild:
                try:
                    cache_frames(dbid, seq, self.cache_dir)
                except Exception as e:
                    if not skip_failed:
                        raise
                    print(f"skip {dbid}: {e}")
                    continue
            rows.append((dbid, seq))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        dbid, seq = self.rows[i]
        path = cache_frames(dbid, seq, self.cache_dir)
        rec = torch.load(path, weights_only=False)
        rec["id"] = dbid
        return rec


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, keep sequence and id as lists."""
    keys = ["R", "t", "mask", "true_ca_coords", "true_all_backbone_coords"]
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["sequence"] = [b["sequence"] for b in batch]
    out["id"] = [b["id"] for b in batch]
    return out
