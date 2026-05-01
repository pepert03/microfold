"""Pre-cache PDBs + frame tensors for every peptide in the index.

Slow first run (network-bound). Skips entries with no PDB or sequence-mismatch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from microfold.geometry import cache_frames

INDEX = Path("data_cache/index.csv")
OK_INDEX = Path("data_cache/index_ok.csv")


def prebuild(index_csv: Path = INDEX, out_csv: Path = OK_INDEX) -> Path:
    df = pd.read_csv(index_csv)
    rows = []
    failed = []
    for _, r in df.iterrows():
        dbid = int(r["ID"])
        seq = str(r["SEQUENCE"])
        try:
            cache_frames(dbid, seq)
            rows.append({"ID": dbid, "SEQUENCE": seq, "LENGTH": int(r["LENGTH"])})
        except Exception as e:
            failed.append((dbid, str(e)))

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"cached {len(rows)} / {len(df)} peptides; {len(failed)} failed")
    if failed:
        print("first failures:")
        for dbid, msg in failed[:10]:
            print(f"  {dbid}: {msg}")
    return out_csv


if __name__ == "__main__":
    prebuild()
