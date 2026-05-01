"""Filter the DBAASP CSV to peptides usable by the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_CSV = Path("data/peptides.csv")
DEFAULT_OUT = Path("data_cache/index.csv")
MAX_LEN = 30


def filter_peptides(csv_path: Path = DEFAULT_CSV, max_len: int = MAX_LEN) -> pd.DataFrame:
    """Return DataFrame with [ID, SEQUENCE, LENGTH] for peptides of length 1..max_len with canonical AAs."""
    df = pd.read_csv(csv_path)
    df = df[["ID", "SEQUENCE"]].copy()
    df["SEQUENCE"] = df["SEQUENCE"].astype(str).str.strip().str.upper()
    df["LENGTH"] = df["SEQUENCE"].str.len()
    df = df[(df["LENGTH"] > 0) & (df["LENGTH"] <= max_len)]
    df = df[df["SEQUENCE"].apply(lambda s: set(s).issubset(CANONICAL_AA))]
    df = df.reset_index(drop=True)
    return df


def main(csv_path: Path = DEFAULT_CSV, out_path: Path = DEFAULT_OUT) -> None:
    df = filter_peptides(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Kept {len(df)} peptides. min_len={df['LENGTH'].min()} max_len={df['LENGTH'].max()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
