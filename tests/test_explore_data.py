from __future__ import annotations

from pathlib import Path

from microfold.explore_data import CANONICAL_AA, filter_peptides

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "peptides.csv"


def test_filter_bounds() -> None:
    df = filter_peptides(CSV)
    assert len(df) > 0
    assert df["LENGTH"].max() <= 30
    assert df["LENGTH"].min() > 0
    assert {"ID", "SEQUENCE", "LENGTH"} <= set(df.columns)
    for seq in df["SEQUENCE"]:
        assert set(seq).issubset(CANONICAL_AA), seq
