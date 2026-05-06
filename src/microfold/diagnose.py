"""Diagnose why some DBAASP peptides fail extraction.

For each failure: fetch PDB + JSON, list per-chain canonical sequences,
compute longest contiguous substring shared with DBAASP sequence, classify reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import biotite.structure.io.pdb as pdb_io
import numpy as np
import pandas as pd

from microfold.geometry import (
    BACKBONE_ATOMS,
    THREE_TO_ONE,
    fetch_pdb_for_dbaasp,
)


def _chain_seq(atoms, chain_id: str) -> str:
    ch = atoms[atoms.chain_id == chain_id]
    s: list[str] = []
    for rid in np.unique(ch.res_id):
        res = ch[ch.res_id == int(rid)]
        if not all(a in set(res.atom_name) for a in BACKBONE_ATOMS):
            continue
        one = THREE_TO_ONE.get(res.res_name[0])
        if one:
            s.append(one)
    return "".join(s)


def _lcs(a: str, b: str) -> tuple[int, int, int]:
    """Longest common contiguous substring; returns (i_a, i_b, length)."""
    n, m = len(a), len(b)
    if not n or not m:
        return 0, 0, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = (0, 0, 0)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > best[2]:
                    best = (i + 1 - dp[i + 1][j + 1], j + 1 - dp[i + 1][j + 1], dp[i + 1][j + 1])
    return best


def diagnose() -> None:
    df = pd.read_csv("data_cache/index.csv")
    ok = set(pd.read_csv("data_cache/index_ok.csv")["ID"].astype(int))
    failed = df[~df["ID"].astype(int).isin(ok)]
    print(f"failed peptides: {len(failed)}\n")

    classes: dict[str, list[int]] = {"truncated": [], "mutation_or_partial": [], "no_match": [], "no_pdb": [], "other": []}
    rows = []
    for _, r in failed.iterrows():
        dbid = int(r["ID"])
        seq = str(r["SEQUENCE"])
        try:
            pdb = fetch_pdb_for_dbaasp(dbid)
        except Exception as e:
            classes["no_pdb"].append(dbid)
            rows.append({"id": dbid, "len": len(seq), "reason": "no_pdb", "detail": str(e)[:60]})
            continue

        try:
            struct = pdb_io.PDBFile.read(str(pdb)).get_structure(model=1)
            atoms = struct[(struct.element != "H") & (~struct.hetero)]
            best_lcs_len = 0
            best_chain = ""
            best_chain_seq = ""
            best_offsets: tuple[int, int] = (0, 0)
            for cid in np.unique(atoms.chain_id):
                cseq = _chain_seq(atoms, str(cid))
                ia, ib, L = _lcs(seq, cseq)
                if L > best_lcs_len:
                    best_lcs_len = L
                    best_chain = str(cid)
                    best_chain_seq = cseq
                    best_offsets = (ia, ib)
            frac = best_lcs_len / len(seq) if seq else 0.0
            if best_lcs_len == len(seq):
                kind = "should_have_matched"  # shouldn't happen since these failed
            elif best_lcs_len >= len(seq) - 6 and frac >= 0.7:
                kind = "truncated"
            elif frac >= 0.4:
                kind = "mutation_or_partial"
            else:
                kind = "no_match"
            classes[kind].append(dbid) if kind in classes else classes.setdefault(kind, []).append(dbid)
            rows.append(
                {
                    "id": dbid,
                    "len": len(seq),
                    "reason": kind,
                    "lcs_len": best_lcs_len,
                    "lcs_frac": round(frac, 2),
                    "chain": best_chain,
                    "dbaasp_offset": best_offsets[0],
                    "chain_offset": best_offsets[1],
                    "dbaasp_seq": seq,
                    "chain_seq_first40": best_chain_seq[:40],
                    "chain_seq_len": len(best_chain_seq),
                }
            )
        except Exception as e:
            classes["other"].append(dbid)
            rows.append({"id": dbid, "len": len(seq), "reason": "other", "detail": str(e)[:60]})

    print("classes:")
    for k, v in classes.items():
        print(f"  {k}: {len(v)}  ids={v[:8]}{'...' if len(v) > 8 else ''}")
    print()
    out = Path("data_cache/failure_diagnosis.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    print("\nfirst rows:")
    print(pd.DataFrame(rows).to_string(index=False, max_colwidth=42))


if __name__ == "__main__":
    diagnose()
