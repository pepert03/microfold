from __future__ import annotations

from pathlib import Path

import torch

from microfold.geometry import cache_frames, extract_backbone_frames, fetch_pdb_for_dbaasp

DBAASP_ID = 11
SEQUENCE = "RVKRVWPLVIRTVIAGYNLYRAIKKK"


def test_fetch_and_extract(tmp_path: Path) -> None:
    pdb = fetch_pdb_for_dbaasp(DBAASP_ID)
    assert pdb.exists() and pdb.stat().st_size > 0

    rec = extract_backbone_frames(pdb, SEQUENCE)
    n = len(SEQUENCE)
    assert rec["R"].shape == (30, 3, 3)
    assert rec["t"].shape == (30, 3)
    assert rec["mask"].shape == (30,)
    assert int(rec["mask"].sum()) == n

    R_real = rec["R"][:n]
    eye = torch.eye(3).expand(n, 3, 3)
    assert torch.allclose(R_real @ R_real.transpose(-1, -2), eye, atol=1e-4)
    dets = torch.linalg.det(R_real)
    assert torch.allclose(dets, torch.ones(n), atol=1e-4)


def test_cache_frames_idempotent() -> None:
    p1 = cache_frames(DBAASP_ID, SEQUENCE)
    p2 = cache_frames(DBAASP_ID, SEQUENCE)
    assert p1 == p2 and p1.exists()


def test_partial_match_truncated_terminus() -> None:
    """ID 1095 has DBAASP seq of 26 aa but PDB chain misses the trailing 'A' (25 visible)."""
    truncated_id = 1095
    truncated_seq = "FKCRRWQWRMKKLGAPSITCVRRAFA"
    pdb = fetch_pdb_for_dbaasp(truncated_id)
    rec = extract_backbone_frames(pdb, truncated_seq)
    assert int(rec["mask"].sum()) == 25, "should resolve all but the unresolved terminal residue"
    assert rec["mask"][24].item() == 1.0, "second-to-last residue must be resolved"
    assert rec["mask"][25].item() == 0.0, "unresolved C-terminal residue must be masked out"
