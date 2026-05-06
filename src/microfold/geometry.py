"""Extract backbone rigid frames (R, t) from a peptide PDB matching a DBAASP sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
import numpy as np
import requests
import torch

DBAASP_API = "https://dbaasp.org/peptides/{id}"
CACHE_PDB_DIR = Path("data_cache/pdb")
CACHE_FRAMES_DIR = Path("data_cache/frames")
MAX_LEN = 30
BACKBONE_ATOMS = ("N", "CA", "C")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class FrameRecord(TypedDict):
    sequence: str
    R: torch.Tensor          # [max_len, 3, 3]
    t: torch.Tensor          # [max_len, 3]
    mask: torch.Tensor       # [max_len] float
    true_ca_coords: torch.Tensor              # [max_len, 3]
    true_all_backbone_coords: torch.Tensor    # [max_len, 3, 3]  atom order N, CA, C


def fetch_pdb_for_dbaasp(dbaasp_id: int | str, cache_dir: Path = CACHE_PDB_DIR) -> Path:
    """Download the first PDB structure for a DBAASP entry and return the local path.

    Hits https://dbaasp.org/peptides/{id} for JSON, then downloads pdbs[0].pdbFileUrl.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{dbaasp_id}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out

    meta_path = cache_dir / f"{dbaasp_id}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        r = requests.get(DBAASP_API.format(id=dbaasp_id), headers={"Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        meta = r.json()
        meta_path.write_text(json.dumps(meta))

    pdbs = meta.get("pdbs") or []
    if not pdbs:
        raise RuntimeError(f"DBAASP {dbaasp_id} has no PDB entries")
    # DBAASP sometimes returns "<PDBID>_<chain_idx>.pdb" which 404s on RCSB.
    # Always rebuild from the 4-char PDB code so we hit the real file.
    pdb_code = pdbs[0]["name"].split("_")[0].upper()
    url = f"https://files.rcsb.org/view/{pdb_code}.pdb"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def _chain_residues(atoms: struc.AtomArray, chain_id: str) -> tuple[np.ndarray, str]:
    """Return (sorted residue ids in chain, one-letter sequence built from residues with all backbone atoms)."""
    ch = atoms[atoms.chain_id == chain_id]
    res_ids = np.unique(ch.res_id)
    seq_chars: list[str] = []
    keep_ids: list[int] = []
    for rid in res_ids:
        res = ch[ch.res_id == rid]
        names = set(res.atom_name)
        if not all(a in names for a in BACKBONE_ATOMS):
            continue
        rname = res.res_name[0]
        one = THREE_TO_ONE.get(rname)
        if one is None:
            continue
        seq_chars.append(one)
        keep_ids.append(int(rid))
    return np.array(keep_ids, dtype=np.int64), "".join(seq_chars)


def _gram_schmidt_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """AF2-style frame: e1 from C-CA, e2 orthogonal to e1 toward N-CA, e3 = e1 x e2."""
    v1 = c - ca
    e1 = v1 / np.linalg.norm(v1)
    v2 = n - ca
    u2 = v2 - np.dot(v2, e1) * e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)  # columns = local axes -> [3,3]


def _longest_common_substring(a: str, b: str) -> tuple[int, int, int]:
    """Return (offset_in_a, offset_in_b, length) of the longest contiguous shared substring."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0, 0, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = (0, 0, 0)
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > best[2]:
                    L = dp[i + 1][j + 1]
                    best = (i + 1 - L, j + 1 - L, L)
    return best


def extract_backbone_frames(
    pdb_path: Path,
    sequence: str,
    max_len: int = MAX_LEN,
    min_match_frac: float = 0.6,
) -> FrameRecord:
    """Parse a PDB, locate `sequence` inside one chain, return padded frames.

    Tries an exact substring match first, then falls back to the longest contiguous
    substring shared between `sequence` and any chain (handles unresolved terminal
    residues that the experiment couldn't capture). Unmatched positions get mask=0
    with R=I, t=0, and the model treats them as padding.

    Args:
        min_match_frac: minimum (LCS length / sequence length) accepted; below this we drop the entry.

    Raises:
        RuntimeError if no chain shares enough residues with `sequence`.
    """
    sequence = sequence.upper()
    if len(sequence) > max_len:
        raise ValueError(f"sequence longer than max_len={max_len}: {len(sequence)}")

    struct = pdb_io.PDBFile.read(str(pdb_path)).get_structure(model=1)
    atoms = struct[(struct.element != "H") & (~struct.hetero)]

    # Find the alignment with the longest contiguous match across all chains.
    best_chain_id: str | None = None
    best_res_ids: np.ndarray | None = None
    best_offset_seq = 0          # where in `sequence` the match starts
    best_len = 0
    for chain_id in np.unique(atoms.chain_id):
        res_ids, chain_seq = _chain_residues(atoms, str(chain_id))
        # Fast path: exact substring match — pin offset_seq to 0.
        idx = chain_seq.find(sequence)
        if idx >= 0:
            best_chain_id = str(chain_id)
            best_res_ids = res_ids[idx : idx + len(sequence)]
            best_offset_seq = 0
            best_len = len(sequence)
            break
        # Fallback: LCS.
        off_seq, off_chain, L = _longest_common_substring(sequence, chain_seq)
        if L > best_len:
            best_chain_id = str(chain_id)
            best_res_ids = res_ids[off_chain : off_chain + L]
            best_offset_seq = off_seq
            best_len = L

    if best_chain_id is None or best_res_ids is None or best_len == 0:
        raise RuntimeError(f"Sequence {sequence!r} not found in any chain of {pdb_path.name}")

    frac = best_len / len(sequence)
    if frac < min_match_frac:
        raise RuntimeError(
            f"best match for {sequence!r} in {pdb_path.name} only {best_len}/{len(sequence)} "
            f"residues ({frac:.2f} < {min_match_frac:.2f})"
        )

    chain = atoms[atoms.chain_id == best_chain_id]
    R = np.tile(np.eye(3), (max_len, 1, 1)).astype(np.float32)
    t = np.zeros((max_len, 3), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)
    ca_coords = np.zeros((max_len, 3), dtype=np.float32)
    all_bb = np.zeros((max_len, 3, 3), dtype=np.float32)

    for i, rid in enumerate(best_res_ids):
        seq_pos = best_offset_seq + i
        res = chain[chain.res_id == int(rid)]
        n_atom = res[res.atom_name == "N"].coord[0]
        ca_atom = res[res.atom_name == "CA"].coord[0]
        c_atom = res[res.atom_name == "C"].coord[0]

        R[seq_pos] = _gram_schmidt_frame(n_atom, ca_atom, c_atom)
        t[seq_pos] = ca_atom
        ca_coords[seq_pos] = ca_atom
        all_bb[seq_pos, 0] = n_atom
        all_bb[seq_pos, 1] = ca_atom
        all_bb[seq_pos, 2] = c_atom
        mask[seq_pos] = 1.0

    assert mask.sum() == best_len

    return {
        "sequence": sequence,
        "R": torch.from_numpy(R),
        "t": torch.from_numpy(t),
        "mask": torch.from_numpy(mask),
        "true_ca_coords": torch.from_numpy(ca_coords),
        "true_all_backbone_coords": torch.from_numpy(all_bb),
    }


def cache_frames(dbaasp_id: int | str, sequence: str, cache_dir: Path = CACHE_FRAMES_DIR) -> Path:
    """Download PDB (if needed), extract frames, save dict to data_cache/frames/{id}.pt."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{dbaasp_id}.pt"
    if out.exists():
        return out
    pdb_path = fetch_pdb_for_dbaasp(dbaasp_id)
    record = extract_backbone_frames(pdb_path, sequence)
    torch.save(record, out)
    return out
