"""ESM-2 (frozen) + simplified Invariant Point Attention stack.

Simplifications vs. AF2 IPA:
- No pair representation z_ij — attention logits use only scalar and point terms.
- No structure module's `Transition`/`backbone_update` separation; one linear emits 6-d
  (axis-angle 3 + local translation 3) per-residue update.
- Rotations detached between layers so gradients only flow through translations.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ESM_NAME = "facebook/esm2_t12_35M_UR50D"  # esm2_t6_8M_UR50D is the smallest ESM-2 model, with 6 layers and 8M parameters. It produces 320-d per-residue embeddings.
MAX_LEN = 30


class ESMEmbedder(nn.Module):
    """Frozen ESM-2 encoder producing per-residue embeddings padded to max_len."""

    def __init__(self, model_name: str = ESM_NAME, max_len: int = MAX_LEN) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.c_s = int(self.model.config.hidden_size)
        self.max_len = max_len

    @torch.no_grad()
    def forward(self, sequences: list[str]) -> torch.Tensor:
        """Return s[B, max_len, c_s]. Real-residue rows are filled, padding rows are zeros."""
        device = next(self.model.parameters()).device
        tok = self.tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        out = self.model(**tok).last_hidden_state  # [B, T, c_s]
        b = len(sequences)
        s = out.new_zeros(b, self.max_len, self.c_s)
        for i, seq in enumerate(sequences):
            n = min(len(seq), self.max_len)
            s[i, :n] = out[i, 1 : 1 + n]  # strip <cls>
        return s


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula. aa: [..., 3] -> R: [..., 3, 3]."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / theta
    x, y, z = axis.unbind(-1)
    K = torch.stack(
        [
            torch.zeros_like(x),
            -z,
            y,
            z,
            torch.zeros_like(x),
            -x,
            -y,
            x,
            torch.zeros_like(x),
        ],
        dim=-1,
    ).reshape(*aa.shape[:-1], 3, 3)
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


class IPABlock(nn.Module):
    """Simplified IPA (no pair rep). Inputs: s, frames (R, t), mask."""

    def __init__(
        self,
        c_s: int,
        c_hidden: int = 16,
        n_heads: int = 4,
        n_qpoints: int = 4,
        n_vpoints: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_h = c_hidden
        self.h = n_heads
        self.pq = n_qpoints
        self.pv = n_vpoints

        self.lin_q = nn.Linear(c_s, n_heads * c_hidden)
        self.lin_k = nn.Linear(c_s, n_heads * c_hidden)
        self.lin_v = nn.Linear(c_s, n_heads * c_hidden)
        self.lin_qp = nn.Linear(c_s, n_heads * n_qpoints * 3)
        self.lin_kp = nn.Linear(c_s, n_heads * n_qpoints * 3)
        self.lin_vp = nn.Linear(c_s, n_heads * n_vpoints * 3)

        # Per-head learned scalar weight on point-distance term, softplus-positive.
        self.gamma = nn.Parameter(torch.zeros(n_heads))

        self.lin_out = nn.Linear(n_heads * (c_hidden + n_vpoints * 3 + n_vpoints), c_s)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(c_s)

        # Update head: 3 axis-angle + 3 local translation per residue
        self.update = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(c_s, 6),
        )

    def forward(
        self,
        s: torch.Tensor,  # [B, N, c_s]
        R: torch.Tensor,  # [B, N, 3, 3]
        t: torch.Tensor,  # [B, N, 3]
        mask: torch.Tensor,  # [B, N]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = s.shape
        H, C, Pq, Pv = self.h, self.c_h, self.pq, self.pv

        q = self.lin_q(s).view(B, N, H, C)
        k = self.lin_k(s).view(B, N, H, C)
        v = self.lin_v(s).view(B, N, H, C)

        qp_loc = self.lin_qp(s).view(B, N, H, Pq, 3)
        kp_loc = self.lin_kp(s).view(B, N, H, Pq, 3)
        vp_loc = self.lin_vp(s).view(B, N, H, Pv, 3)

        # Local -> global via current frame at residue i
        # frames[B,N,3,3] applied to points[B,N,H,P,3]
        def to_global(pts_loc: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bnij,bnhpj->bnhpi", R, pts_loc) + t.view(B, N, 1, 1, 3)

        qp = to_global(qp_loc)  # [B, N, H, Pq, 3]
        kp = to_global(kp_loc)
        vp = to_global(vp_loc)

        # Scalar attention: [B, H, N(query), N(key)]
        scalar_logits = torch.einsum("bnhc,bmhc->bhnm", q, k) / math.sqrt(C)
        scalar_logits = scalar_logits / math.sqrt(2.0)

        # Point attention: -gamma * w * sum_p ||qp_i - kp_j||^2
        # qp[B,N(q),H,P,3] - kp[B,N(k),H,P,3]
        diff = qp.unsqueeze(2) - kp.unsqueeze(1)  # [B, Nq, Nk, H, P, 3]
        sq_dist = diff.pow(2).sum(dim=(-1, -2))  # [B, Nq, Nk, H]
        sq_dist = sq_dist.permute(0, 3, 1, 2).contiguous()  # [B, H, Nq, Nk]
        gamma = nn.functional.softplus(self.gamma).view(1, H, 1, 1)
        w_point = math.sqrt(2.0 / (9 * Pq))
        point_logits = -0.5 * gamma * w_point * sq_dist

        logits = scalar_logits + point_logits

        # Mask key positions where mask_j == 0
        key_mask = mask.view(B, 1, 1, N)  # broadcast over h, query
        logits = logits.masked_fill(key_mask < 0.5, -1e9)

        attn = torch.softmax(logits, dim=-1)  # [B, H, Nq, Nk]
        # Zero out queries that are padding (so their output is harmless)
        q_mask = mask.view(B, 1, N, 1)
        attn = attn * q_mask

        # Aggregate scalar values
        o_scalar = torch.einsum("bhnm,bmhc->bnhc", attn, v)  # [B, N, H, C]

        # Aggregate point values then transform back to local
        o_vp_global = torch.einsum("bhnm,bmhpi->bnhpi", attn, vp)  # [B, N, H, Pv, 3]
        o_vp_centered = o_vp_global - t.view(B, N, 1, 1, 3)
        o_vp_local = torch.einsum(
            "bnji,bnhpj->bnhpi", R, o_vp_centered
        )  # R^T applied -> use R[..,j,i]
        o_vp_norm = o_vp_local.norm(dim=-1)  # [B, N, H, Pv]

        out = torch.cat(
            [
                o_scalar.reshape(B, N, H * C),
                o_vp_local.reshape(B, N, H * Pv * 3),
                o_vp_norm.reshape(B, N, H * Pv),
            ],
            dim=-1,
        )
        s_new = self.norm(s + self.dropout(self.lin_out(out)))

        upd = self.update(s_new)  # [B, N, 6]
        d_axis_angle = upd[..., :3]
        d_t_local = upd[..., 3:]
        return s_new, d_axis_angle, d_t_local


class BackboneModel(nn.Module):
    """ESM-2 frozen embedder + n_layers IPABlocks updating rigid frames."""

    def __init__(
        self,
        n_layers: int = 8,
        c_s: int | None = None,
        c_hidden: int = 16,
        n_heads: int = 4,
        n_qpoints: int = 4,
        n_vpoints: int = 8,
        dropout: float = 0.0,
        max_len: int = MAX_LEN,
        esm_name: str = ESM_NAME,
    ) -> None:
        super().__init__()
        self.embedder = ESMEmbedder(esm_name, max_len=max_len)
        cs = c_s if c_s is not None else self.embedder.c_s
        self.proj = (
            nn.Linear(self.embedder.c_s, cs)
            if cs != self.embedder.c_s
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                IPABlock(
                    cs,
                    c_hidden=c_hidden,
                    n_heads=n_heads,
                    n_qpoints=n_qpoints,
                    n_vpoints=n_vpoints,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.max_len = max_len

    def forward(
        self,
        sequences: list[str],
        mask: torch.Tensor,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        """Run the structure module.

        Returns:
            {
              "intermediate": list of (R, t) snapshots, length n_layers,
              "R": final R [B, N, 3, 3],
              "t": final t [B, N, 3],
            }
        """
        s = self.embedder(sequences)  # [B, N, c_s_esm]
        s = self.proj(s)
        B, N, _ = s.shape
        device = s.device

        R = torch.eye(3, device=device).expand(B, N, 3, 3).contiguous()
        t = torch.zeros(B, N, 3, device=device)

        intermediates: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            s, d_aa, d_tl = layer(s, R, t, mask)
            dR = _axis_angle_to_matrix(d_aa)  # [B, N, 3, 3]
            R_new = R @ dR
            # local translation update: world delta = R @ d_tl
            t_new = t + torch.einsum("bnij,bnj->bni", R, d_tl)
            intermediates.append((R_new, t_new))
            R = R_new.detach()  # stop rotation grads between layers
            t = t_new

        return {
            "intermediate": intermediates,
            "R": intermediates[-1][0],
            "t": intermediates[-1][1],
        }
