"""Histology branch: ABMIL over pre-extracted UNI patch features, plus the
zero-initialised gates that let tissue morphology modulate the curve.

Design rule (guide §5.4 / §5.5): a modality present in only one domain enters
through a gate whose scalar coefficient is initialised to exactly zero.  With
gamma = 0 the model is *bit-identical* to the model without that modality,
which is what makes the likelihood-ratio test in §7.3 valid — the two models
are exactly nested.  ``tests/test_gamma.py`` pins this down.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Gated attention MIL producing K prototype tokens per slide."""

    def __init__(
        self,
        feature_dim: int = 1024,
        d_model: int = 256,
        n_prototypes: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_prototypes = n_prototypes
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Sigmoid())
        self.attn_w = nn.Linear(d_model // 2, n_prototypes)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, patches: torch.Tensor, patch_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, N, F) patch features -> (B, K, d) tokens and (B, K, N) attention."""
        h = self.proj(patches)                       # (B, N, d)
        a = self.attn_w(self.attn_v(h) * self.attn_u(h))  # (B, N, K)
        if patch_mask is not None:
            a = a.masked_fill(~patch_mask.unsqueeze(-1), float("-inf"))
        a = a.transpose(1, 2)                        # (B, K, N)
        weights = F.softmax(a, dim=-1)
        weights = torch.nan_to_num(weights)          # slides with zero patches
        tokens = torch.bmm(weights, h)               # (B, K, d)
        return self.norm(tokens), weights


class HistologyGate(nn.Module):
    """gamma * h(H_i, z_ij) — a single scalar coefficient per gated parameter.

    ``gamma`` starts at exactly 0.0 and, in Stage 0 (GDSC, where cell lines have
    no slides), is frozen there.  Keeping ``gamma_E`` and ``gamma_m`` separate is
    the whole hypothesis: morphology is expected to act on efficacy, not potency.
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, histology_repr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(B, d) + (B, d) -> (B, 1) additive contribution, already scaled by gamma."""
        h = self.mlp(torch.cat([histology_repr, z], dim=-1))
        return self.gamma * h

    def set_trainable(self, flag: bool) -> None:
        self.gamma.requires_grad_(flag)
        for p in self.mlp.parameters():
            p.requires_grad_(flag)

    @torch.no_grad()
    def zero_gamma(self) -> None:
        self.gamma.zero_()


class HistologyBranch(nn.Module):
    """ABMIL + a z-conditioned pooling of the prototype tokens.

    Patients without a slide get a learned ``no_histology`` embedding.  That
    embedding is used *only inside the gated path*, so a missing slide can never
    change the ungated prediction.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        d_model: int = 256,
        n_prototypes: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.abmil = ABMIL(feature_dim, d_model, n_prototypes, dropout)
        self.query = nn.Linear(d_model, d_model)
        self.no_histology = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        patches: torch.Tensor | None,
        patch_mask: torch.Tensor | None,
        z: torch.Tensor,
        slide_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """-> (B, d) slide representation conditioned on z, and ABMIL attention."""
        if patches is None:
            return self.norm(self.no_histology.expand(z.shape[0], -1)), None
        tokens, attn = self.abmil(patches, patch_mask)          # (B, K, d)
        q = self.query(z).unsqueeze(1)                          # (B, 1, d)
        scores = (tokens * q).sum(-1) / (tokens.shape[-1] ** 0.5)  # (B, K)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bk,bkd->bd", weights, tokens)    # (B, d)
        if slide_available is not None:
            avail = slide_available.reshape(-1, 1).to(pooled.dtype)
            pooled = avail * pooled + (1.0 - avail) * self.no_histology.expand_as(pooled)
        return self.norm(pooled), attn
