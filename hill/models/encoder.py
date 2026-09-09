"""Shared encoder: pathway-grouped omics tokens + a drug token -> z_ij.

This part of the model is deliberately *not* a contribution.  Group-wise
pathway tokenisation follows SurvPath (CVPR 2024) / Pathformer (Bioinformatics
2024), and conditioning pathway attention on a drug token follows DRPreter
(IJMS 2022).  We reuse it and cite it; the novelty of this project lives
entirely in the output head and the likelihood (``curve_head.py``, ``losses.py``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupTokenizer(nn.Module):
    """One token per biological group, with a *separate* projection per group.

        t_a = LayerNorm( U_a . x[G_a] + e_a + m_mod(a) )

    ``U_a`` is independent per group (no shared MLP) and ``e_a`` is the group
    identity embedding — without it two pathways with identical scores would
    produce identical tokens and the model could not tell them apart.
    """

    def __init__(
        self,
        gene_index: np.ndarray,     # (G, M) int, -1 padded
        gene_mask: np.ndarray,      # (G, M) bool
        modality_id: np.ndarray,    # (G,) int, index into the modality embedding
        d_model: int,
        n_modalities: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        g, m = gene_index.shape
        self.n_groups, self.max_genes = int(g), int(m)
        self.d_model = d_model

        idx = np.clip(gene_index, 0, None).astype(np.int64)
        self.register_buffer("gene_index", torch.from_numpy(idx), persistent=True)
        self.register_buffer(
            "gene_mask", torch.from_numpy(gene_mask.astype(np.float32)), persistent=True
        )
        self.register_buffer(
            "modality_id", torch.from_numpy(modality_id.astype(np.int64)), persistent=True
        )

        # Per-group projection U_a: (G, M, d).  Scaled by 1/sqrt(group size) so
        # groups of very different sizes start on the same footing.
        scale = 1.0 / np.sqrt(np.maximum(gene_mask.sum(axis=1, keepdims=True), 1.0))
        w = torch.randn(g, m, d_model) * torch.from_numpy(
            (scale * 0.5).astype(np.float32)
        ).unsqueeze(-1)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(g, d_model))
        self.group_embed = nn.Parameter(torch.randn(g, d_model) * 0.02)
        self.modality_embed = nn.Embedding(max(n_modalities, 1), d_model)
        nn.init.normal_(self.modality_embed.weight, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, F) omics features -> (B, G, d) group tokens."""
        gathered = x[:, self.gene_index.reshape(-1)].view(
            x.shape[0], self.n_groups, self.max_genes
        )                                                   # (B, G, M)
        gathered = gathered * self.gene_mask                # zero the padding
        tokens = torch.einsum("bgm,gmd->bgd", gathered, self.weight) + self.bias
        tokens = tokens + self.group_embed + self.modality_embed(self.modality_id)
        return self.dropout(self.norm(tokens))


class LatentQueryTokenizer(nn.Module):
    """Tokens for features that have no biological grouping.

    Ungrouped features are first compressed into ``n_chunks`` block tokens
    (block-wise independent projections, same idea as a group), then a small set
    of learnable latent queries cross-attends over those blocks.  Direct
    attention over tens of thousands of raw features would be intractable.
    """

    def __init__(
        self,
        feature_index: np.ndarray,   # (F_u,) indices into the omics vector
        d_model: int,
        n_latent: int = 32,
        n_chunks: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        n_feat = int(feature_index.size)
        self.n_latent = int(n_latent)
        self.enabled = n_feat > 0 and n_latent > 0
        if not self.enabled:
            return
        n_chunks = int(max(1, min(n_chunks, n_feat)))
        chunk_size = int(np.ceil(n_feat / n_chunks))
        padded = np.full(n_chunks * chunk_size, -1, dtype=np.int64)
        padded[:n_feat] = feature_index.astype(np.int64)
        mask = (padded >= 0).reshape(n_chunks, chunk_size)
        self.register_buffer(
            "feature_index", torch.from_numpy(np.clip(padded, 0, None).reshape(n_chunks, chunk_size))
        )
        self.register_buffer("feature_mask", torch.from_numpy(mask.astype(np.float32)))
        self.chunk_weight = nn.Parameter(
            torch.randn(n_chunks, chunk_size, d_model) / np.sqrt(chunk_size)
        )
        self.chunk_bias = nn.Parameter(torch.zeros(n_chunks, d_model))
        self.chunk_embed = nn.Parameter(torch.randn(n_chunks, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """(B, F) -> (B, n_latent, d) or ``None`` when disabled."""
        if not self.enabled:
            return None
        b = x.shape[0]
        n_chunks, chunk_size = self.feature_index.shape
        gathered = x[:, self.feature_index.reshape(-1)].view(b, n_chunks, chunk_size)
        gathered = gathered * self.feature_mask
        chunks = torch.einsum("bcs,csd->bcd", gathered, self.chunk_weight) + self.chunk_bias
        chunks = chunks + self.chunk_embed
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, chunks, chunks, need_weights=False)
        return self.norm(out)


class AttentionPool(nn.Module):
    """Single-query attention pooling over the token axis."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, N, d) -> pooled (B, d) and attention (B, N)."""
        scores = self.score(tokens).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", weights, tokens)
        return pooled, weights


class OmicsDrugEncoder(nn.Module):
    """Transformer over [omics tokens ; drug token] with a type embedding.

    Returns the pooled pair representation ``z_ij`` used by every head.
    """

    N_TOKEN_TYPES = 3  # 0 = pathway/group token, 1 = latent token, 2 = drug token

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.type_embed = nn.Embedding(self.N_TOKEN_TYPES, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        layer.self_attn.dropout = attn_dropout
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool(d_model)

    def forward(
        self, tokens: torch.Tensor, token_types: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, N, d) + (N,) type ids -> z (B, d), pooling attention (B, N)."""
        h = tokens + self.type_embed(token_types).unsqueeze(0)
        h = self.encoder(h)
        h = self.norm(h)
        return self.pool(h)
