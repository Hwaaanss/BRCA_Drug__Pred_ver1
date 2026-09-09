"""Tokenisation: independent projections, group identity, and no feature leakage."""

from __future__ import annotations

import numpy as np
import torch

from hill.data.tokenize import TokenSpec, read_gmt
from hill.models.encoder import GroupTokenizer, LatentQueryTokenizer


def test_token_spec_respects_size_filters(token_spec):
    sizes = token_spec.gene_mask.sum(axis=1)
    assert sizes.min() >= 5
    assert sizes.max() <= 16
    assert token_spec.gene_index[token_spec.gene_mask].max() < token_spec.n_features


def test_token_spec_roundtrip(token_spec, tmp_path):
    path = tmp_path / "spec.npz"
    token_spec.save(path)
    loaded = TokenSpec.load(path)
    assert loaded.group_names == token_spec.group_names
    assert np.array_equal(loaded.gene_index, token_spec.gene_index)
    assert loaded.n_latent == token_spec.n_latent


def test_group_projections_are_independent(token_spec):
    """Each group owns its own U_a: touching one group cannot move another token."""
    tok = GroupTokenizer(token_spec.gene_index, token_spec.gene_mask, token_spec.modality_id,
                         d_model=16, n_modalities=len(token_spec.modality_names))
    tok.eval()
    x = torch.randn(2, token_spec.n_features)
    with torch.no_grad():
        before = tok(x)
        tok.weight[0] += 5.0
        after = tok(x)
    assert not torch.allclose(before[:, 0], after[:, 0])
    assert torch.allclose(before[:, 1:], after[:, 1:]), "groups must not share parameters"


def test_group_identity_embedding_separates_identical_scores(token_spec):
    """Two groups with the same input score must still produce different tokens."""
    tok = GroupTokenizer(token_spec.gene_index, token_spec.gene_mask, token_spec.modality_id,
                         d_model=16, n_modalities=len(token_spec.modality_names))
    tok.eval()
    with torch.no_grad():
        tok.weight.zero_()          # remove the input pathway entirely
        tokens = tok(torch.randn(1, token_spec.n_features))
    assert not torch.allclose(tokens[0, 0], tokens[0, 1]), "group identity embedding is missing"


def test_padding_contributes_nothing():
    gene_index = np.array([[0, 1, -1, -1]])
    gene_mask = np.array([[True, True, False, False]])
    tok = GroupTokenizer(gene_index, gene_mask, np.zeros(1, dtype=int), d_model=8, n_modalities=1)
    tok.eval()
    x = torch.zeros(1, 5)
    x[0, 0] = 1.0
    with torch.no_grad():
        a = tok(x)
        x2 = x.clone()
        x2[0, 2:] = 1000.0          # only features outside the group change
        b = tok(x2)
    assert torch.allclose(a, b)


def test_latent_tokenizer_shapes_and_disabling():
    lat = LatentQueryTokenizer(np.arange(40), d_model=8, n_latent=3, n_chunks=5, n_heads=2)
    lat.eval()
    with torch.no_grad():
        out = lat(torch.randn(2, 50))
    assert out.shape == (2, 3, 8)
    empty = LatentQueryTokenizer(np.array([], dtype=int), d_model=8, n_latent=3)
    assert empty(torch.randn(2, 50)) is None


def test_read_gmt(tmp_path):
    path = tmp_path / "sets.gmt"
    path.write_text("PW1\tdesc\tTP53\tEGFR\nPW2\tdesc\tMYC\n", encoding="utf-8")
    sets = read_gmt(path)
    assert sets == {"PW1": ["EGFR", "TP53"], "PW2": ["MYC"]}
