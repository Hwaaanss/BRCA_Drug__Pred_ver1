"""Shared fixtures. Everything here is tiny and CPU-only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hill.config import Config
from hill.data.tokenize import build_token_spec


@pytest.fixture(scope="session", autouse=True)
def _cpu_only():
    torch.set_num_threads(2)
    torch.manual_seed(0)


@pytest.fixture
def small_config() -> Config:
    return Config().copy_with(
        [
            "device=cpu",
            "model.d_model=32",
            "model.n_layers=2",
            "model.n_heads=4",
            "model.dropout=0.0",
            "model.attn_dropout=0.0",
            "model.head_dropout=0.0",
            "data.drug_features=onehot",
            "train.amp_dtype=none",
        ]
    )


@pytest.fixture
def token_spec():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(120)]
    names = genes + [f"{g}_mut" for g in genes[:40]]
    mods = ["expression"] * 120 + ["mutation"] * 40
    gene_sets = {
        f"PW{k}": list(rng.choice(genes, size=int(rng.integers(8, 25)), replace=False))
        for k in range(24)
    }
    return build_token_spec(
        names, mods, gene_sets, target_n_tokens=24, min_genes_per_group=5,
        max_genes_per_group=16, n_latent=4, feature_variance=rng.random(len(names)),
    )


@pytest.fixture
def curve_params():
    """1000 random admissible (E_inf, s, m) triples."""
    g = torch.Generator().manual_seed(7)
    e = torch.rand(1000, 1, generator=g) * 0.95
    s = torch.rand(1000, 1, generator=g) * 4.0 + 1e-3
    m = torch.randn(1000, 1, generator=g) * 2.0
    return e, s, m
