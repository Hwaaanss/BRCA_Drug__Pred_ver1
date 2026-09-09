"""Preprocessing must be fitted inside the fold (guide §4.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hill.data.datasets import PairData, Standardizer, fit_fold_scaler
from hill.data.tokenize import build_token_spec


def _pair_data(n_cells: int = 40, n_feat: int = 30, n_pairs: int = 120) -> PairData:
    rng = np.random.default_rng(0)
    omics = rng.normal(0, 1, (n_cells, n_feat)).astype(np.float32)
    # make the last 10 cells wildly different: if they leak into the scaler the
    # training statistics move detectably
    omics[30:] += 50.0
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs),
            "cell_id": rng.integers(0, n_cells, n_pairs).astype(str),
            "drug_id": rng.integers(0, 5, n_pairs),
        }
    )
    return PairData(
        pairs=pairs,
        log_conc=np.zeros((n_pairs, 4), dtype=np.float32),
        viability=np.zeros((n_pairs, 4), dtype=np.float32),
        point_mask=np.ones((n_pairs, 4), dtype=bool),
        omics=omics,
        cell_ids=[str(i) for i in range(n_cells)],
        cell_index=rng.integers(0, n_cells, n_pairs),
        drug_features=np.eye(5, dtype=np.float32),
        drug_ids=list(range(5)),
        drug_index=rng.integers(0, 5, n_pairs),
        token_spec=build_token_spec(
            [f"G{i}" for i in range(n_feat)], ["expression"] * n_feat,
            {"PW": [f"G{i}" for i in range(n_feat)]},
            target_n_tokens=4, min_genes_per_group=3, max_genes_per_group=8, n_latent=2,
        ),
    )


def test_scaler_fit_on_train_only():
    data = _pair_data()
    train_idx = np.flatnonzero(data.cell_index < 30)
    scaler = fit_fold_scaler(data, train_idx)
    train_cells = np.unique(data.cell_index[train_idx])
    expected = data.omics[train_cells].mean(axis=0)
    assert np.allclose(scaler.mean_, expected, atol=1e-5)
    assert scaler.n_fit_rows_ == train_cells.size
    # the held-out block is far away, so a leaked fit would be obvious
    all_mean = data.omics.mean(axis=0)
    assert not np.allclose(scaler.mean_, all_mean, atol=1.0)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        Standardizer().transform(np.zeros((2, 3)))


def test_transform_is_clipped_and_finite():
    s = Standardizer(clip=5.0).fit(np.random.default_rng(0).normal(size=(20, 4)))
    z = s.transform(np.array([[1e9, -1e9, np.nan, 0.0]]))
    assert np.isfinite(z).all()
    assert z.max() <= 5.0 and z.min() >= -5.0


def test_constant_feature_does_not_divide_by_zero():
    x = np.ones((10, 3))
    s = Standardizer().fit(x)
    assert np.all(s.std_ == 1.0)
    assert np.isfinite(s.transform(x)).all()
