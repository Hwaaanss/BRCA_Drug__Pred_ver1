"""Split integrity: a pair is atomic and a held-out group never leaks (guide §4.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hill.data.splits import Splits, build_splits, check_leakage


@pytest.fixture
def pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame(
        {
            "pair_id": np.arange(n),
            "cell_id": rng.integers(0, 50, n).astype(str),
            "drug_id": rng.integers(0, 20, n),
        }
    )


@pytest.mark.parametrize("scheme", ["LPO", "LCO", "LDO"])
def test_no_leakage_in_splits(pairs, scheme):
    splits = build_splits(pairs, scheme, n_folds=5, seed=42, val_fraction=0.15)
    result = check_leakage(pairs, splits)
    assert all(result.values()), result

    # every pair appears in exactly one test fold
    counts = np.bincount(splits.fold_of_pair, minlength=5)
    assert counts.sum() == len(pairs)
    for fold in range(5):
        idx = np.concatenate([splits.train_idx(fold), splits.val_idx(fold), splits.test_idx(fold)])
        assert np.array_equal(np.sort(idx), np.arange(len(pairs))), "partitions must cover every pair once"


def test_all_points_of_a_pair_stay_together(pairs):
    """Concentration points are keyed by pair_id, so folds are defined on pairs only."""
    splits = build_splits(pairs, "LCO", n_folds=5, seed=1)
    points = pd.DataFrame(
        {
            "pair_id": np.repeat(pairs["pair_id"].to_numpy(), 7),
            "conc": np.tile(np.arange(7), len(pairs)),
        }
    )
    fold_of_point = splits.fold_of_pair[points["pair_id"].to_numpy()]
    per_pair_folds = pd.Series(fold_of_point).groupby(points["pair_id"].to_numpy()).nunique()
    assert bool((per_pair_folds == 1).all()), "a pair's points landed in more than one fold"


def test_splits_are_deterministic_and_roundtrip(pairs, tmp_path):
    a = build_splits(pairs, "LCO", 5, 42, 0.15)
    b = build_splits(pairs, "LCO", 5, 42, 0.15)
    assert np.array_equal(a.fold_of_pair, b.fold_of_pair)
    assert np.array_equal(a.val_mask, b.val_mask)
    path = tmp_path / "s.npz"
    a.save(path)
    loaded = Splits.load(path)
    assert np.array_equal(loaded.fold_of_pair, a.fold_of_pair)
    assert loaded.scheme == a.scheme and loaded.seed == a.seed


def test_different_seed_changes_the_partition(pairs):
    a = build_splits(pairs, "LCO", 5, 42)
    b = build_splits(pairs, "LCO", 5, 7)
    assert not np.array_equal(a.fold_of_pair, b.fold_of_pair)
