"""Cross-validation splits.

Four schemes (guide §4.4):

    LPO   random (sample, drug) pairs         sanity check only
    LCO   sample held out entirely            PRIMARY metric
    LDO   drug held out entirely              needs a feature-based drug encoder
    XDOM  train on GDSC, evaluate on TCGA     domain transfer

Two invariants are enforced and tested:
    * every concentration point of a pair lands in the same fold — splitting at
      the measurement level would leak the pair's curve across folds;
    * the held-out group (cell line / drug) never appears in the training part.

Splits are written to disk and reloaded, so every run in the project — HPO
trial, ablation seed, figure — sees exactly the same partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hill.utils.logging import get_logger

log = get_logger("data.splits")

_GROUP_COLUMN = {"LPO": "pair_id", "LCO": "cell_id", "LDO": "drug_id"}


@dataclass
class Splits:
    """Fold assignment for one scheme over one pair table."""

    scheme: str
    n_folds: int
    seed: int
    pair_ids: np.ndarray          # (P,) pair_id in row order
    fold_of_pair: np.ndarray      # (P,) test fold index for each pair
    val_mask: np.ndarray          # (n_folds, P) True where the pair is validation
    meta: dict

    def test_idx(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_of_pair == fold)

    def val_idx(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.val_mask[fold])

    def train_idx(self, fold: int) -> np.ndarray:
        return np.flatnonzero((self.fold_of_pair != fold) & ~self.val_mask[fold])

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            pair_ids=self.pair_ids,
            fold_of_pair=self.fold_of_pair,
            val_mask=self.val_mask,
            json_blob=np.array(
                json.dumps(
                    {"scheme": self.scheme, "n_folds": self.n_folds, "seed": self.seed, "meta": self.meta}
                )
            ),
        )
        log.info("wrote splits: %s", p)

    @classmethod
    def load(cls, path: str | Path) -> "Splits":
        with np.load(Path(path), allow_pickle=False) as z:
            blob = json.loads(str(z["json_blob"]))
            return cls(
                scheme=blob["scheme"],
                n_folds=int(blob["n_folds"]),
                seed=int(blob["seed"]),
                pair_ids=z["pair_ids"],
                fold_of_pair=z["fold_of_pair"],
                val_mask=z["val_mask"],
                meta=blob.get("meta", {}),
            )


def _balanced_group_folds(
    groups: np.ndarray, n_folds: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign whole groups to folds, greedily balancing the number of pairs."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(uniq.size)
    uniq, counts = uniq[order], counts[order]
    order = np.argsort(-counts, kind="stable")   # largest groups first
    fold_load = np.zeros(n_folds, dtype=np.int64)
    group_fold: dict = {}
    for i in order:
        f = int(np.argmin(fold_load))
        group_fold[uniq[i]] = f
        fold_load[f] += counts[i]
    return np.array([group_fold[g] for g in groups], dtype=np.int16)


def build_splits(
    pairs: pd.DataFrame,
    scheme: str,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> Splits:
    """Construct a grouped k-fold split with an inner validation partition."""
    if scheme not in _GROUP_COLUMN:
        raise ValueError(f"build_splits does not handle scheme {scheme!r} (XDOM has no folds)")
    col = _GROUP_COLUMN[scheme]
    if col not in pairs.columns:
        raise KeyError(f"pair table lacks the {col!r} column needed for {scheme}")

    groups = pairs[col].to_numpy()
    rng = np.random.default_rng(seed)
    fold_of_pair = _balanced_group_folds(groups, n_folds, rng)

    n_pairs = len(pairs)
    val_mask = np.zeros((n_folds, n_pairs), dtype=bool)
    for f in range(n_folds):
        train_pos = np.flatnonzero(fold_of_pair != f)
        sub_rng = np.random.default_rng([seed, f])
        train_groups = np.unique(groups[train_pos])
        sub_rng.shuffle(train_groups)
        target = val_fraction * train_pos.size
        chosen: set = set()
        taken = 0
        sizes = pd.Series(groups[train_pos]).value_counts()
        for g in train_groups:
            if taken >= target:
                break
            chosen.add(g)
            taken += int(sizes.get(g, 0))
        sel = np.array([g in chosen for g in groups[train_pos]])
        val_mask[f, train_pos[sel]] = True

    meta = {
        "group_column": col,
        "n_pairs": int(n_pairs),
        "n_groups": int(np.unique(groups).size),
        "val_fraction_requested": float(val_fraction),
        "fold_sizes": [int((fold_of_pair == f).sum()) for f in range(n_folds)],
        "val_sizes": [int(val_mask[f].sum()) for f in range(n_folds)],
    }
    log.info(
        "%s split: %d pairs, %d groups, fold sizes %s",
        scheme, n_pairs, meta["n_groups"], meta["fold_sizes"],
    )
    return Splits(
        scheme=scheme,
        n_folds=n_folds,
        seed=seed,
        pair_ids=pairs["pair_id"].to_numpy(),
        fold_of_pair=fold_of_pair,
        val_mask=val_mask,
        meta=meta,
    )


def build_all_splits(
    pairs: pd.DataFrame,
    schemes: Sequence[str],
    out_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> dict[str, Splits]:
    """Build and persist every requested scheme (XDOM is handled separately)."""
    out: dict[str, Splits] = {}
    out_dir = Path(out_dir)
    for scheme in schemes:
        if scheme == "XDOM":
            continue
        sp = build_splits(pairs, scheme, n_folds=n_folds, seed=seed, val_fraction=val_fraction)
        sp.save(out_dir / f"{scheme}_{n_folds}fold_seed{seed}.npz")
        out[scheme] = sp
    return out


def load_or_build_splits(
    pairs: pd.DataFrame,
    scheme: str,
    out_dir: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> Splits:
    path = Path(out_dir) / f"{scheme}_{n_folds}fold_seed{seed}.npz"
    if path.exists():
        sp = Splits.load(path)
        if sp.pair_ids.size == len(pairs) and np.array_equal(sp.pair_ids, pairs["pair_id"].to_numpy()):
            return sp
        log.warning("cached split %s does not match the current pair table; rebuilding", path)
    sp = build_splits(pairs, scheme, n_folds=n_folds, seed=seed, val_fraction=val_fraction)
    sp.save(path)
    return sp


def check_leakage(pairs: pd.DataFrame, splits: Splits) -> dict[str, bool]:
    """Assertions used by ``tests/test_splits.py`` and by the audit report."""
    col = _GROUP_COLUMN[splits.scheme]
    groups = pairs[col].to_numpy()
    results = {"disjoint_folds": True, "group_disjoint": True, "val_disjoint": True}
    for f in range(splits.n_folds):
        test = set(splits.test_idx(f))
        train = set(splits.train_idx(f))
        val = set(splits.val_idx(f))
        if test & train or test & val or train & val:
            results["disjoint_folds"] = False
        if set(groups[list(test)]) & set(groups[list(train)]):
            results["group_disjoint"] = False
        if set(groups[list(val)]) & set(groups[list(test)]):
            results["val_disjoint"] = False
    return results
