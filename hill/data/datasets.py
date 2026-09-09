"""Torch datasets and fold-internal preprocessing.

Two rules that are easy to get wrong and are therefore enforced here:

1. **Scalers are fitted inside the fold.**  ``Standardizer.fit`` is only ever
   called with the training rows of the current fold (see
   ``tests/test_scaler.py``); fitting on the full matrix leaks test statistics.
2. **A pair is atomic.**  All K concentration points of one (cell, drug) pair
   live in one row of the padded arrays, so a pair cannot straddle two folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hill.data.tokenize import TokenSpec
from hill.utils.logging import get_logger

log = get_logger("data.datasets")


class Standardizer:
    """Mean/std standardiser with an explicit fitted flag and clipping."""

    def __init__(self, clip: float = 10.0) -> None:
        self.clip = clip
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_fit_rows_: int = 0

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError(f"Standardizer.fit expects a non-empty 2-D array, got {x.shape}")
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[~np.isfinite(std) | (std < 1e-8)] = 1.0
        self.std_ = std
        self.n_fit_rows_ = int(x.shape[0])
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Standardizer.transform called before fit — this would leak or crash")
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.std_
        z = np.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean_, "std": self.std_, "clip": self.clip, "n_fit_rows": self.n_fit_rows_}


@dataclass
class PairData:
    """Everything the source-domain (GDSC) model needs, already aligned by row."""

    pairs: pd.DataFrame            # (P,) pair-level table
    log_conc: np.ndarray           # (P, K) float32
    viability: np.ndarray          # (P, K) float32
    point_mask: np.ndarray         # (P, K) bool
    omics: np.ndarray              # (C, F) float32, raw (unscaled)
    cell_ids: list[str]            # (C,)
    cell_index: np.ndarray         # (P,) row into omics
    drug_features: np.ndarray      # (D, F_drug) float32
    drug_ids: list[int]            # (D,)
    drug_index: np.ndarray         # (P,) row into drug_features
    token_spec: TokenSpec
    meta: dict = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def max_points(self) -> int:
        return int(self.log_conc.shape[1])

    def subset_stats(self, idx: np.ndarray) -> dict[str, int]:
        return {
            "n_pairs": int(idx.size),
            "n_cells": int(np.unique(self.cell_index[idx]).size),
            "n_drugs": int(np.unique(self.drug_index[idx]).size),
            "n_points": int(self.point_mask[idx].sum()),
        }


def fit_fold_scaler(data: PairData, train_idx: np.ndarray, clip: float = 10.0) -> Standardizer:
    """Fit the omics standardiser on the cell lines seen in ``train_idx`` only."""
    train_cells = np.unique(data.cell_index[train_idx])
    if train_cells.size == 0:
        raise ValueError("no training cells — cannot fit a scaler")
    return Standardizer(clip=clip).fit(data.omics[train_cells])


class CurveDataset(Dataset):
    """Source-domain dataset: one item = one (cell, drug) pair with all its points."""

    def __init__(
        self,
        data: PairData,
        indices: np.ndarray,
        omics_scaled: np.ndarray,
        target_ln_ic50: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.omics = torch.from_numpy(np.ascontiguousarray(omics_scaled))
        self.drug_features = torch.from_numpy(np.ascontiguousarray(data.drug_features))
        self.log_conc = torch.from_numpy(data.log_conc)
        self.viability = torch.from_numpy(data.viability)
        self.point_mask = torch.from_numpy(data.point_mask)
        self.target_ln_ic50 = (
            torch.from_numpy(np.asarray(target_ln_ic50, dtype=np.float32))
            if target_ln_ic50 is not None
            else None
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        p = int(self.indices[i])
        item = {
            "pair_row": torch.tensor(p, dtype=torch.long),
            "omics": self.omics[int(self.data.cell_index[p])],
            "drug_features": self.drug_features[int(self.data.drug_index[p])],
            "drug_index": torch.tensor(int(self.data.drug_index[p]), dtype=torch.long),
            "log_conc": self.log_conc[p],
            "viability": self.viability[p],
            "point_mask": self.point_mask[p],
        }
        if self.target_ln_ic50 is not None:
            item["ln_ic50_target"] = self.target_ln_ic50[p]
        return item


def collate_pairs(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class ClinicalDataset(Dataset):
    """Target-domain dataset: TCGA patient x administered drug, with optional WSI."""

    def __init__(
        self,
        table: pd.DataFrame,          # patient_id, drug_index, responder, log_cmax
        omics_scaled: np.ndarray,     # (N_patients, F)
        patient_index: np.ndarray,    # (rows,) into omics
        drug_features: np.ndarray,    # (D, F_drug)
        histology_dir: str | Path | None = None,
        max_patches: int = 2048,
        histo_feature_dim: int = 1024,
        rng_seed: int = 0,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.omics = torch.from_numpy(np.ascontiguousarray(omics_scaled))
        self.patient_index = np.asarray(patient_index, dtype=np.int64)
        self.drug_features = torch.from_numpy(np.ascontiguousarray(drug_features))
        self.histology_dir = Path(histology_dir) if histology_dir else None
        self.max_patches = int(max_patches)
        self.histo_feature_dim = int(histo_feature_dim)
        self._rng = np.random.default_rng(rng_seed)

    def __len__(self) -> int:
        return len(self.table)

    def _load_histology(self, patient_id: str) -> torch.Tensor | None:
        if self.histology_dir is None:
            return None
        for suffix in (".npy", ".pt"):
            path = self.histology_dir / f"{patient_id}{suffix}"
            if path.exists():
                feats = (
                    torch.from_numpy(np.load(path))
                    if suffix == ".npy"
                    else torch.load(path, map_location="cpu", weights_only=True)
                )
                feats = feats.float()
                if feats.shape[0] > self.max_patches:
                    sel = self._rng.choice(feats.shape[0], self.max_patches, replace=False)
                    feats = feats[np.sort(sel)]
                return feats
        return None

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.table.iloc[i]
        histo = self._load_histology(str(row["patient_id"]))
        item: dict[str, Any] = {
            "omics": self.omics[int(self.patient_index[i])],
            "drug_features": self.drug_features[int(row["drug_index"])],
            "drug_index": torch.tensor(int(row["drug_index"]), dtype=torch.long),
            "log_cmax": torch.tensor(float(row["log_cmax"]), dtype=torch.float32),
            "responder": torch.tensor(float(row["responder"]), dtype=torch.float32),
            "slide_available": torch.tensor(float(histo is not None), dtype=torch.float32),
            "histology": histo,
        }
        return item


def collate_clinical(batch: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack clinical items, padding the variable-length patch dimension."""
    out: dict[str, torch.Tensor] = {
        k: torch.stack([b[k] for b in batch])
        for k in ("omics", "drug_features", "drug_index", "log_cmax", "responder", "slide_available")
    }
    present = [b["histology"] for b in batch if b["histology"] is not None]
    if present:
        max_n = max(h.shape[0] for h in present)
        dim = present[0].shape[1]
        histo = torch.zeros(len(batch), max_n, dim)
        mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
        for i, b in enumerate(batch):
            h = b["histology"]
            if h is not None and h.shape[0] > 0:
                histo[i, : h.shape[0]] = h
                mask[i, : h.shape[0]] = True
        out["histology"] = histo
        out["histo_mask"] = mask
    return out


def make_loaders(
    data: PairData,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Any,
    target_ln_ic50: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[Any, Any, Any, Standardizer]:
    """Build train/val/test loaders with a scaler fitted on the training fold only."""
    from torch.utils.data import DataLoader

    from hill.utils.resources import dataloader_workers
    from hill.utils.seed import seed_worker

    scaler = fit_fold_scaler(data, train_idx)
    omics_scaled = scaler.transform(data.omics)

    workers = dataloader_workers(cfg)
    common = dict(
        num_workers=workers,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_pairs,
        persistent_workers=bool(cfg.train.persistent_workers and workers > 0),
    )
    if workers > 0:
        common["prefetch_factor"] = cfg.train.prefetch_factor

    gen = torch.Generator()
    gen.manual_seed(seed)

    def build(idx: np.ndarray, shuffle: bool, batch_size: int):
        ds = CurveDataset(data, idx, omics_scaled, target_ln_ic50)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            worker_init_fn=seed_worker if workers > 0 else None,
            generator=gen if shuffle else None,
            **common,
        )

    return (
        build(train_idx, True, cfg.train.batch_size),
        build(val_idx, False, cfg.train.eval_batch_size),
        build(test_idx, False, cfg.train.eval_batch_size),
        scaler,
    )
