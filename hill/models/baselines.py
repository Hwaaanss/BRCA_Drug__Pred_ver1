"""Baselines, all trained on the same folds and the same preprocessing.

    NaiveMeanEffects   drug mean + sample mean (DrEval's sanity floor and the
                       denominator of the project's primary metric)
    ElasticNet / RF    classic per-drug regressors on the omics features
    MOLI               per-drug multi-omics late integration + triplet loss
                       (Sharifi-Noghabi et al., Bioinformatics 2019)
    SuperFELT          per-drug: encoders trained separately with a triplet
                       loss, then frozen, then a regressor on top
                       (Park et al., BMC Bioinformatics 2021)

``ScalarHILL`` is *not* here: it is the HILL model with ``model.head=scalar``,
so it shares the encoder, the data and the splits with HILL exactly, and the
HILL - ScalarHILL difference isolates the head and the likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hill.utils.logging import get_logger

log = get_logger("baselines")


class BaselineModel:
    """Common interface: fit on training pairs, predict ln IC50 for any pairs."""

    name = "baseline"

    def fit(self, omics: np.ndarray, cell_index: np.ndarray, drug_index: np.ndarray,
            y: np.ndarray, train_idx: np.ndarray) -> "BaselineModel":
        raise NotImplementedError

    def predict(self, omics: np.ndarray, cell_index: np.ndarray, drug_index: np.ndarray,
                idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NaiveMeanEffects(BaselineModel):
    """ln IC50 ~ global mean + drug effect + sample effect.

    Under LCO the held-out cell line has no estimable sample effect, so the
    prediction degrades to (global + drug) — which is exactly the point: any
    model that cannot beat this has not learned anything about the sample.
    """

    name = "naive"

    def __init__(self) -> None:
        self.global_mean = 0.0
        self.drug_effect: dict[int, float] = {}
        self.cell_effect: dict[int, float] = {}

    def fit(self, omics, cell_index, drug_index, y, train_idx) -> "NaiveMeanEffects":
        yt = np.asarray(y, dtype=float)[train_idx]
        dt = np.asarray(drug_index)[train_idx]
        ct = np.asarray(cell_index)[train_idx]
        ok = np.isfinite(yt)
        yt, dt, ct = yt[ok], dt[ok], ct[ok]
        if yt.size == 0:
            raise ValueError("NaiveMeanEffects.fit: no finite training targets")
        self.global_mean = float(yt.mean())
        resid = yt - self.global_mean
        for d in np.unique(dt):
            self.drug_effect[int(d)] = float(resid[dt == d].mean())
        resid2 = resid - np.array([self.drug_effect[int(d)] for d in dt])
        for c in np.unique(ct):
            self.cell_effect[int(c)] = float(resid2[ct == c].mean())
        return self

    def predict(self, omics, cell_index, drug_index, idx) -> np.ndarray:
        d = np.asarray(drug_index)[idx]
        c = np.asarray(cell_index)[idx]
        return np.array(
            [
                self.global_mean
                + self.drug_effect.get(int(di), 0.0)
                + self.cell_effect.get(int(ci), 0.0)
                for di, ci in zip(d, c)
            ],
            dtype=float,
        )


@dataclass
class _PerDrugState:
    models: dict[int, Any] = field(default_factory=dict)
    fallback: dict[int, float] = field(default_factory=dict)


class _PerDrugSklearn(BaselineModel):
    """Shared machinery for the per-drug scikit-learn baselines.

    Omics are reduced with a PCA fitted *inside the training fold* — 12k raw
    features against a few hundred cell lines would otherwise make ElasticNet
    and RandomForest both slow and degenerate.
    """

    def __init__(self, n_components: int = 256, min_pairs: int = 10, seed: int = 0, n_jobs: int = 8) -> None:
        self.n_components = n_components
        self.min_pairs = min_pairs
        self.seed = seed
        self.n_jobs = n_jobs
        self.state = _PerDrugState()
        self.pca = None
        self.global_mean = 0.0

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, omics, cell_index, drug_index, y, train_idx) -> "_PerDrugSklearn":
        from sklearn.decomposition import PCA

        y = np.asarray(y, dtype=float)
        train_cells = np.unique(np.asarray(cell_index)[train_idx])
        n_comp = int(min(self.n_components, train_cells.size - 1, omics.shape[1]))
        n_comp = max(n_comp, 2)
        self.pca = PCA(n_components=n_comp, random_state=self.seed).fit(omics[train_cells])
        z_all = self.pca.transform(omics)

        yt = y[train_idx]
        self.global_mean = float(np.nanmean(yt)) if np.isfinite(yt).any() else 0.0
        for d in np.unique(np.asarray(drug_index)[train_idx]):
            sel = train_idx[(np.asarray(drug_index)[train_idx] == d) & np.isfinite(yt)]
            if sel.size < self.min_pairs:
                self.state.fallback[int(d)] = float(np.nanmean(y[sel])) if sel.size else self.global_mean
                continue
            x = z_all[np.asarray(cell_index)[sel]]
            est = self._make_estimator().fit(x, y[sel])
            self.state.models[int(d)] = est
            self.state.fallback[int(d)] = float(np.nanmean(y[sel]))
        return self

    def predict(self, omics, cell_index, drug_index, idx) -> np.ndarray:
        z = self.pca.transform(omics)
        d = np.asarray(drug_index)[idx]
        c = np.asarray(cell_index)[idx]
        out = np.empty(idx.size, dtype=float)
        for i, (di, ci) in enumerate(zip(d, c)):
            est = self.state.models.get(int(di))
            if est is None:
                out[i] = self.state.fallback.get(int(di), self.global_mean)
            else:
                out[i] = float(est.predict(z[ci : ci + 1])[0])
        return out


class ElasticNetBaseline(_PerDrugSklearn):
    name = "elasticnet"

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, **kw: Any) -> None:
        super().__init__(**kw)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _make_estimator(self):
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000, random_state=self.seed)


class RandomForestBaseline(_PerDrugSklearn):
    name = "randomforest"

    def __init__(self, n_estimators: int = 200, **kw: Any) -> None:
        super().__init__(**kw)
        self.n_estimators = n_estimators

    def _make_estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=self.n_jobs, min_samples_leaf=2
        )


# ---------------------------------------------------------------------------
# Deep baselines
# ---------------------------------------------------------------------------


class _ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _triplet_loss(z: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Batch-hard triplet loss on a median split of the drug's responses.

    MOLI and SuperFELT both use a triplet loss over sensitive/resistant labels;
    for a regression target we binarise at the training median, as is standard
    when these models are adapted to continuous IC50.
    """
    if z.shape[0] < 4:
        return z.new_zeros(())
    labels = (y > y.median()).long()
    if labels.unique().numel() < 2:
        return z.new_zeros(())
    d = torch.cdist(z, z)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    pos = torch.where(same & ~eye, d, torch.zeros_like(d)).max(dim=1).values
    neg = torch.where(~same, d, torch.full_like(d, float("inf"))).min(dim=1).values
    valid = torch.isfinite(neg)
    if valid.sum() == 0:
        return z.new_zeros(())
    return F.relu(pos[valid] - neg[valid] + margin).mean()


class _PerDrugTorch(BaselineModel):
    """Per-drug multi-omics network trained on the fold's cell lines."""

    def __init__(
        self,
        modality_slices: Sequence[tuple[str, int, int]],
        hidden: int = 128,
        embed: int = 32,
        dropout: float = 0.3,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        triplet_weight: float = 0.5,
        min_pairs: int = 20,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.slices = list(modality_slices)
        self.hidden, self.embed, self.dropout = hidden, embed, dropout
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.triplet_weight = triplet_weight
        self.min_pairs = min_pairs
        self.seed = seed
        self.device = torch.device(device)
        self.models: dict[int, nn.Module] = {}
        self.fallback: dict[int, float] = {}
        self.global_mean = 0.0

    def _build(self) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def _train_one(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def fit(self, omics, cell_index, drug_index, y, train_idx) -> "_PerDrugTorch":
        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=float)
        yt = y[train_idx]
        self.global_mean = float(np.nanmean(yt)) if np.isfinite(yt).any() else 0.0
        drugs = np.asarray(drug_index)
        cells = np.asarray(cell_index)
        for d in np.unique(drugs[train_idx]):
            sel = train_idx[(drugs[train_idx] == d) & np.isfinite(yt)]
            self.fallback[int(d)] = float(np.nanmean(y[sel])) if sel.size else self.global_mean
            if sel.size < self.min_pairs:
                continue
            x = torch.tensor(omics[cells[sel]], dtype=torch.float32, device=self.device)
            target = torch.tensor(y[sel], dtype=torch.float32, device=self.device)
            # One small network per drug: keep them on the host, otherwise a few
            # hundred drugs would pin several GB of GPU memory for no reason.
            self.models[int(d)] = self._train_one(x, target).to("cpu")
            del x, target
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict(self, omics, cell_index, drug_index, idx) -> np.ndarray:
        drugs = np.asarray(drug_index)[idx]
        cells = np.asarray(cell_index)[idx]
        out = np.empty(idx.size, dtype=float)
        for d in np.unique(drugs):
            sel = np.flatnonzero(drugs == d)
            model = self.models.get(int(d))
            if model is None:
                out[sel] = self.fallback.get(int(d), self.global_mean)
                continue
            model.eval().to(self.device)
            x = torch.tensor(omics[cells[sel]], dtype=torch.float32, device=self.device)
            out[sel] = model(x)[0].squeeze(-1).cpu().numpy()
            model.to("cpu")
        return out


class _MOLINet(nn.Module):
    def __init__(self, slices, hidden, embed, dropout) -> None:
        super().__init__()
        self.slices = slices
        self.encoders = nn.ModuleList(
            [_ModalityEncoder(hi - lo, hidden, embed, dropout) for _, lo, hi in slices]
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed * len(slices), 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([enc(x[:, lo:hi]) for enc, (_, lo, hi) in zip(self.encoders, self.slices)], dim=1)
        return self.head(z), z


class MOLIBaseline(_PerDrugTorch):
    """Late integration: one encoder per modality, concatenated, MSE + triplet."""

    name = "moli"

    def _train_one(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        model = _MOLINet(self.slices, self.hidden, self.embed, self.dropout).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred, z = model(x)
            loss = F.mse_loss(pred.squeeze(-1), y) + self.triplet_weight * _triplet_loss(z, y)
            loss.backward()
            opt.step()
        return model


class _SuperFELTNet(nn.Module):
    def __init__(self, encoders: nn.ModuleList, slices, embed, dropout) -> None:
        super().__init__()
        self.encoders = encoders
        self.slices = slices
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed * len(slices), 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            z = torch.cat(
                [enc(x[:, lo:hi]) for enc, (_, lo, hi) in zip(self.encoders, self.slices)], dim=1
            )
        return self.head(z), z


class SuperFELTBaseline(_PerDrugTorch):
    """Encoders trained separately with a triplet loss, frozen, then a regressor."""

    name = "superfelt"

    def _train_one(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        encoders = nn.ModuleList(
            [_ModalityEncoder(hi - lo, self.hidden, self.embed, self.dropout).to(self.device)
             for _, lo, hi in self.slices]
        )
        for enc, (_, lo, hi) in zip(encoders, self.slices):
            opt = torch.optim.AdamW(enc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            enc.train()
            for _ in range(self.epochs):
                opt.zero_grad(set_to_none=True)
                loss = _triplet_loss(enc(x[:, lo:hi]), y)
                if not loss.requires_grad:
                    break
                loss.backward()
                opt.step()
        for p in encoders.parameters():
            p.requires_grad_(False)
        model = _SuperFELTNet(encoders, self.slices, self.embed, self.dropout).to(self.device)
        opt = torch.optim.AdamW(model.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x)
            F.mse_loss(pred.squeeze(-1), y).backward()
            opt.step()
        return model


def modality_slices(feature_modality: Sequence[str]) -> list[tuple[str, int, int]]:
    """Contiguous [start, end) index ranges per modality in the omics vector."""
    out: list[tuple[str, int, int]] = []
    start = 0
    current = feature_modality[0] if len(feature_modality) else "expression"
    for i, m in enumerate(feature_modality):
        if m != current:
            out.append((current, start, i))
            current, start = m, i
    out.append((current, start, len(feature_modality)))
    return out


def build_baseline(name: str, cfg: Any, feature_modality: Sequence[str], seed: int = 0) -> BaselineModel:
    n_jobs = max(1, cfg.resources.torch_threads - 1)
    if name == "naive":
        return NaiveMeanEffects()
    if name == "elasticnet":
        return ElasticNetBaseline(seed=seed, n_jobs=n_jobs)
    if name == "randomforest":
        return RandomForestBaseline(seed=seed, n_jobs=n_jobs)
    if name in {"moli", "superfelt"}:
        slices = modality_slices(feature_modality)
        device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        cls = MOLIBaseline if name == "moli" else SuperFELTBaseline
        return cls(slices, seed=seed, device=device)
    raise ValueError(f"unknown baseline {name!r}")
