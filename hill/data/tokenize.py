"""Group-wise tokenisation of the omics vector.

Produces a :class:`TokenSpec`: the static description of which features feed
which token, which modality each token belongs to, and which features are left
over for the latent-query tokens.  The spec is built once from the training
feature universe and saved next to the processed data, so every run — HPO
trial, ablation seed, figure — tokenises identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from hill.utils.logging import get_logger

log = get_logger("tokenize")


def read_gmt(path: str | Path) -> dict[str, list[str]]:
    """Parse a GMT gene-set file: ``name <tab> description <tab> gene1 gene2 ...``."""
    sets: dict[str, list[str]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"gene set file not found: {p}. Download it with "
            "`python -m hill.data.download --what pathways`."
        )
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = [g.strip().upper() for g in parts[2:] if g.strip()]
            if genes:
                sets[name] = sorted(set(genes))
    if not sets:
        raise ValueError(f"no gene sets parsed from {p}")
    return sets


@dataclass
class TokenSpec:
    """Static tokenisation plan for one omics feature universe."""

    feature_names: list[str]
    feature_modality: list[str]
    group_names: list[str]
    group_modality: list[str]
    modality_names: list[str]
    gene_index: np.ndarray      # (G, M) int64, padded with -1
    gene_mask: np.ndarray       # (G, M) bool
    modality_id: np.ndarray     # (G,) int64
    ungrouped_index: np.ndarray  # (F_u,) int64
    n_latent: int
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_groups(self) -> int:
        return len(self.group_names)

    @property
    def n_tokens(self) -> int:
        """Total omics tokens (groups + latent). The drug token is added by the model."""
        return self.n_groups + (self.n_latent if self.ungrouped_index.size else 0)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            gene_index=self.gene_index,
            gene_mask=self.gene_mask,
            modality_id=self.modality_id,
            ungrouped_index=self.ungrouped_index,
            json_blob=np.array(
                json.dumps(
                    {
                        "feature_names": self.feature_names,
                        "feature_modality": self.feature_modality,
                        "group_names": self.group_names,
                        "group_modality": self.group_modality,
                        "modality_names": self.modality_names,
                        "n_latent": self.n_latent,
                        "meta": self.meta,
                    }
                )
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokenSpec":
        with np.load(Path(path), allow_pickle=False) as z:
            blob = json.loads(str(z["json_blob"]))
            return cls(
                feature_names=blob["feature_names"],
                feature_modality=blob["feature_modality"],
                group_names=blob["group_names"],
                group_modality=blob["group_modality"],
                modality_names=blob["modality_names"],
                gene_index=z["gene_index"],
                gene_mask=z["gene_mask"],
                modality_id=z["modality_id"],
                ungrouped_index=z["ungrouped_index"],
                n_latent=int(blob["n_latent"]),
                meta=blob.get("meta", {}),
            )


def _gene_symbol(feature_name: str) -> str:
    """``TP53_mut`` / ``TP53_cnv`` / ``TP53`` -> ``TP53``."""
    base = feature_name.split("__")[0]
    for suffix in ("_mut", "_cnv", "_exp", "_rna", "_prot"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.upper()


def build_token_spec(
    feature_names: Sequence[str],
    feature_modality: Sequence[str],
    gene_sets: Mapping[str, Iterable[str]],
    target_n_tokens: int = 400,
    min_genes_per_group: int = 10,
    max_genes_per_group: int = 128,
    n_latent: int = 32,
    feature_variance: np.ndarray | None = None,
    modality_budget: Mapping[str, float] | None = None,
) -> TokenSpec:
    """Assign features to pathway groups, respecting a total token budget.

    Groups are ranked per modality by ``n_members * mean_feature_variance`` so
    that the budget goes to pathways that are both well covered and variable in
    this dataset.  Oversized pathways are truncated to their most variable
    ``max_genes_per_group`` members; undersized ones are dropped.
    """
    feature_names = list(feature_names)
    feature_modality = list(feature_modality)
    if len(feature_names) != len(feature_modality):
        raise ValueError("feature_names and feature_modality must have the same length")
    n_feat = len(feature_names)
    var = (
        np.ones(n_feat, dtype=np.float64)
        if feature_variance is None
        else np.nan_to_num(np.asarray(feature_variance, dtype=np.float64), nan=0.0)
    )

    modality_names = sorted(set(feature_modality))
    mod_of_feature = np.array([modality_names.index(m) for m in feature_modality], dtype=np.int64)

    # symbol -> feature indices, per modality
    symbol_map: dict[tuple[int, str], list[int]] = {}
    for i, (name, mod) in enumerate(zip(feature_names, feature_modality)):
        symbol_map.setdefault((modality_names.index(mod), _gene_symbol(name)), []).append(i)

    if modality_budget is None:
        default_share = {"expression": 0.6, "mutation": 0.2, "cnv": 0.2, "proteomic": 0.2}
        modality_budget = {m: default_share.get(m, 1.0) for m in modality_names}
    total_share = sum(modality_budget.get(m, 0.0) for m in modality_names) or 1.0

    latent_budget = n_latent if n_latent > 0 else 0
    group_budget = max(1, target_n_tokens - latent_budget)

    groups: list[tuple[str, int, list[int], float]] = []  # (name, modality_id, indices, score)
    for mod_idx, mod in enumerate(modality_names):
        share = modality_budget.get(mod, 0.0) / total_share
        quota = max(1, int(round(group_budget * share)))
        candidates: list[tuple[float, str, list[int]]] = []
        for set_name, genes in gene_sets.items():
            members: list[int] = []
            for g in genes:
                members.extend(symbol_map.get((mod_idx, g.upper()), ()))
            if len(members) < min_genes_per_group:
                continue
            members = sorted(set(members))
            if len(members) > max_genes_per_group:
                members = list(np.asarray(members)[np.argsort(-var[members])[:max_genes_per_group]])
                members.sort()
            score = float(len(members) * var[members].mean())
            candidates.append((score, set_name, members))
        candidates.sort(key=lambda t: -t[0])
        for score, set_name, members in candidates[:quota]:
            groups.append((f"{set_name}|{mod}", mod_idx, members, score))

    if not groups:
        raise ValueError(
            "no gene set matched the feature universe — check that feature names are gene "
            "symbols and that the GMT file uses the same symbol namespace"
        )

    max_m = max(len(g[2]) for g in groups)
    gene_index = np.full((len(groups), max_m), -1, dtype=np.int64)
    gene_mask = np.zeros((len(groups), max_m), dtype=bool)
    for gi, (_, _, members, _) in enumerate(groups):
        gene_index[gi, : len(members)] = members
        gene_mask[gi, : len(members)] = True

    covered = np.zeros(n_feat, dtype=bool)
    covered[gene_index[gene_mask]] = True
    ungrouped = np.flatnonzero(~covered).astype(np.int64)

    spec = TokenSpec(
        feature_names=feature_names,
        feature_modality=feature_modality,
        group_names=[g[0] for g in groups],
        group_modality=[modality_names[g[1]] for g in groups],
        modality_names=modality_names,
        gene_index=gene_index,
        gene_mask=gene_mask,
        modality_id=np.array([g[1] for g in groups], dtype=np.int64),
        ungrouped_index=ungrouped,
        n_latent=latent_budget,
        meta={
            "target_n_tokens": int(target_n_tokens),
            "n_gene_sets_available": len(gene_sets),
            "min_genes_per_group": int(min_genes_per_group),
            "max_genes_per_group": int(max_genes_per_group),
            "coverage_fraction": float(covered.mean()),
            "modality_feature_counts": {
                m: int((mod_of_feature == i).sum()) for i, m in enumerate(modality_names)
            },
        },
    )
    log.info(
        "token spec: %d group tokens + %d latent tokens (target %d); %.1f%% of %d features grouped",
        spec.n_groups, spec.n_latent if ungrouped.size else 0, target_n_tokens,
        100 * covered.mean(), n_feat,
    )
    return spec
