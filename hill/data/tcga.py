"""TCGA target domain: omics alignment, clinical response labels, Cmax.

The transfer only works if the patient omics vector lives in the *same* feature
space as the cell-line vector the encoder was trained on, so
:func:`align_to_feature_universe` projects TCGA matrices onto the cell-line
feature list and reports coverage instead of silently zero-filling everything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from hill.utils.logging import get_logger

log = get_logger("data.tcga")

RESPONDER_LABELS = {
    "complete response": 1,
    "partial response": 1,
    "stable disease": 0,
    "progressive disease": 0,
    "clinical progressive disease": 0,
    "radiographic progressive disease": 0,
}


@dataclass
class TCGAData:
    omics: np.ndarray                 # (N, F) aligned to the cell-line feature universe
    patient_ids: list[str]
    clinical: pd.DataFrame            # patient_id, drug_id, drug_index, responder, log_cmax
    feature_coverage: dict[str, float]
    histology_dir: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)


def _strip_suffix(name: str) -> str:
    base = str(name).split("__")[0]
    for suffix in ("_mut", "_cnv", "_exp", "_rna", "_prot"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    return base


def align_to_feature_universe(
    matrices: Mapping[str, pd.DataFrame],
    feature_names: Sequence[str],
    feature_modality: Sequence[str],
    log1p_expression: bool = True,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Project per-modality patient matrices onto the training feature universe.

    Missing features are zero-filled *after* standardisation upstream, and the
    per-modality coverage is returned so a poorly covered modality is reported
    rather than quietly dominating the input with zeros.
    """
    patients = None
    for df in matrices.values():
        idx = set(df.index.astype(str))
        patients = idx if patients is None else (patients & idx)
    if not patients:
        raise ValueError("no patient is present in every TCGA modality")
    patient_ids = sorted(patients)

    out = np.zeros((len(patient_ids), len(feature_names)), dtype=np.float32)
    coverage: dict[str, float] = {}
    by_modality: dict[str, list[int]] = {}
    for j, (name, mod) in enumerate(zip(feature_names, feature_modality)):
        by_modality.setdefault(mod, []).append(j)

    for mod, cols in by_modality.items():
        if mod not in matrices:
            coverage[mod] = 0.0
            log.warning("TCGA lacks modality %r — those features stay at zero", mod)
            continue
        df = matrices[mod]
        df.index = df.index.astype(str)
        df = df.loc[[p for p in patient_ids if p in df.index]]
        lookup = {str(c).upper(): c for c in df.columns}
        hits = 0
        block = np.zeros((len(patient_ids), len(cols)), dtype=np.float32)
        pos = {p: i for i, p in enumerate(patient_ids)}
        rows = np.array([pos[p] for p in df.index])
        for k, j in enumerate(cols):
            src = lookup.get(_strip_suffix(feature_names[j]).upper())
            if src is None:
                continue
            hits += 1
            block[rows, k] = np.nan_to_num(df[src].to_numpy(dtype=np.float32))
        if mod == "expression" and log1p_expression and block.min() >= 0:
            block = np.log1p(block)
        out[:, cols] = block
        coverage[mod] = hits / max(len(cols), 1)
        log.info("TCGA %s: %.1f%% of %d training features matched", mod, 100 * coverage[mod], len(cols))

    return out, patient_ids, coverage


def load_cmax_table(path: str | Path, require_verified: bool = False) -> pd.DataFrame:
    """Read ``data/clinical_cmax.csv``.

    Rows with an empty ``cmax_um`` are dropped and reported: a drug with no
    clinical peak concentration cannot enter the clinical likelihood (guide §6.2).
    Values marked ``verified=false`` are literature look-ups that a human has not
    yet checked against the cited source; with ``require_verified=True`` they are
    excluded too.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"clinical Cmax table not found: {p}. It ships with the repository; "
            "fill in the empty rows from the literature before using Stage 1."
        )
    df = pd.read_csv(p, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("drug_id", "drug_name", "cmax_um"):
        if col not in df.columns:
            raise KeyError(f"{p.name}: missing required column {col!r}")
    if "verified" not in df.columns:
        df["verified"] = False
    df["verified"] = df["verified"].astype(str).str.lower().isin({"true", "1", "yes"})
    missing = df[df["cmax_um"].isna()]
    if len(missing):
        log.warning(
            "no Cmax for %d drug(s): %s — excluded from the clinical loss",
            len(missing), ", ".join(map(str, missing["drug_name"].tolist())),
        )
    df = df[df["cmax_um"].notna()].copy()
    unverified = df[~df["verified"]]
    if len(unverified):
        log.warning(
            "%d Cmax value(s) are UNVERIFIED literature look-ups: %s. "
            "Check them against the cited source before publishing.",
            len(unverified), ", ".join(map(str, unverified["drug_name"].tolist())),
        )
    if require_verified:
        df = df[df["verified"]]
    df["log_cmax"] = np.log(df["cmax_um"].to_numpy(dtype=float))
    return df


def map_agents_to_drug_ids(
    agents: Sequence[str], compound_annotation: pd.DataFrame
) -> dict[str, int]:
    """Match free-text therapeutic agent names onto GDSC drug ids."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    lookup: dict[str, int] = {}
    for _, row in compound_annotation.iterrows():
        did = int(row["drug_id"])
        names = [row.get("drug_name", "")]
        syn = row.get("synonyms", "")
        if isinstance(syn, str):
            names.extend(syn.split(","))
        for n in names:
            key = norm(n)
            if key:
                lookup.setdefault(key, did)
    out: dict[str, int] = {}
    for a in dict.fromkeys(agents):
        key = norm(a)
        if key in lookup:
            out[str(a)] = lookup[key]
    log.info("matched %d / %d clinical agents to GDSC drug ids", len(out), len(set(agents)))
    return out


def build_clinical_table(
    treatments: pd.DataFrame,
    compound_annotation: pd.DataFrame,
    cmax: pd.DataFrame,
    drug_ids: Sequence[int],
) -> pd.DataFrame:
    """One row per (patient, drug) with a binary response label and log Cmax."""
    df = treatments.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("patient_id", "therapeutic_agents", "treatment_outcome"):
        if col not in df.columns:
            raise KeyError(f"treatment table missing {col!r}; got {list(df.columns)}")
    df["outcome_norm"] = df["treatment_outcome"].astype(str).str.strip().str.lower()
    df["responder"] = df["outcome_norm"].map(RESPONDER_LABELS)
    n_before = len(df)
    df = df[df["responder"].notna()].copy()
    log.info("clinical outcomes: %d / %d treatment records have an evaluable RECIST-like outcome",
             len(df), n_before)

    df = df.assign(agent=df["therapeutic_agents"].astype(str).str.split(r"[,;+]")).explode("agent")
    df["agent"] = df["agent"].str.strip()
    mapping = map_agents_to_drug_ids(df["agent"].tolist(), compound_annotation)
    df["drug_id"] = df["agent"].map(mapping)
    df = df[df["drug_id"].notna()].copy()
    df["drug_id"] = df["drug_id"].astype(int)

    cmax_map = dict(zip(cmax["drug_id"].astype(int), cmax["log_cmax"]))
    df["log_cmax"] = df["drug_id"].map(cmax_map)
    dropped = df["log_cmax"].isna().sum()
    if dropped:
        log.warning("dropped %d (patient, drug) rows with no Cmax value", int(dropped))
    df = df[df["log_cmax"].notna()]

    index_of_drug = {int(d): i for i, d in enumerate(drug_ids)}
    df["drug_index"] = df["drug_id"].map(index_of_drug)
    unknown = df["drug_index"].isna().sum()
    if unknown:
        log.warning("dropped %d rows whose drug is absent from the GDSC training set", int(unknown))
    df = df[df["drug_index"].notna()].copy()
    df["drug_index"] = df["drug_index"].astype(int)

    out = (
        df[["patient_id", "drug_id", "drug_index", "responder", "log_cmax"]]
        .drop_duplicates(["patient_id", "drug_id"])
        .reset_index(drop=True)
    )
    out["responder"] = out["responder"].astype(int)
    log.info(
        "clinical table: %d (patient, drug) rows, %d patients, %d drugs, %.1f%% responders",
        len(out), out["patient_id"].nunique(), out["drug_id"].nunique(),
        100 * out["responder"].mean() if len(out) else float("nan"),
    )
    return out


def histology_availability(histology_dir: str | Path | None, patient_ids: Sequence[str]) -> np.ndarray:
    """Boolean per patient: is a pre-extracted UNI feature file present?"""
    if histology_dir is None:
        return np.zeros(len(patient_ids), dtype=bool)
    d = Path(histology_dir)
    if not d.exists():
        log.warning("histology feature directory not found: %s", d)
        return np.zeros(len(patient_ids), dtype=bool)
    have = {p.stem for p in list(d.glob("*.npy")) + list(d.glob("*.pt"))}
    avail = np.array([pid in have for pid in patient_ids], dtype=bool)
    log.info("histology available for %d / %d patients", int(avail.sum()), len(patient_ids))
    return avail
