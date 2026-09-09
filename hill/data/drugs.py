"""Drug featurisation.

LDO (leave-drugs-out) only works if a drug is described by *features*, so the
default representation is a Morgan fingerprint plus a handful of physicochemical
descriptors, computed once and cached to parquet.

SMILES are resolved from the GDSC compound annotation via PubChem and cached on
disk.  If RDKit is unavailable the code falls back to a deterministic hashed
character-n-gram fingerprint of the SMILES string — chemically weaker but still
a *feature* representation, so LDO stays valid; the fallback is logged loudly
and recorded in the feature table's metadata.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from hill.utils.logging import get_logger

log = get_logger("data.drugs")

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "CanonicalSMILES,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
)


@dataclass
class DrugFeatures:
    drug_ids: list[int]
    values: np.ndarray            # (D, F) float32
    feature_names: list[str]
    method: str                   # rdkit_morgan | hashed_ngram | onehot
    coverage: float               # fraction of drugs with a resolved structure

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.values, columns=self.feature_names)
        df.insert(0, "drug_id", self.drug_ids)
        df.attrs["method"] = self.method
        df.attrs["coverage"] = self.coverage
        return df

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p.with_suffix(".npz"),
            values=self.values,
            drug_ids=np.array(self.drug_ids, dtype=np.int64),
            feature_names=np.array(self.feature_names),
            json_blob=np.array(json.dumps({"method": self.method, "coverage": self.coverage})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DrugFeatures":
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as z:
            blob = json.loads(str(z["json_blob"]))
            return cls(
                drug_ids=[int(d) for d in z["drug_ids"]],
                values=z["values"],
                feature_names=[str(f) for f in z["feature_names"]],
                method=blob["method"],
                coverage=float(blob["coverage"]),
            )


def load_compound_annotation(path: str | Path) -> pd.DataFrame:
    """GDSC screened-compound table: DRUG_ID, DRUG_NAME, TARGET, TARGET_PATHWAY."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"compound annotation not found: {p}. "
            "Run `python -m hill.data.download --what gdsc-annotation`."
        )
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    df.columns = [str(c).strip().upper() for c in df.columns]
    rename = {"DRUG_ID": "drug_id", "DRUG_NAME": "drug_name", "SYNONYMS": "synonyms",
              "TARGET": "target", "TARGET_PATHWAY": "target_pathway"}
    keep = {k: v for k, v in rename.items() if k in df.columns}
    if "DRUG_ID" not in df.columns:
        raise KeyError(f"{p.name}: no DRUG_ID column; got {list(df.columns)[:10]}")
    out = df[list(keep)].rename(columns=keep).drop_duplicates("drug_id")
    out["drug_id"] = out["drug_id"].astype(int)
    return out


def fetch_smiles(
    names: Iterable[str], cache_path: str | Path, sleep: float = 0.25, timeout: int = 20
) -> dict[str, dict[str, Any]]:
    """Resolve compound names to SMILES + descriptors via PubChem, with a disk cache."""
    import requests

    cache_file = Path(cache_path)
    cache: dict[str, dict[str, Any]] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    missing = [n for n in dict.fromkeys(names) if n and n not in cache]
    for i, name in enumerate(missing):
        try:
            resp = requests.get(PUBCHEM_URL.format(name=requests.utils.quote(str(name))), timeout=timeout)
            if resp.status_code == 200:
                props = resp.json()["PropertyTable"]["Properties"][0]
                cache[name] = props
            else:
                cache[name] = {}
        except Exception as exc:  # noqa: BLE001
            log.warning("PubChem lookup failed for %s: %s", name, exc)
            cache[name] = {}
        if i % 25 == 0:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        time.sleep(sleep)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    resolved = sum(1 for n in names if cache.get(n, {}).get("CanonicalSMILES"))
    log.info("SMILES cache: %d entries, %d resolved", len(cache), resolved)
    return cache


def _morgan_matrix(smiles: Sequence[str | None], bits: int, radius: int) -> tuple[np.ndarray, str]:
    """Morgan fingerprints via RDKit, or a hashed n-gram fallback."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator

        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
        out = np.zeros((len(smiles), bits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = gen.GetFingerprintAsNumPy(mol)
            out[i] = fp.astype(np.float32)
        return out, "rdkit_morgan"
    except ImportError:
        log.warning(
            "RDKit not installed — falling back to hashed character n-gram fingerprints. "
            "Install rdkit for proper Morgan fingerprints (see environment.yml)."
        )
        out = np.zeros((len(smiles), bits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            if not smi:
                continue
            s = str(smi)
            for n in (2, 3, 4):
                for j in range(len(s) - n + 1):
                    out[i, hash(s[j : j + n]) % bits] = 1.0
        return out, "hashed_ngram"


def build_drug_features(
    drug_ids: Sequence[int],
    cfg: Any,
    annotation: pd.DataFrame | None = None,
) -> DrugFeatures:
    """Assemble the (D, F) drug feature matrix for the given drug ids."""
    drug_ids = [int(d) for d in drug_ids]
    if cfg.data.drug_features == "onehot":
        values = np.eye(len(drug_ids), dtype=np.float32)
        return DrugFeatures(
            drug_ids=drug_ids,
            values=values,
            feature_names=[f"drug_onehot_{d}" for d in drug_ids],
            method="onehot",
            coverage=1.0,
        )

    raw_dir = Path(cfg.paths.raw_dir)
    if annotation is None:
        annotation = load_compound_annotation(raw_dir / "gdsc_annotation" / "screened_compounds.csv")
    ann = annotation.set_index("drug_id")
    names = [str(ann["drug_name"].get(d, "")) if "drug_name" in ann.columns else "" for d in drug_ids]

    cache = fetch_smiles(names, raw_dir / "drugs" / "pubchem_cache.json")
    smiles = [cache.get(n, {}).get("CanonicalSMILES") for n in names]
    coverage = float(np.mean([bool(s) for s in smiles])) if smiles else 0.0
    if coverage < 0.5:
        log.warning(
            "only %.0f%% of drugs have a resolved structure — LDO results will be weak; "
            "consider supplying SMILES manually in data/raw/drugs/pubchem_cache.json",
            100 * coverage,
        )

    fp, method = _morgan_matrix(smiles, cfg.data.fingerprint_bits, cfg.data.fingerprint_radius)

    desc_keys = ["MolecularWeight", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount"]
    desc = np.array(
        [[float(cache.get(n, {}).get(k, np.nan) or np.nan) for k in desc_keys] for n in names],
        dtype=np.float32,
    )
    col_mean = np.nanmean(np.where(np.isfinite(desc), desc, np.nan), axis=0)
    col_mean = np.nan_to_num(col_mean)
    inds = np.where(np.isfinite(desc), desc, col_mean)
    std = inds.std(axis=0)
    std[std < 1e-8] = 1.0
    desc_z = ((inds - inds.mean(axis=0)) / std).astype(np.float32)

    # Target-pathway one-hot: cheap, biologically meaningful, and available for
    # every GDSC compound even when the structure lookup fails.
    pathway = pd.Series(
        [str(ann["target_pathway"].get(d, "unknown")) if "target_pathway" in ann.columns else "unknown"
         for d in drug_ids]
    ).fillna("unknown")
    pw_dummies = pd.get_dummies(pathway, prefix="pw").astype(np.float32).to_numpy()

    values = np.concatenate([fp, desc_z, pw_dummies], axis=1).astype(np.float32)
    feature_names = (
        [f"fp_{i}" for i in range(fp.shape[1])]
        + [f"desc_{k}" for k in desc_keys]
        + [f"pw_{i}" for i in range(pw_dummies.shape[1])]
    )
    log.info("drug features: %d drugs x %d dims (%s, coverage %.0f%%)",
             len(drug_ids), values.shape[1], method, 100 * coverage)
    return DrugFeatures(
        drug_ids=drug_ids,
        values=values,
        feature_names=feature_names,
        method=method,
        coverage=coverage,
    )
