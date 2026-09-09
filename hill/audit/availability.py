"""G-5 — what is actually available on the TCGA side?

Counts patients with omics, with pre-extracted slide features, with both, and
with an evaluable treatment outcome; and reports which drugs have a clinical
Cmax (needed by the Stage-1 likelihood) and which do not.

    python -m hill.audit.availability --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from hill.audit.base import audit_dir, is_synthetic, save_json
from hill.config import load_config
from hill.data.tcga import RESPONDER_LABELS, load_cmax_table
from hill.utils.logging import get_logger

log = get_logger("audit.g5")


def run(cfg: Any) -> dict[str, Any]:
    tcga_dir = Path(cfg.data.tcga_dir)
    out: dict[str, Any] = {"gate": "G-5", "synthetic": is_synthetic(cfg), "tcga_dir": str(tcga_dir)}

    omics_patients: set[str] = set()
    for name, fname in (("expression", "expression_tpm.parquet"), ("mutation", "mutations_binary.parquet")):
        path = tcga_dir / fname
        if path.exists():
            df = pd.read_parquet(path)
            ids = set(df.index.astype(str))
            out[f"n_patients_{name}"] = len(ids)
            omics_patients = ids if not omics_patients else (omics_patients & ids)
        else:
            out[f"n_patients_{name}"] = 0
            log.warning("missing TCGA %s matrix: %s", name, path)
    out["n_patients_all_omics"] = len(omics_patients)

    histo_dir = Path(cfg.data.histology_dir)
    slide_ids = set()
    if histo_dir.exists():
        slide_ids = {p.stem for p in list(histo_dir.glob("*.npy")) + list(histo_dir.glob("*.pt"))}
    out["n_patients_with_slide_features"] = len(slide_ids)
    out["n_patients_omics_and_slide"] = len(omics_patients & slide_ids)

    treat_path = tcga_dir / "treatments.csv"
    if treat_path.exists():
        tr = pd.read_csv(treat_path)
        tr.columns = [c.strip().lower() for c in tr.columns]
        outcome = tr.get("treatment_outcome", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
        evaluable = outcome.isin(RESPONDER_LABELS)
        out["n_treatment_records"] = int(len(tr))
        out["n_evaluable_outcomes"] = int(evaluable.sum())
        out["n_patients_with_evaluable_outcome"] = int(tr.loc[evaluable, "patient_id"].nunique()) if evaluable.any() else 0
        out["outcome_value_counts"] = outcome.value_counts().head(20).to_dict()
        agents = (
            tr.loc[evaluable, "therapeutic_agents"].astype(str).str.split(r"[,;+]").explode().str.strip()
        )
        out["agent_counts"] = agents.value_counts().head(30).to_dict()
        out["n_patients_omics_slide_outcome"] = int(
            len(omics_patients & slide_ids & set(tr.loc[evaluable, "patient_id"].astype(str)))
        )
    else:
        log.warning("missing TCGA treatment table: %s", treat_path)
        out["n_treatment_records"] = 0
        out["n_evaluable_outcomes"] = 0

    try:
        cmax = load_cmax_table(cfg.data.clinical_cmax_csv, require_verified=False)
        raw = pd.read_csv(cfg.data.clinical_cmax_csv, comment="#")
        raw.columns = [c.strip().lower() for c in raw.columns]
        out["cmax"] = {
            "n_drugs_listed": int(len(raw)),
            "n_with_value": int(len(cmax)),
            "n_verified": int(cmax["verified"].sum()),
            "drugs_without_cmax": raw.loc[raw["cmax_um"].isna(), "drug_name"].astype(str).tolist(),
            "unverified_drugs": cmax.loc[~cmax["verified"], "drug_name"].astype(str).tolist(),
        }
    except FileNotFoundError as exc:
        out["cmax"] = {"error": str(exc)}

    save_json(out, audit_dir(cfg) / "g5_availability.json")
    log.info(
        "G-5: %d patients with omics, %d with slides, %d with both, %d evaluable outcomes",
        out.get("n_patients_all_omics", 0), out["n_patients_with_slide_features"],
        out["n_patients_omics_and_slide"], out.get("n_evaluable_outcomes", 0),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-5 TCGA availability audit")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
