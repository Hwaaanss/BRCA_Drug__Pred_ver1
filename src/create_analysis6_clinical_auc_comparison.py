#!/usr/bin/env python3
"""Create the clinical AUC comparison JSON used by v8 nature figures.

The underlying clinical validation is produced by ``w1w2w3_resolution.py`` as
``results/strengthening/w1_clinical_validation.json``.  The v8 nature figures
expect a compact, plotting-oriented file with historical method names:

  results/strengthening/analysis6_clinical_auc_comparison.json

This adapter keeps the original W1 schema untouched and writes only the alias
file consumed by ``src.figures_v8_nature.fig3`` and ``fig7``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
RESULTS = PROJECT_ROOT / "results"
W1_PATH = RESULTS / "strengthening" / "w1_clinical_validation.json"
FAIR_PATH = RESULTS / "reinforce" / "fair_embedding_and_bootstrap.json"
OUT_PATH = RESULTS / "strengthening" / "analysis6_clinical_auc_comparison.json"

CLINICAL_DRUGS = ["Docetaxel", "Paclitaxel", "Cyclophosphamide"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def ci_to_std(ci: list[float] | None) -> float:
    if not ci or len(ci) != 2:
        return 0.0
    low, high = ci
    if low is None or high is None:
        return 0.0
    # Approximate SD from a two-sided 95% interval when only summary stats exist.
    return max(0.0, float(high) - float(low)) / 3.92


def w1_entry(method: dict[str, Any] | None) -> dict[str, float | None]:
    if not method:
        return {"auc_mean": None, "auc_std": 0.0}
    auc = method.get("auc_mean", method.get("auc"))
    std = method.get("auc_std")
    if std is None:
        std = ci_to_std(method.get("bootstrap_ci_95"))
    return {
        "auc_mean": float(auc) if auc is not None and not math.isnan(float(auc)) else None,
        "auc_std": float(std) if std is not None else 0.0,
    }


def fair_entry(method: dict[str, Any] | None) -> dict[str, float | None]:
    if not method:
        return {"auc_mean": None, "auc_std": 0.0}
    auc = method.get("auc_mean", method.get("auc"))
    std = method.get("auc_std")
    if std is None:
        std = ci_to_std(method.get("bootstrap_ci95") or method.get("bootstrap_ci_95"))
    return {
        "auc_mean": float(auc) if auc is not None and not math.isnan(float(auc)) else None,
        "auc_std": float(std) if std is not None else 0.0,
    }


def main() -> int:
    if not W1_PATH.exists():
        raise FileNotFoundError(f"Run src/w1w2w3_resolution.py first: {W1_PATH}")

    w1 = load_json(W1_PATH)
    fair = load_json(FAIR_PATH) if FAIR_PATH.exists() else {"drugs": {}}
    out: dict[str, Any] = {}

    for drug in CLINICAL_DRUGS:
        if drug not in w1.get("drugs", {}):
            raise KeyError(f"{W1_PATH} is missing clinical drug: {drug}")
        w1_drug = w1["drugs"][drug]
        w1_methods = w1_drug.get("methods", {})
        fair_methods = fair.get("drugs", {}).get(drug, {}).get("methods", {})

        out[drug] = {
            "n": w1_drug.get("n"),
            "n_pos": w1_drug.get("n_pos"),
            "n_neg": w1_drug.get("n_neg"),
            "PathOmicDRP_4modal": w1_entry(w1_methods.get("PathOmicDRP_4modal")),
            "PathOmicDRP_3modal": w1_entry(w1_methods.get("PathOmicDRP_3modal_nohisto")),
            "ElasticNet_IC50_13d": w1_entry(w1_methods.get("ElasticNet_IC50")),
            "Raw_omics_2657d": w1_entry(w1_methods.get("Raw_omics")),
            "Raw_omics+histo_3681d": fair_entry(fair_methods.get("Raw_4modal")),
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
