"""G-2 — how much of the published label set is censored?

``LN_IC50 > ln(MAX_CONC)`` means the two-parameter fit placed the IC50 beyond
the highest concentration ever tested: an extrapolation, not a measurement.
This fraction is the size of the bias this project is trying to remove.

    python -m hill.audit.censoring --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from hill.audit.base import audit_dir, is_synthetic, load_tables, save_json
from hill.config import load_config
from hill.utils.logging import get_logger

log = get_logger("audit.g2")


def run(cfg: Any) -> dict[str, Any]:
    _, pairs, _ = load_tables(cfg)
    if "censored" not in pairs.columns:
        raise KeyError("pair table has no 'censored' column — rerun hill.data.prepare")

    have_fit = pairs["ln_ic50_published"].notna()
    censored = pairs["censored"].fillna(False).astype(bool)

    per_drug = (
        pairs.assign(censored=censored)
        .groupby("drug_id")
        .agg(n_pairs=("censored", "size"), n_censored=("censored", "sum"))
        .reset_index()
    )
    per_drug["censored_fraction"] = per_drug["n_censored"] / per_drug["n_pairs"]
    per_drug = per_drug.sort_values("censored_fraction", ascending=False)

    result = {
        "gate": "G-2",
        "synthetic": is_synthetic(cfg),
        "n_pairs": int(len(pairs)),
        "n_pairs_with_published_fit": int(have_fit.sum()),
        "censored_fraction_overall": float(censored[have_fit].mean()) if have_fit.any() else float("nan"),
        "n_censored": int(censored.sum()),
        "n_drugs": int(pairs["drug_id"].nunique()),
        "drugs_fully_censored": int((per_drug["censored_fraction"] == 1.0).sum()),
        "drugs_never_censored": int((per_drug["censored_fraction"] == 0.0).sum()),
        "per_drug": per_drug.to_dict(orient="records"),
        "note": (
            "A censoring fraction below 10% would weaken one of this project's arguments; "
            "the number is reported either way."
        ),
    }
    save_json(result, audit_dir(cfg) / "g2_censoring.json")
    per_drug.to_csv(audit_dir(cfg) / "g2_censoring_per_drug.csv", index=False)
    log.info(
        "G-2: %.1f%% of pairs are censored (%d / %d)",
        100 * result["censored_fraction_overall"], result["n_censored"], result["n_pairs"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-2 censoring audit")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--gdsc-version", type=int, default=None, help="restrict to one GDSC release")
    args = ap.parse_args(argv)
    overrides = list(args.set)
    if args.gdsc_version:
        overrides.append(f"data.gdsc_versions=[{args.gdsc_version}]")
    cfg = load_config(args.config, overrides=overrides)
    res = run(cfg)
    print(f"censored fraction = {100 * res['censored_fraction_overall']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
