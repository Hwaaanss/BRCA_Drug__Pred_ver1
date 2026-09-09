"""G-4 — how much of the label variance is just the drug main effect?

If ``f = Var(drug main effect) / Var(total)`` is large, a drug-pooled ("global")
correlation mostly measures which drug is which, not whether the model
understands a sample.  That is why §7.1 forbids reporting global PCC and
requires per-drug metrics normalised against a naive mean-effects predictor.

    python -m hill.audit.variance --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from hill.audit.base import audit_dir, is_synthetic, load_tables, save_json
from hill.config import load_config
from hill.utils.logging import get_logger
from hill.utils.stats import variance_decomposition

log = get_logger("audit.g4")


def run(cfg: Any) -> dict[str, Any]:
    _, pairs, _ = load_tables(cfg)
    out: dict[str, Any] = {"gate": "G-4", "synthetic": is_synthetic(cfg)}

    for label in ("ln_ic50_published", "auc_published"):
        if label not in pairs.columns:
            continue
        wide = pairs.pivot_table(index="cell_id", columns="drug_id", values=label, aggfunc="mean")
        decomp = variance_decomposition(wide.to_numpy(dtype=float))
        decomp["matrix_shape"] = list(wide.shape)
        decomp["fill_fraction"] = float(np.isfinite(wide.to_numpy(dtype=float)).mean())
        out[label] = decomp
        log.info(
            "G-4 %s: f_drug=%.3f f_sample=%.3f f_residual=%.3f",
            label, decomp["f_drug"], decomp["f_sample"], decomp["f_residual"],
        )

    primary = out.get("ln_ic50_published", {})
    out["f_drug"] = primary.get("f_drug", float("nan"))
    out["global_pcc_would_be_misleading"] = bool(primary.get("f_drug", 0) > 0.5)
    out["note"] = (
        "f_drug is the share of label variance explained by the drug main effect alone. "
        "The evaluation code refuses to compute drug-pooled correlations (see "
        "hill/evaluate.py::GlobalPCCForbiddenError)."
    )
    save_json(out, audit_dir(cfg) / "g4_variance.json")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G-4 drug main-effect variance share")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    res = run(cfg)
    print(f"f_drug = {res['f_drug']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
