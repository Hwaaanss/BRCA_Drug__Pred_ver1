"""Build every processed artefact the model needs, once.

    python -m hill.data.prepare --config configs/base.yaml

Outputs (under ``paths.processed_dir``)
    gdsc_points.parquet   one row per (pair, concentration): the training signal
    gdsc_pairs.parquet    one row per (cell, drug), with published IC50/AUC and
                          the censoring flag — for evaluation only
    padded.npz            dense (P, K) log-concentration / viability / mask
    cell_omics.npz        (C, F) aligned omics matrix
    drug_features.npz     (D, F_drug) drug descriptors
    token_spec.npz        the tokenisation plan
    prepare_report.json   counts and provenance for reports/gate0.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hill.config import Config, load_config
from hill.data.datasets import PairData
from hill.data.drugs import DrugFeatures, build_drug_features, load_compound_annotation
from hill.data.gdsc import attach_published, build_points_table, load_fitted, points_to_padded
from hill.data.omics import assemble_cell_omics, load_omics, save_omics
from hill.data.splits import build_all_splits
from hill.data.tokenize import TokenSpec, build_token_spec, read_gmt
from hill.utils.logging import get_logger
from hill.utils.provenance import capture_provenance

log = get_logger("data.prepare")


def _raw_files(cfg: Config) -> dict[str, Path]:
    raw = Path(cfg.paths.raw_dir) / "gdsc_raw"
    out: dict[str, Path] = {}
    for v in cfg.data.gdsc_versions:
        matches = sorted(raw.glob(f"GDSC{v}_public_raw_data*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"no raw viability file for GDSC{v} under {raw}. "
                "Run `python -m hill.data.download --what gdsc-raw`."
            )
        out[f"GDSC{v}"] = matches[0]
    return out


def _fitted_files(cfg: Config) -> list[Path]:
    d = Path(cfg.paths.raw_dir) / "gdsc_fitted"
    return [p for v in cfg.data.gdsc_versions for p in sorted(d.glob(f"GDSC{v}_fitted_dose_response*"))]


def build_all(cfg: Config, force: bool = False) -> dict[str, Any]:
    """Run the full preparation pipeline and return a summary report."""
    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    points_path = out_dir / "gdsc_points.parquet"
    pairs_path = out_dir / "gdsc_pairs.parquet"
    if force or not (points_path.exists() and pairs_path.exists()):
        points, pairs, qc = build_points_table(_raw_files(cfg), cfg)
        fitted = load_fitted(_fitted_files(cfg))
        pairs = attach_published(pairs, fitted)
        points.to_parquet(points_path, index=False)
        pairs.to_parquet(pairs_path, index=False)
        report["ingest_qc"] = qc.to_dict()
    else:
        points = pd.read_parquet(points_path)
        pairs = pd.read_parquet(pairs_path)
        log.info("reusing cached point/pair tables (%d points, %d pairs)", len(points), len(pairs))

    omics_path = out_dir / "cell_omics.npz"
    if force or not omics_path.exists():
        omics = assemble_cell_omics(cfg)
        save_omics(omics, omics_path)
    else:
        omics = load_omics(omics_path)
        log.info("reusing cached omics matrix %s", omics.values.shape)

    # keep only pairs whose cell line has omics
    have = set(omics.cell_ids)
    before = len(pairs)
    pairs = pairs[pairs["cell_id"].astype(str).isin(have)].reset_index(drop=True)
    log.info("pairs with omics: %d / %d", len(pairs), before)
    if pairs.empty:
        raise ValueError(
            "no (cell, drug) pair survived the omics join — the COSMIC identifiers in the raw "
            "viability file and the omics matrix do not overlap. Check hill/data/omics.py adapters."
        )
    points = points[points["pair_id"].isin(set(pairs["pair_id"]))].reset_index(drop=True)

    drug_ids = sorted(pairs["drug_id"].unique().tolist())
    drug_path = out_dir / "drug_features"
    if force or not drug_path.with_suffix(".npz").exists():
        annotation = None
        ann_file = Path(cfg.paths.raw_dir) / "gdsc_annotation" / "screened_compounds.csv"
        if ann_file.exists():
            annotation = load_compound_annotation(ann_file)
        drug_feats = build_drug_features(drug_ids, cfg, annotation)
        drug_feats.save(drug_path)
    else:
        drug_feats = DrugFeatures.load(drug_path)
        if drug_feats.drug_ids != drug_ids:
            log.warning("cached drug features do not match the current drug set; rebuilding")
            drug_feats = build_drug_features(drug_ids, cfg)
            drug_feats.save(drug_path)

    spec_path = out_dir / "token_spec.npz"
    if force or not spec_path.exists():
        gene_sets = read_gmt(cfg.data.pathway_gmt)
        for extra in cfg.data.extra_gmt:
            gene_sets.update(read_gmt(extra))
        spec = build_token_spec(
            feature_names=omics.feature_names,
            feature_modality=omics.feature_modality,
            gene_sets=gene_sets,
            target_n_tokens=cfg.data.target_n_tokens,
            min_genes_per_group=cfg.data.min_genes_per_group,
            max_genes_per_group=cfg.data.max_genes_per_group,
            n_latent=cfg.data.n_latent_tokens,
            feature_variance=omics.variance(),
        )
        spec.save(spec_path)
    else:
        spec = TokenSpec.load(spec_path)

    max_points = int(min(cfg.data.max_points_per_pair, points.groupby("pair_id").size().max()))
    padded = points_to_padded(points, pairs, max_points)
    cell_pos = {c: i for i, c in enumerate(omics.cell_ids)}
    drug_pos = {d: i for i, d in enumerate(drug_feats.drug_ids)}
    padded["cell_index"] = np.array([cell_pos[str(c)] for c in pairs["cell_id"]], dtype=np.int64)
    padded["drug_index"] = np.array([drug_pos[int(d)] for d in pairs["drug_id"]], dtype=np.int64)
    np.savez_compressed(out_dir / "padded.npz", **padded)

    splits_dir = Path(cfg.paths.splits_dir)
    build_all_splits(
        pairs, cfg.splits.schemes, splits_dir,
        n_folds=cfg.splits.n_folds, seed=cfg.splits.seed, val_fraction=cfg.splits.val_fraction,
    )

    report.update(
        {
            "n_points": int(len(points)),
            "n_pairs": int(len(pairs)),
            "n_cells": int(pairs["cell_id"].nunique()),
            "n_drugs": int(pairs["drug_id"].nunique()),
            "max_points_per_pair": max_points,
            "omics_shape": list(omics.values.shape),
            "n_omics_tokens": int(spec.n_tokens),
            "n_group_tokens": int(spec.n_groups),
            "token_coverage_fraction": float(spec.meta.get("coverage_fraction", float("nan"))),
            "drug_feature_dim": int(drug_feats.dim),
            "drug_feature_method": drug_feats.method,
            "drug_structure_coverage": float(drug_feats.coverage),
            "censored_fraction": float(pairs["censored"].mean()) if "censored" in pairs else float("nan"),
            "points_per_pair_mean": float(points.groupby("pair_id").size().mean()),
        }
    )
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    capture_provenance(out_dir / "prepare_provenance.json", cfg, extra=report)
    log.info("prepare complete: %s", {k: report[k] for k in ("n_points", "n_pairs", "n_cells", "n_drugs")})
    return report


def load_pair_data(cfg: Config) -> PairData:
    """Load the processed artefacts into the in-memory bundle used for training."""
    out_dir = Path(cfg.paths.processed_dir)
    required = ["gdsc_pairs.parquet", "padded.npz", "cell_omics.npz", "drug_features.npz", "token_spec.npz"]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing processed files {missing} in {out_dir}. Run `python -m hill.data.prepare` first."
        )
    pairs = pd.read_parquet(out_dir / "gdsc_pairs.parquet")
    with np.load(out_dir / "padded.npz", allow_pickle=False) as z:
        padded = {k: z[k] for k in z.files}
    omics = load_omics(out_dir / "cell_omics.npz")
    drug_feats = DrugFeatures.load(out_dir / "drug_features")
    spec = TokenSpec.load(out_dir / "token_spec.npz")
    return PairData(
        pairs=pairs,
        log_conc=padded["log_conc"],
        viability=padded["viability"],
        point_mask=padded["mask"],
        omics=omics.values,
        cell_ids=omics.cell_ids,
        cell_index=padded["cell_index"],
        drug_features=drug_feats.values,
        drug_ids=drug_feats.drug_ids,
        drug_index=padded["drug_index"],
        token_spec=spec,
        meta={
            "drug_feature_method": drug_feats.method,
            "drug_structure_coverage": drug_feats.coverage,
            "feature_names": omics.feature_names,
            "feature_modality": omics.feature_modality,
        },
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare HILL training data")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.gdsc_versions=[1,2]")
    ap.add_argument("--force", action="store_true", help="rebuild cached artefacts")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    report = build_all(cfg, force=args.force)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
