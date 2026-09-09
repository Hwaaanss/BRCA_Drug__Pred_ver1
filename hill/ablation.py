"""The ablation ladder (guide §7.4) plus the baseline suite, over N seeds.

    step 0  ScalarHILL on GDSC LCO          is the encoder sound?      vs MOLI
    step 1  HILL curve head (single sigma)  the curve head's value     vs step 0
    step 2  + heteroscedastic sigma_j       the likelihood's shape     vs step 1
    step 3  + TCGA transfer (gamma = 0)     domain transfer            vs step 2
    step 4  + gamma_E released              LRT on efficacy            vs step 3
    step 5  + gamma_m released              does shape touch potency?  vs step 4

Stop rule: if step 0 does not beat MOLI, the ladder halts.  A curve head bolted
onto a broken encoder would make the cause impossible to isolate.

Seeds: ``ablation.seeds`` runs (default 10).  With ``fold_cycling`` (default)
seed *s* is evaluated on fold ``s % n_folds``, so 10 seeds sweep all five LCO
folds twice at a fifth of the cost of a full seed x fold grid; set
``ablation.full_cv=true`` for the complete grid.

    python -m hill.ablation --config configs/base.yaml --extra results/hpo/best_params.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hill.config import Config, load_config
from hill.data.datasets import PairData, fit_fold_scaler
from hill.data.prepare import load_pair_data
from hill.data.splits import Splits, load_or_build_splits
from hill.evaluate import per_drug_correlation
from hill.models.baselines import NaiveMeanEffects, build_baseline
from hill.train import run_fold
from hill.utils.logging import get_logger
from hill.utils.provenance import capture_provenance
from hill.utils.resources import configure_runtime
from hill.utils.seed import seed_everything
from hill.utils.stats import bootstrap_ci, paired_test

log = get_logger("ablation")

STEP_CONFIGS: dict[int, dict[str, Any]] = {
    0: {"model.head": "scalar"},
    1: {"model.head": "curve", "loss.heteroscedastic": False},
    2: {"model.head": "curve", "loss.heteroscedastic": True},
}
STEP_LABELS = {
    0: "ScalarHILL (encoder sanity)",
    1: "HILL curve head, single sigma",
    2: "HILL curve head, heteroscedastic sigma_j",
    3: "HILL + TCGA transfer (gamma = 0)",
    4: "HILL + TCGA + gamma_E released",
    5: "HILL + TCGA + gamma_E + gamma_m released",
}


def evaluate_baseline_predictions(
    name: str,
    pred: np.ndarray,
    data: PairData,
    test_idx: np.ndarray,
    naive_pred: np.ndarray,
    min_pairs_per_drug: int,
    bootstrap_n: int,
) -> dict[str, float]:
    """Score a scalar baseline with exactly the metrics used for the models."""
    obs = data.pairs["ln_ic50_published"].to_numpy(dtype=float)[test_idx]
    drug = data.drug_index[test_idx]
    per_drug = per_drug_correlation(pred, obs, drug, min_pairs_per_drug).rename(columns={"corr": "pcc"})
    naive_per_drug = per_drug_correlation(naive_pred, obs, drug, min_pairs_per_drug).rename(
        columns={"corr": "pcc_naive"}
    )
    merged = per_drug.merge(naive_per_drug[["drug_index", "pcc_naive"]], on="drug_index", how="left")
    delta = merged["pcc"] - merged["pcc_naive"]
    point, lo, hi = bootstrap_ci(delta.to_numpy(), n_boot=bootstrap_n)
    return {
        "per_drug_pcc": float(np.nanmean(merged["pcc"])),
        "delta_pcc_vs_naive": float(np.nanmean(delta)),
        "delta_pcc_ci_low": lo,
        "delta_pcc_ci_high": hi,
        "n_drugs_evaluated": float(delta.notna().sum()),
        "viability_rmse": float("nan"),
        "n_pairs": float(test_idx.size),
        "method": name,
    }


def run_baselines(
    cfg: Config, data: PairData, splits: Splits, fold: int, seed: int
) -> list[dict[str, Any]]:
    """Fit every configured baseline on the training fold and score the test fold."""
    train_idx, test_idx = splits.train_idx(fold), splits.test_idx(fold)
    targets = data.pairs["ln_ic50_published"].to_numpy(dtype=np.float32)
    scaler = fit_fold_scaler(data, train_idx)
    omics = scaler.transform(data.omics)

    naive = NaiveMeanEffects().fit(omics, data.cell_index, data.drug_index, targets, train_idx)
    naive_pred = naive.predict(omics, data.cell_index, data.drug_index, test_idx)

    rows: list[dict[str, Any]] = []
    for name in cfg.ablation.baselines:
        t0 = time.time()
        seed_everything(seed)
        model = build_baseline(name, cfg, data.meta["feature_modality"], seed=seed)
        model.fit(omics, data.cell_index, data.drug_index, targets, train_idx)
        pred = (
            naive_pred if name == "naive"
            else model.predict(omics, data.cell_index, data.drug_index, test_idx)
        )
        metrics = evaluate_baseline_predictions(
            name, pred, data, test_idx, naive_pred, cfg.eval.min_pairs_per_drug, cfg.eval.bootstrap_n
        )
        metrics.update({"seconds": time.time() - t0, "fold": fold, "seed": seed,
                        "step": -1, "label": f"baseline: {name}"})
        rows.append(metrics)
        log.info("baseline %-12s fold %d seed %d: dPCC %.4f", name, fold, seed,
                 metrics["delta_pcc_vs_naive"])
    return rows


def run_ladder_step(
    cfg: Config, data: PairData, splits: Splits, step: int, fold: int, seed: int,
    refit: pd.DataFrame | None, out_dir: Path,
) -> dict[str, Any]:
    """Train and score one GDSC-side ladder step (0-2)."""
    step_cfg = cfg.copy_with(STEP_CONFIGS[step])
    tag = f"step{step}"
    outcome = run_fold(
        step_cfg, data, splits, fold, seed, out_dir=out_dir / tag, tag=f"{tag}_f{fold}_s{seed}",
        refit=refit,
    )
    row = {
        "step": step,
        "label": STEP_LABELS[step],
        "method": tag,
        "fold": fold,
        "seed": seed,
        "checkpoint": outcome.checkpoint,
        "seconds": outcome.seconds,
        "best_epoch": outcome.best_epoch,
        "n_parameters": outcome.n_parameters,
    }
    row.update({k: v for k, v in outcome.test_metrics.items()})
    row.update({f"val_{k}": v for k, v in outcome.val_metrics.items()})
    return row


def run_transfer_steps(
    cfg: Config, data: PairData, checkpoint: str, seed: int, steps: list[int]
) -> list[dict[str, Any]]:
    """Ladder steps 3-5: TCGA transfer and the two histology gates."""
    from hill.stage1 import load_tcga_bundle, run_stage1

    try:
        bundle = load_tcga_bundle(cfg, data)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        log.warning("skipping ladder steps 3-5: %s", exc)
        return [
            {"step": s, "label": STEP_LABELS[s], "method": f"step{s}", "seed": seed,
             "skipped_reason": str(exc)}
            for s in steps
        ]

    gate_map = {3: (False, False), 4: (True, False), 5: (True, True)}
    rows = []
    for step in steps:
        res = run_stage1(cfg, data, bundle, checkpoint, gate_map[step], seed=seed,
                         n_folds=min(5, cfg.splits.n_folds), tag=f"step{step}_s{seed}")
        rows.append(
            {
                "step": step,
                "label": STEP_LABELS[step],
                "method": f"step{step}",
                "seed": seed,
                "fold": -1,
                "clinical_roc_auc": res["clinical"].get("roc_auc", float("nan")),
                "clinical_average_precision": res["clinical"].get("average_precision", float("nan")),
                "clinical_brier": res["clinical"].get("brier", float("nan")),
                "clinical_loglik": res["loglik"],
                "seconds": res["seconds"],
                **res["gammas"],
            }
        )
    return rows


def stop_rule_triggered(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Step 0 must beat MOLI, otherwise the ladder is not interpretable."""
    df = pd.DataFrame(rows)
    if df.empty or "method" not in df:
        return False, {}
    step0 = df[df["method"] == "step0"].sort_values("seed")["delta_pcc_vs_naive"]
    moli = df[df["method"] == "moli"].sort_values("seed")["delta_pcc_vs_naive"]
    if step0.empty or moli.empty:
        return False, {"reason": "step 0 or MOLI missing; stop rule not evaluated"}
    n = min(len(step0), len(moli))
    test = paired_test(step0.to_numpy()[:n], moli.to_numpy()[:n])
    info = {
        "step0_mean": float(step0.mean()),
        "moli_mean": float(moli.mean()),
        "paired_test": test,
        "n_seeds": n,
    }
    return bool(step0.mean() <= moli.mean()), info


def run_ablation(cfg: Config, seeds: int | None = None, steps: list[int] | None = None) -> pd.DataFrame:
    data = load_pair_data(cfg)
    splits = load_or_build_splits(
        data.pairs, cfg.splits.primary, cfg.paths.splits_dir,
        n_folds=cfg.splits.n_folds, seed=cfg.splits.seed, val_fraction=cfg.splits.val_fraction,
    )
    refit_path = Path(cfg.paths.results_dir) / "gate0" / "g3_refit_per_pair.parquet"
    refit = pd.read_parquet(refit_path) if refit_path.exists() else None

    n_seeds = seeds or cfg.ablation.seeds
    steps = steps or list(cfg.ablation.steps)
    out_dir = Path(cfg.paths.checkpoint_dir) / "ablation"
    results_dir = Path(cfg.paths.results_dir)
    rows: list[dict[str, Any]] = []
    checkpoints: dict[int, str] = {}

    gdsc_steps = [s for s in steps if s in STEP_CONFIGS]
    transfer_steps = [s for s in steps if s in (3, 4, 5)]

    for seed_i in range(n_seeds):
        seed = cfg.seed + seed_i
        folds = range(cfg.splits.n_folds) if cfg.ablation.full_cv else [seed_i % cfg.splits.n_folds]
        for fold in folds:
            rows.extend(run_baselines(cfg, data, splits, fold, seed))
            for step in gdsc_steps:
                row = run_ladder_step(cfg, data, splits, step, fold, seed, refit, out_dir)
                rows.append(row)
                if step == max(gdsc_steps):
                    checkpoints[seed] = row["checkpoint"]
            pd.DataFrame(rows).to_csv(results_dir / "ablation.csv", index=False)

        triggered, info = stop_rule_triggered(rows)
        if triggered and cfg.ablation.enforce_stop_rule:
            log.error(
                "STOP RULE: step 0 (%.4f) does not beat MOLI (%.4f) over %d seed(s). "
                "Halting the ladder — fix the encoder or the data pipeline first.",
                info["step0_mean"], info["moli_mean"], info["n_seeds"],
            )
            (results_dir / "ablation_stop_rule.json").write_text(
                json.dumps({"triggered": True, **info}, indent=2, default=str), encoding="utf-8"
            )
            break

        if transfer_steps and seed in checkpoints:
            rows.extend(run_transfer_steps(cfg, data, checkpoints[seed], seed, transfer_steps))
            pd.DataFrame(rows).to_csv(results_dir / "ablation.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "ablation.csv", index=False)

    triggered, info = stop_rule_triggered(rows)
    (results_dir / "ablation_stop_rule.json").write_text(
        json.dumps({"triggered": bool(triggered), **info}, indent=2, default=str), encoding="utf-8"
    )
    summary = summarise_ablation(df)
    summary.to_csv(results_dir / "ablation_summary.csv", index=False)
    log.info("ablation complete: %d runs -> %s", len(df), results_dir / "ablation.csv")
    return df


def summarise_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """mean +/- sd over seeds for every method (never a single-seed number)."""
    if df.empty:
        return df
    metric_cols = [
        c for c in ("delta_pcc_vs_naive", "per_drug_pcc", "per_drug_pcc_uncensored",
                    "per_drug_pcc_emax", "viability_rmse", "clinical_roc_auc",
                    "clinical_average_precision", "gamma_e", "gamma_m")
        if c in df.columns
    ]
    grouped = df.groupby(["step", "method"], dropna=False)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    if "label" in df.columns:
        labels = df.groupby(["step", "method"], dropna=False)["label"].first().reset_index()
        out = out.merge(labels, on=["step", "method"], how="left")
    return out.sort_values(["step", "method"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ablation ladder over N seeds")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--extra", nargs="*", default=[], help="extra config files layered on top")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set, extra_files=args.extra)
    cfg.paths.ensure()
    configure_runtime(cfg)
    capture_provenance(Path(cfg.paths.results_dir) / "ablation_provenance.json", cfg)

    df = run_ablation(cfg, args.seeds, args.steps)
    summary = summarise_ablation(df)
    cols = [c for c in ("step", "method", "label", "delta_pcc_vs_naive_mean", "delta_pcc_vs_naive_std",
                        "delta_pcc_vs_naive_count") if c in summary.columns]
    print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
