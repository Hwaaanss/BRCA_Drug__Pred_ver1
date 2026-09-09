"""Optuna hyper-parameter search.

Ten TPE trials on a single validation fold of the primary split (LCO), scored by
the project's primary metric (dPCC vs NaiveMeanEffects).  Trials are pruned on
the median rule once past a warm-up.

Every trial's search space, sampled values and score are written to the study
database and to ``results/hpo/trials.csv``, so what was searched — and over what
range — is on the record (project rule R4).

    python -m hill.hpo --config configs/base.yaml --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hill.config import Config, load_config
from hill.data.prepare import load_pair_data
from hill.data.splits import load_or_build_splits
from hill.train import PrunedTrial, run_fold
from hill.utils.logging import get_logger
from hill.utils.provenance import capture_provenance
from hill.utils.resources import configure_runtime
from hill.utils.seed import seed_everything

log = get_logger("hpo")

# The search space is declared here (not scattered through the code) so the
# report can state exactly what was explored.
SEARCH_SPACE: dict[str, Any] = {
    "train.lr": {"type": "loguniform", "low": 1e-4, "high": 1.5e-3},
    "train.weight_decay": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
    "train.warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.15},
    "train.batch_size": {"type": "categorical", "choices": [128, 256, 512]},
    "model.d_model": {"type": "categorical", "choices": [128, 256, 384]},
    "model.n_layers": {"type": "int", "low": 2, "high": 6},
    "model.n_heads": {"type": "categorical", "choices": [4, 8]},
    "model.ffn_mult": {"type": "categorical", "choices": [2, 4]},
    "model.dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
    "model.head_dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
}


def suggest(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in space.items():
        if spec["type"] == "loguniform":
            out[key] = trial.suggest_float(key, spec["low"], spec["high"], log=True)
        elif spec["type"] == "uniform":
            out[key] = trial.suggest_float(key, spec["low"], spec["high"])
        elif spec["type"] == "int":
            out[key] = trial.suggest_int(key, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            out[key] = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"unknown search space type {spec['type']!r} for {key}")
    return out


def run_search(cfg: Config, n_trials: int | None = None, space: dict[str, Any] | None = None) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = space or SEARCH_SPACE
    n_trials = n_trials or cfg.hpo.n_trials

    data = load_pair_data(cfg)
    splits = load_or_build_splits(
        data.pairs, cfg.splits.primary, cfg.paths.splits_dir,
        n_folds=cfg.splits.n_folds, seed=cfg.splits.seed, val_fraction=cfg.splits.val_fraction,
    )
    refit_path = Path(cfg.paths.results_dir) / "gate0" / "g3_refit_per_pair.parquet"
    refit = pd.read_parquet(refit_path) if refit_path.exists() else None

    out_dir = Path(cfg.hpo.storage).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: Any) -> float:
        overrides = suggest(trial, space)
        overrides["train.epochs"] = cfg.hpo.epochs
        overrides["train.patience"] = cfg.hpo.patience
        if overrides["model.d_model"] % overrides["model.n_heads"] != 0:
            raise optuna.TrialPruned("d_model not divisible by n_heads")
        trial_cfg = cfg.copy_with(overrides)
        seed_everything(cfg.hpo.seed + trial.number)

        def callback(epoch: int, value: float) -> None:
            trial.report(value, epoch)
            if epoch >= cfg.hpo.pruner_warmup_epochs and trial.should_prune():
                raise PrunedTrial(f"trial {trial.number} pruned at epoch {epoch}")

        try:
            outcome = run_fold(
                trial_cfg, data, splits, fold=cfg.hpo.fold, seed=cfg.hpo.seed,
                out_dir=out_dir / f"trial{trial.number}", tag=f"hpo_trial{trial.number}",
                epoch_callback=callback, refit=refit, save_predictions=False,
                save_checkpoint=False,
            )
        except PrunedTrial as exc:
            log.info("%s", exc)
            raise optuna.TrialPruned() from exc
        except FloatingPointError as exc:
            log.warning("trial %d diverged: %s", trial.number, exc)
            return -1e6
        trial.set_user_attr("val_metrics", {k: float(v) for k, v in outcome.val_metrics.items()})
        trial.set_user_attr("test_metrics", {k: float(v) for k, v in outcome.test_metrics.items()})
        trial.set_user_attr("seconds", outcome.seconds)
        return float(outcome.best_metric)

    study = optuna.create_study(
        study_name=cfg.hpo.study_name,
        storage=f"sqlite:///{cfg.hpo.storage}",
        direction=cfg.hpo.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.hpo.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=cfg.hpo.pruner_warmup_epochs),
    )
    study.optimize(objective, n_trials=n_trials, timeout=(cfg.hpo.timeout_hours or 0) * 3600 or None)

    trials = study.trials_dataframe()
    trials.to_csv(out_dir / "trials.csv", index=False)

    best = study.best_trial
    best_params = dict(best.params)
    best_params["train.epochs"] = cfg.train.epochs  # search used a shortened schedule
    result = {
        "study_name": cfg.hpo.study_name,
        "n_trials_requested": n_trials,
        "n_trials_completed": int(sum(t.state.name == "COMPLETE" for t in study.trials)),
        "n_trials_pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)),
        "metric": cfg.hpo.metric,
        "best_value": float(best.value),
        "best_trial_number": best.number,
        "best_params": best_params,
        "search_space": space,
        "search_fold": cfg.hpo.fold,
        "search_epochs": cfg.hpo.epochs,
    }
    (out_dir / "best_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # A layered YAML that can be passed straight to any command as an extra config.
    nested: dict[str, Any] = {}
    for key, value in best_params.items():
        node = nested
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    best_yaml = out_dir / "best_params.yaml"
    best_yaml.write_text(
        "# Written by `python -m hill.hpo`. Layer it on top of the base config:\n"
        f"#   python -m hill.ablation --config configs/base.yaml --extra {best_yaml}\n"
        + yaml.safe_dump(nested, sort_keys=True),
        encoding="utf-8",
    )
    result_path_note = str(best_yaml)
    result["best_params_yaml"] = result_path_note
    log.info("best trial %d: %s = %.4f -> %s", best.number, cfg.hpo.metric, best.value, result_path_note)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optuna hyper-parameter search")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    configure_runtime(cfg)
    capture_provenance(Path(cfg.paths.results_dir) / "hpo_provenance.json", cfg)
    result = run_search(cfg, args.n_trials)
    print(json.dumps({k: result[k] for k in
                      ("best_value", "best_trial_number", "n_trials_completed", "n_trials_pruned")}, indent=2))
    print("best params ->", json.dumps(result["best_params"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
