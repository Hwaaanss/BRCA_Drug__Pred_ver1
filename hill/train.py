"""Training.

Stage 0   GDSC pre-training.  gamma frozen at 0 (cell lines have no slides).
          Loss = viability likelihood for HILL, ln IC50 regression for ScalarHILL.
Stage 1   TCGA fine-tuning.  gamma released, clinical BCE at Cmax, optionally
          keeping the source likelihood with weight ``loss.source_weight_stage1``.

Early stopping uses the *validation primary metric* (dPCC vs NaiveMeanEffects),
never the training loss.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from hill.config import Config, load_config
from hill.data.datasets import PairData, make_loaders
from hill.data.prepare import load_pair_data
from hill.data.splits import Splits, load_or_build_splits
from hill.evaluate import evaluate_predictions, predict_pairs, save_eval
from hill.losses import ScalarRegressionLoss, ViabilityLikelihood
from hill.models.baselines import NaiveMeanEffects
from hill.models.hill import build_model
from hill.utils.logging import JsonlLogger, get_logger
from hill.utils.provenance import capture_provenance
from hill.utils.resources import amp_dtype, configure_runtime, resolve_device
from hill.utils.seed import seed_everything

log = get_logger("train")


class PrunedTrial(RuntimeError):
    """Raised by an HPO pruning callback to abort a hopeless trial."""


@dataclass
class TrainOutcome:
    best_metric: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    gammas: dict[str, float] = field(default_factory=dict)
    checkpoint: str | None = None
    n_parameters: int = 0
    seconds: float = 0.0
    stopped_early: bool = False

    def as_row(self) -> dict[str, Any]:
        row = {f"val_{k}": v for k, v in self.val_metrics.items()}
        row.update({f"test_{k}": v for k, v in self.test_metrics.items()})
        row.update(self.gammas)
        row.update(
            {
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "n_parameters": self.n_parameters,
                "seconds": self.seconds,
                "stopped_early": self.stopped_early,
            }
        )
        return row


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _scalar_targets(data: PairData) -> np.ndarray:
    if "ln_ic50_published" not in data.pairs.columns:
        raise KeyError("pair table lacks ln_ic50_published — ScalarHILL has no target to regress")
    return data.pairs["ln_ic50_published"].to_numpy(dtype=np.float32)


def _load_refit(cfg: Config) -> pd.DataFrame | None:
    path = Path(cfg.paths.results_dir) / "gate0" / "g3_refit_per_pair.parquet"
    if path.exists():
        return pd.read_parquet(path)
    log.warning("no G-3 refit found at %s — Emax correlation will not be reported", path)
    return None


def run_fold(
    cfg: Config,
    data: PairData,
    splits: Splits,
    fold: int,
    seed: int,
    out_dir: str | Path,
    tag: str = "run",
    epoch_callback: Callable[[int, float], None] | None = None,
    refit: pd.DataFrame | None = None,
    save_predictions: bool = True,
    save_checkpoint: bool = True,
) -> TrainOutcome:
    """Train one model on one fold and evaluate it."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg)
    dtype = amp_dtype(cfg)

    train_idx, val_idx, test_idx = splits.train_idx(fold), splits.val_idx(fold), splits.test_idx(fold)
    if min(train_idx.size, val_idx.size, test_idx.size) == 0:
        raise ValueError(f"fold {fold} has an empty partition: "
                         f"train={train_idx.size} val={val_idx.size} test={test_idx.size}")

    targets = _scalar_targets(data)
    train_loader, val_loader, test_loader, scaler = make_loaders(
        data, train_idx, val_idx, test_idx, cfg, target_ln_ic50=targets, seed=seed
    )

    if splits.scheme == "LDO":
        # A held-out drug has no embedding row and no learned sigma, so LDO is only
        # meaningful with feature-based drug representations (guide §4.4).
        if cfg.data.drug_features != "fingerprint":
            raise ValueError(
                "LDO requires data.drug_features=fingerprint: a one-hot drug identity cannot "
                "generalise to a drug the model has never seen."
            )
        if cfg.loss.heteroscedastic and not cfg.loss.sigma_from_drug_features:
            raise ValueError(
                "LDO with a per-drug sigma requires loss.sigma_from_drug_features=true; an "
                "embedding table has no row for an unseen drug."
            )

    model = build_model(cfg, data.token_spec, drug_feature_dim=data.drug_features.shape[1],
                        n_drugs=len(data.drug_ids)).to(device)
    is_curve = cfg.model.head == "curve"
    if is_curve:
        criterion = ViabilityLikelihood(
            cfg.loss, n_drugs=len(data.drug_ids), drug_feature_dim=data.drug_features.shape[1]
        ).to(device)
    else:
        criterion = ScalarRegressionLoss(cfg.loss.scalar_loss).to(device)

    params = model.param_groups(cfg.train.weight_decay)
    params.append({"params": [p for p in criterion.parameters() if p.requires_grad], "weight_decay": 0.0})
    opt = torch.optim.AdamW(params, lr=cfg.train.lr)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup(s, total_steps, warmup_steps)
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    naive = NaiveMeanEffects().fit(data.omics, data.cell_index, data.drug_index, targets, train_idx)

    jsonl = JsonlLogger(Path(cfg.paths.log_dir) / f"{tag}.jsonl", run_id=f"{tag}-f{fold}-s{seed}")
    jsonl.log("run_start", tag=tag, fold=fold, seed=seed, head=cfg.model.head,
              n_train=int(train_idx.size), n_val=int(val_idx.size), n_test=int(test_idx.size),
              n_parameters=model.n_parameters(), device=str(device))

    best_metric = -float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    patience_left = cfg.train.patience
    stopped_early = False

    for epoch in range(cfg.train.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            omics = batch["omics"].to(device, non_blocking=True)
            drug_features = batch["drug_features"].to(device, non_blocking=True)
            drug_index = batch["drug_index"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ctx = (
                torch.autocast(device_type=device.type, dtype=dtype)
                if dtype is not None and device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with ctx:
                out = model(omics, drug_features)
                if is_curve:
                    log_conc = batch["log_conc"].to(device, non_blocking=True)
                    viability = batch["viability"].to(device, non_blocking=True)
                    mask = batch["point_mask"].to(device, non_blocking=True)
                    params_c = model.curve_params(out)
                    pred_v = params_c.viability(log_conc)
                    loss, parts = criterion(pred_v, viability, mask, drug_index, drug_features)
                else:
                    target = batch["ln_ic50_target"].to(device, non_blocking=True)
                    loss, parts = criterion(out["ln_ic50_pred"], target)
            if not torch.isfinite(loss):
                jsonl.log("nonfinite_loss", epoch=epoch)
                raise FloatingPointError(
                    f"loss became {loss.item()} at epoch {epoch}; training diverged "
                    "(lower train.lr or check the input scaling)"
                )
            if scaler_amp.is_enabled():
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
            sched.step()
            running += float(loss.detach())
            n_batches += 1

        val_arrays = predict_pairs(model, val_loader, device, dtype, criterion if is_curve else None)
        val_naive = naive.predict(data.omics, data.cell_index, data.drug_index,
                                  val_arrays["pair_row"].astype(int))
        val_eval = evaluate_predictions(
            val_arrays, data.pairs, naive_ln_ic50=val_naive, reference=cfg.eval.reference,
            refit=refit, min_pairs_per_drug=cfg.eval.min_pairs_per_drug, bootstrap_n=0,
        )
        primary = val_eval.metrics.get("delta_pcc_vs_naive", float("nan"))
        if not np.isfinite(primary):
            # too few drugs to correlate: fall back to the likelihood itself
            primary = -val_eval.metrics.get("viability_rmse", float("inf"))
        record = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_primary": float(primary),
            "val_viability_rmse": val_eval.metrics.get("viability_rmse", float("nan")),
            "val_per_drug_pcc": val_eval.metrics.get("per_drug_pcc", float("nan")),
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(record)
        jsonl.log("epoch", **record)

        if epoch_callback is not None:
            epoch_callback(epoch, float(primary))

        if primary > best_metric + 1e-6:
            best_metric, best_epoch = float(primary), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_criterion = {k: v.detach().cpu().clone() for k, v in criterion.state_dict().items()}
            patience_left = cfg.train.patience
        else:
            patience_left -= 1
            if patience_left <= 0 and epoch + 1 >= cfg.train.min_epochs:
                stopped_early = True
                jsonl.log("early_stop", epoch=epoch, best_epoch=best_epoch, best_metric=best_metric)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        criterion.load_state_dict(best_criterion)

    val_arrays = predict_pairs(model, val_loader, device, dtype, criterion if is_curve else None)
    val_naive = naive.predict(data.omics, data.cell_index, data.drug_index, val_arrays["pair_row"].astype(int))
    val_eval = evaluate_predictions(val_arrays, data.pairs, val_naive, cfg.eval.reference, refit,
                                    cfg.eval.min_pairs_per_drug, cfg.eval.bootstrap_n)
    test_arrays = predict_pairs(model, test_loader, device, dtype, criterion if is_curve else None)
    test_naive = naive.predict(data.omics, data.cell_index, data.drug_index, test_arrays["pair_row"].astype(int))
    test_eval = evaluate_predictions(test_arrays, data.pairs, test_naive, cfg.eval.reference, refit,
                                     cfg.eval.min_pairs_per_drug, cfg.eval.bootstrap_n)

    ckpt_path = out_dir / f"{tag}_fold{fold}_seed{seed}.pt"
    if save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "config": cfg.to_dict(),
                "scaler": scaler.state_dict(),
                "fold": fold,
                "seed": seed,
                "best_epoch": best_epoch,
            },
            ckpt_path,
        )
    if save_predictions:
        save_eval(test_eval, out_dir / "predictions", f"{tag}_fold{fold}_seed{seed}")

    outcome = TrainOutcome(
        best_metric=best_metric,
        best_epoch=best_epoch,
        history=history,
        val_metrics=val_eval.metrics,
        test_metrics=test_eval.metrics,
        gammas=model.gammas(),
        checkpoint=str(ckpt_path) if save_checkpoint else None,
        n_parameters=model.n_parameters(),
        seconds=time.time() - t0,
        stopped_early=stopped_early,
    )
    jsonl.log("run_end", **outcome.as_row())
    jsonl.close()
    log.info(
        "%s fold %d seed %d: best epoch %d, val dPCC %.4f, test dPCC %.4f, %.0fs",
        tag, fold, seed, best_epoch,
        val_eval.metrics.get("delta_pcc_vs_naive", float("nan")),
        test_eval.metrics.get("delta_pcc_vs_naive", float("nan")),
        outcome.seconds,
    )
    return outcome


def train_all_folds(
    cfg: Config,
    data: PairData,
    scheme: str | None = None,
    folds: list[int] | None = None,
    seed: int | None = None,
    tag: str = "hill",
) -> pd.DataFrame:
    """Train every requested fold of one split scheme; returns a tidy result table."""
    scheme = scheme or cfg.splits.primary
    seed = cfg.seed if seed is None else seed
    splits = load_or_build_splits(
        data.pairs, scheme, cfg.paths.splits_dir,
        n_folds=cfg.splits.n_folds, seed=cfg.splits.seed, val_fraction=cfg.splits.val_fraction,
    )
    refit = _load_refit(cfg)
    rows = []
    for fold in (folds if folds is not None else range(cfg.splits.n_folds)):
        outcome = run_fold(
            cfg, data, splits, fold, seed,
            out_dir=Path(cfg.paths.checkpoint_dir) / tag, tag=tag, refit=refit,
        )
        rows.append({"tag": tag, "scheme": scheme, "fold": fold, "seed": seed, **outcome.as_row()})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train HILL / ScalarHILL")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--scheme", default=None, help="LCO (default) / LPO / LDO")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="hill")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    configure_runtime(cfg)
    data = load_pair_data(cfg)
    capture_provenance(Path(cfg.paths.results_dir) / f"{args.tag}_provenance.json", cfg)

    table = train_all_folds(cfg, data, args.scheme, args.folds, args.seed, args.tag)
    out = Path(cfg.paths.results_dir) / f"{args.tag}_folds.csv"
    table.to_csv(out, index=False)
    print(table[["tag", "fold", "seed", "test_delta_pcc_vs_naive", "test_per_drug_pcc",
                 "test_viability_rmse"]].to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
