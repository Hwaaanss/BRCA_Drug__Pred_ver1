"""Stage 1 — TCGA fine-tuning, histology gates, and the likelihood-ratio test.

The Stage-0 model is loaded, the histology branch is attached, and the gate
scalars ``gamma_E`` / ``gamma_m`` are released one at a time:

    raw_e = base_e(z) + gamma_E * h_E(H_i, z)
    raw_m = base_m(z) + gamma_m * h_m(H_i, z)

Because both gates start at exactly 0, the restricted model (gamma = 0) and the
unrestricted model are *exactly* nested — ``tests/test_gamma.py`` asserts bitwise
identical outputs — which is what makes the likelihood-ratio test meaningful.

    python -m hill.stage1 --config configs/base.yaml --checkpoint <stage0.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hill.config import Config, load_config
from hill.data.datasets import ClinicalDataset, PairData, Standardizer, collate_clinical
from hill.data.drugs import load_compound_annotation
from hill.data.prepare import load_pair_data
from hill.data.tcga import align_to_feature_universe, build_clinical_table, load_cmax_table
from hill.evaluate import clinical_metrics
from hill.losses import ClinicalResponseLoss
from hill.models.hill import build_model
from hill.utils.logging import JsonlLogger, get_logger
from hill.utils.resources import configure_runtime, resolve_device
from hill.utils.seed import seed_everything
from hill.utils.stats import lrt_boundary_pvalue

log = get_logger("stage1")


@dataclass
class TCGABundle:
    omics: np.ndarray
    patient_ids: list[str]
    clinical: pd.DataFrame
    coverage: dict[str, float]
    histology_dir: Path | None
    meta: dict[str, Any] = field(default_factory=dict)


def load_tcga_bundle(cfg: Config, data: PairData) -> TCGABundle:
    tcga_dir = Path(cfg.data.tcga_dir)
    matrices: dict[str, pd.DataFrame] = {}
    for modality, fname in (("expression", "expression_tpm.parquet"), ("mutation", "mutations_binary.parquet"),
                            ("cnv", "cnv.parquet")):
        path = tcga_dir / fname
        if path.exists():
            matrices[modality] = pd.read_parquet(path)
    if not matrices:
        raise FileNotFoundError(
            f"no TCGA omics matrices under {tcga_dir}. Run `python -m hill.data.download_tcga`."
        )

    omics, patient_ids, coverage = align_to_feature_universe(
        matrices,
        data.meta["feature_names"],
        data.meta["feature_modality"],
        log1p_expression=cfg.data.expression_log1p,
    )

    treatments = pd.read_csv(tcga_dir / "treatments.csv")
    ann_path = Path(cfg.paths.raw_dir) / "gdsc_annotation" / "screened_compounds.csv"
    annotation = load_compound_annotation(ann_path)
    cmax = load_cmax_table(cfg.data.clinical_cmax_csv, require_verified=cfg.data.require_verified_cmax)
    clinical = build_clinical_table(treatments, annotation, cmax, data.drug_ids)
    clinical = clinical[clinical["patient_id"].astype(str).isin(set(patient_ids))].reset_index(drop=True)
    if clinical.empty:
        raise ValueError("no TCGA (patient, drug) row survived the omics/Cmax/drug joins")

    histo_dir = Path(cfg.data.histology_dir) if Path(cfg.data.histology_dir).exists() else None
    return TCGABundle(
        omics=omics, patient_ids=patient_ids, clinical=clinical, coverage=coverage,
        histology_dir=histo_dir,
        meta={"n_rows": int(len(clinical)), "n_patients": len(patient_ids)},
    )


def _patient_folds(patients: Sequence[str], n_folds: int, seed: int) -> dict[str, int]:
    uniq = np.array(sorted(set(patients)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    return {p: i % n_folds for i, p in enumerate(uniq)}


def _build_loaders(
    bundle: TCGABundle, cfg: Config, train_rows: np.ndarray, test_rows: np.ndarray,
    drug_features: np.ndarray, seed: int
) -> tuple[DataLoader, DataLoader, Standardizer]:
    pos = {p: i for i, p in enumerate(bundle.patient_ids)}
    row_patient = bundle.clinical["patient_id"].astype(str).map(pos).to_numpy()
    train_patients = np.unique(row_patient[train_rows])
    scaler = Standardizer().fit(bundle.omics[train_patients])
    omics_scaled = scaler.transform(bundle.omics)

    def make(rows: np.ndarray, shuffle: bool, batch: int) -> DataLoader:
        ds = ClinicalDataset(
            bundle.clinical.iloc[rows], omics_scaled, row_patient[rows], drug_features,
            histology_dir=bundle.histology_dir, max_patches=cfg.data.max_patches,
            histo_feature_dim=cfg.data.histo_feature_dim, rng_seed=seed,
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate_clinical,
                          num_workers=0)

    return (
        make(train_rows, True, cfg.train.batch_size),
        make(test_rows, False, cfg.train.eval_batch_size),
        scaler,
    )


def _forward_clinical(model, batch, device, head: ClinicalResponseLoss):
    omics = batch["omics"].to(device)
    drug_features = batch["drug_features"].to(device)
    histology = batch.get("histology")
    histo_mask = batch.get("histo_mask")
    if histology is not None:
        histology = histology.to(device)
        histo_mask = histo_mask.to(device)
    out = model(omics, drug_features, histology, histo_mask, batch["slide_available"].to(device))
    log_cmax = batch["log_cmax"].to(device).unsqueeze(-1)
    params = model.curve_params(out)
    viability = params.viability(log_cmax).squeeze(-1)
    return head.logits(viability), out


def run_stage1(
    cfg: Config,
    data: PairData,
    bundle: TCGABundle,
    checkpoint: str | Path,
    gates: tuple[bool, bool] = (True, True),
    seed: int = 0,
    n_folds: int = 5,
    tag: str = "stage1",
) -> dict[str, Any]:
    """Fine-tune on TCGA with the requested gates released; returns metrics + log-likelihood."""
    t0 = time.time()
    seed_everything(seed, cfg.train.deterministic)
    device = resolve_device(cfg)
    gate_e, gate_m = gates

    cfg_h = cfg.copy_with(
        {
            "model.histology.enabled": True,
            "model.histology.gate_efficacy": bool(gate_e),
            "model.histology.gate_potency": bool(gate_m),
            "model.histology.train_gamma": True,
            "model.histology.feature_dim": cfg.data.histo_feature_dim,
        }
    )
    fold_of = _patient_folds(bundle.clinical["patient_id"].astype(str).tolist(), n_folds, seed)
    fold_ids = bundle.clinical["patient_id"].astype(str).map(fold_of).to_numpy()

    jsonl = JsonlLogger(Path(cfg.paths.log_dir) / f"{tag}.jsonl", run_id=f"{tag}-s{seed}")
    all_logits, all_labels, gammas, loglik = [], [], [], 0.0

    for fold in range(n_folds):
        train_rows = np.flatnonzero(fold_ids != fold)
        test_rows = np.flatnonzero(fold_ids == fold)
        if test_rows.size == 0 or train_rows.size == 0:
            continue
        train_loader, test_loader, _ = _build_loaders(
            bundle, cfg_h, train_rows, test_rows, data.drug_features, seed
        )

        model = build_model(cfg_h, data.token_spec, data.drug_features.shape[1], len(data.drug_ids))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        log.info("loaded stage-0 weights (%d new tensors for the histology branch)", len(missing))
        model = model.to(device)
        model.set_gamma_trainable(True)

        head = ClinicalResponseLoss().to(device)
        params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.train.stage1_lr, weight_decay=cfg.train.weight_decay)

        model.train()
        for epoch in range(cfg.train.stage1_epochs):
            total = 0.0
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                logits, _ = _forward_clinical(model, batch, device, head)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["responder"].to(device)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                opt.step()
                total += float(loss.detach())
            jsonl.log("stage1_epoch", fold=fold, epoch=epoch, loss=total / max(len(train_loader), 1),
                      **model.gammas())

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                logits, _ = _forward_clinical(model, batch, device, head)
                y = batch["responder"].to(device)
                loglik += float(
                    -torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
                )
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        gammas.append(model.gammas())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = clinical_metrics(logits, labels)
    gamma_summary = {
        k: float(np.mean([g[k] for g in gammas])) for k in (gammas[0] if gammas else {})
    }
    result = {
        "tag": tag,
        "seed": seed,
        "gate_efficacy": bool(gate_e),
        "gate_potency": bool(gate_m),
        "clinical": metrics,
        "gammas": gamma_summary,
        "gamma_per_fold": gammas,
        "loglik": loglik,
        "n_rows": int(len(bundle.clinical)),
        "seconds": time.time() - t0,
        "feature_coverage": bundle.coverage,
    }
    jsonl.log("stage1_end", **{k: v for k, v in result.items() if k != "gamma_per_fold"})
    jsonl.close()
    log.info("%s: clinical AUC %.3f, gammas %s", tag, metrics.get("roc_auc", float("nan")), gamma_summary)
    return result


def run_lrt(
    cfg: Config, data: PairData, bundle: TCGABundle, checkpoint: str | Path, seed: int = 0,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Likelihood-ratio tests for gamma_E and gamma_m, each against gamma = 0."""
    restricted = run_stage1(cfg, data, bundle, checkpoint, (False, False), seed, n_folds, "stage1_gamma0")
    gate_e = run_stage1(cfg, data, bundle, checkpoint, (True, False), seed, n_folds, "stage1_gammaE")
    gate_m = run_stage1(cfg, data, bundle, checkpoint, (False, True), seed, n_folds, "stage1_gammaM")
    both = run_stage1(cfg, data, bundle, checkpoint, (True, True), seed, n_folds, "stage1_gammaEM")

    out = {
        "restricted": restricted,
        "gamma_E": gate_e,
        "gamma_m": gate_m,
        "both": both,
        "lrt_gamma_E": lrt_boundary_pvalue(gate_e["loglik"], restricted["loglik"]),
        "lrt_gamma_m": lrt_boundary_pvalue(gate_m["loglik"], restricted["loglik"]),
        "interpretation": (
            "gamma_E significant and gamma_m not => tissue morphology acts on efficacy (E_inf) "
            "rather than potency (m), which is the hypothesis that requires the two gates to stay "
            "separate. Both non-significant => no evidence that morphology adds anything here."
        ),
    }
    log.info(
        "LRT: gamma_E lambda=%.2f p=%.4f | gamma_m lambda=%.2f p=%.4f",
        out["lrt_gamma_E"]["lambda"], out["lrt_gamma_E"]["p_boundary"],
        out["lrt_gamma_m"]["lambda"], out["lrt_gamma_m"]["p_boundary"],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage-1 TCGA fine-tuning and the gamma LRT")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lrt", action="store_true", help="run the full nested-model LRT")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    configure_runtime(cfg)
    data = load_pair_data(cfg)
    bundle = load_tcga_bundle(cfg, data)

    out_dir = Path(cfg.paths.results_dir) / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.lrt:
        result = run_lrt(cfg, data, bundle, args.checkpoint, args.seed, args.folds)
        (out_dir / f"lrt_seed{args.seed}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    else:
        result = run_stage1(cfg, data, bundle, args.checkpoint, (True, True), args.seed, args.folds)
        (out_dir / f"stage1_seed{args.seed}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
