"""Evaluation.

Reported metrics (guide §7.1)
    viability RMSE                point level; how well the curve fits
    per-drug PCC of derived IC50  computed per drug, then averaged
    dPCC vs NaiveMeanEffects      PRIMARY metric (DrEval-style normalisation)
    Emax correlation              only this design can produce it
    clinical AUC                  TCGA, evaluated at Cmax
    gamma_E / gamma_m + LRT       histology gate

Forbidden
    Drug-pooled ("global") correlation is not computed.  With a large drug main
    effect (audit G-4) it mostly measures which drug a row belongs to.  Asking
    for it raises :class:`GlobalPCCForbiddenError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from hill.constants import FORBIDDEN_METRICS
from hill.derive import auc_logspace, ln_ic50
from hill.utils.logging import get_logger
from hill.utils.stats import bootstrap_ci, paired_test, pearson, spearman

log = get_logger("evaluate")


class GlobalPCCForbiddenError(RuntimeError):
    """Raised when drug-pooled correlation is requested.

    See guide §7.1 and the G-4 audit: with f_drug large this number is an
    artefact of the drug main effect, so the project does not report it.
    """


def guard_metric(name: str) -> None:
    if name.strip().lower() in FORBIDDEN_METRICS:
        raise GlobalPCCForbiddenError(
            f"metric {name!r} is deliberately not implemented. Report per-drug metrics and "
            "dPCC against NaiveMeanEffects instead (guide §7.1; see results/gate0/g4_variance.json)."
        )


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_drug: pd.DataFrame
    predictions: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self, **extra: Any) -> dict[str, Any]:
        row = dict(self.metrics)
        row.update(extra)
        return row


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_pairs(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    loss_module: Any = None,
) -> dict[str, np.ndarray]:
    """Run the model over a loader and collect per-pair predictions.

    Also accumulates the total observation log-likelihood, which the
    likelihood-ratio test for the histology gates needs.
    """
    model.eval()
    keys = ("pair_row", "drug_index", "e_inf", "slope", "midpoint", "ln_ic50_pred")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    point_sq_err: list[np.ndarray] = []
    point_count = 0
    loglik_total = 0.0
    log_c_min, log_c_max = [], []

    for batch in loader:
        omics = batch["omics"].to(device, non_blocking=True)
        drug_features = batch["drug_features"].to(device, non_blocking=True)
        log_conc = batch["log_conc"].to(device, non_blocking=True)
        viability = batch["viability"].to(device, non_blocking=True)
        mask = batch["point_mask"].to(device, non_blocking=True)
        drug_index = batch["drug_index"].to(device, non_blocking=True)

        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            res = model(omics, drug_features)

        if "e_inf" in res:
            e = res["e_inf"].float()
            s = res["slope"].float()
            m = res["midpoint"].float()
            pred_v = e + (1.0 - e) * torch.sigmoid(
                torch.clamp(-s * (log_conc - m), -30.0, 30.0)
            )
            if loss_module is not None:
                ls = loss_module.log_scale(e.shape[0], e.device, drug_index, drug_features)
                loglik_total += float(loss_module.pointwise_loglik(pred_v, viability, mask, ls).sum())
            err = ((pred_v - viability) ** 2 * mask).sum(dim=1)
            point_sq_err.append(err.cpu().numpy())
            point_count += int(mask.sum())
            out["e_inf"].append(e.squeeze(-1).cpu().numpy())
            out["slope"].append(s.squeeze(-1).cpu().numpy())
            out["midpoint"].append(m.squeeze(-1).cpu().numpy())
            out["ln_ic50_pred"].append(np.full(e.shape[0], np.nan, dtype=np.float32))
        else:
            pred = res["ln_ic50_pred"].float().squeeze(-1)
            out["ln_ic50_pred"].append(pred.cpu().numpy())
            for k in ("e_inf", "slope", "midpoint"):
                out[k].append(np.full(pred.shape[0], np.nan, dtype=np.float32))

        out["pair_row"].append(batch["pair_row"].numpy())
        out["drug_index"].append(batch["drug_index"].numpy())
        lo = torch.where(mask, log_conc, torch.full_like(log_conc, float("inf"))).min(dim=1).values
        hi = torch.where(mask, log_conc, torch.full_like(log_conc, float("-inf"))).max(dim=1).values
        log_c_min.append(lo.cpu().numpy())
        log_c_max.append(hi.cpu().numpy())

    arrays = {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}
    arrays["log_c_min"] = np.concatenate(log_c_min) if log_c_min else np.array([])
    arrays["log_c_max"] = np.concatenate(log_c_max) if log_c_max else np.array([])
    arrays["viability_sse"] = np.concatenate(point_sq_err) if point_sq_err else np.array([])
    arrays["n_points"] = np.array([point_count], dtype=np.int64)
    arrays["loglik_total"] = np.array([loglik_total], dtype=np.float64)

    # derived quantities for curve models
    if arrays["e_inf"].size and np.isfinite(arrays["e_inf"]).any():
        e = torch.from_numpy(arrays["e_inf"]).unsqueeze(1).double()
        s = torch.from_numpy(arrays["slope"]).unsqueeze(1).double()
        m = torch.from_numpy(arrays["midpoint"]).unsqueeze(1).double()
        ic50, defined = ln_ic50(e, s, m)
        arrays["ln_ic50_pred"] = ic50.squeeze(1).numpy()
        arrays["ic50_defined"] = defined.squeeze(1).numpy()
        arrays["emax_pred"] = (1.0 - e).squeeze(1).numpy()
        arrays["auc_pred"] = auc_logspace(
            e, s, m,
            torch.from_numpy(arrays["log_c_min"]).unsqueeze(1).double(),
            torch.from_numpy(arrays["log_c_max"]).unsqueeze(1).double(),
            normalized=True,
        ).squeeze(1).numpy()
    else:
        arrays["ic50_defined"] = np.ones_like(arrays["ln_ic50_pred"], dtype=bool)
        arrays["emax_pred"] = np.full_like(arrays["ln_ic50_pred"], np.nan)
        arrays["auc_pred"] = np.full_like(arrays["ln_ic50_pred"], np.nan)
    return arrays


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def per_drug_correlation(
    pred: np.ndarray,
    obs: np.ndarray,
    drug: np.ndarray,
    min_pairs: int = 5,
    method: str = "pearson",
) -> pd.DataFrame:
    """Correlation computed separately within each drug (never pooled)."""
    fn = pearson if method == "pearson" else spearman
    rows = []
    for d in np.unique(drug):
        sel = drug == d
        p, o = pred[sel], obs[sel]
        ok = np.isfinite(p) & np.isfinite(o)
        rows.append(
            {
                "drug_index": int(d),
                "n_pairs": int(sel.sum()),
                "n_usable": int(ok.sum()),
                "corr": fn(p[ok], o[ok]) if ok.sum() >= min_pairs else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    arrays: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    naive_ln_ic50: np.ndarray | None = None,
    reference: str = "published",
    refit: pd.DataFrame | None = None,
    min_pairs_per_drug: int = 5,
    bootstrap_n: int = 1000,
) -> EvalResult:
    """Turn raw predictions into the reported metric set."""
    rows = arrays["pair_row"].astype(int)
    sub = pairs.iloc[rows].reset_index(drop=True)
    drug = arrays["drug_index"].astype(int)

    obs_published = sub["ln_ic50_published"].to_numpy(dtype=float)
    censored = sub["censored"].fillna(False).to_numpy(dtype=bool) if "censored" in sub else np.zeros(len(sub), bool)

    obs_refit = np.full(len(sub), np.nan)
    obs_emax = np.full(len(sub), np.nan)
    if refit is not None and len(refit):
        r = refit.set_index("pair_id")
        pid = sub["pair_id"].to_numpy()
        have = np.isin(pid, r.index.to_numpy())
        obs_refit[have] = r.loc[pid[have], "m3_ln_ic50"].to_numpy(dtype=float)
        obs_emax[have] = r.loc[pid[have], "m3_emax"].to_numpy(dtype=float)

    obs = obs_refit if reference == "refit_m3" else obs_published
    pred = arrays["ln_ic50_pred"].astype(float)
    defined = arrays["ic50_defined"].astype(bool)

    metrics: dict[str, float] = {}
    n_points = int(arrays["n_points"][0]) if arrays["n_points"].size else 0
    metrics["viability_rmse"] = (
        float(np.sqrt(arrays["viability_sse"].sum() / max(n_points, 1))) if arrays["viability_sse"].size else float("nan")
    )
    metrics["n_pairs"] = float(len(sub))
    metrics["n_points"] = float(n_points)
    metrics["loglik_total"] = float(arrays["loglik_total"][0]) if arrays["loglik_total"].size else float("nan")
    metrics["frac_ic50_undefined_pred"] = float((~defined).mean()) if defined.size else float("nan")
    metrics["frac_censored_reference"] = float(censored.mean()) if censored.size else float("nan")

    # --- per-drug IC50 correlation, excluding undefined predictions ---------
    pd_excl = per_drug_correlation(
        np.where(defined, pred, np.nan), obs, drug, min_pairs_per_drug
    ).rename(columns={"corr": "pcc_excl_undefined"})
    metrics["per_drug_pcc"] = float(np.nanmean(pd_excl["pcc_excl_undefined"])) if len(pd_excl) else float("nan")

    # --- sensitivity: include undefined predictions, imputed conservatively --
    # an undefined IC50 means "beyond the tested range"; the imputation is only
    # used for this one sensitivity metric and never for training.
    imputed = np.where(defined, pred, arrays["log_c_max"] + 1.0)
    pd_incl = per_drug_correlation(imputed, obs, drug, min_pairs_per_drug).rename(
        columns={"corr": "pcc_incl_undefined"}
    )
    metrics["per_drug_pcc_incl_undefined"] = (
        float(np.nanmean(pd_incl["pcc_incl_undefined"])) if len(pd_incl) else float("nan")
    )

    # --- uncensored subset ---------------------------------------------------
    unc = ~censored
    if unc.sum() > 0:
        pd_unc = per_drug_correlation(
            np.where(defined & unc, pred, np.nan), np.where(unc, obs, np.nan), drug, min_pairs_per_drug
        ).rename(columns={"corr": "pcc_uncensored"})
        metrics["per_drug_pcc_uncensored"] = float(np.nanmean(pd_unc["pcc_uncensored"]))
    else:
        pd_unc = pd.DataFrame({"drug_index": pd_excl["drug_index"], "pcc_uncensored": np.nan})
        metrics["per_drug_pcc_uncensored"] = float("nan")

    per_drug = pd_excl.merge(
        pd_incl[["drug_index", "pcc_incl_undefined"]], on="drug_index", how="left"
    ).merge(pd_unc[["drug_index", "pcc_uncensored"]], on="drug_index", how="left")

    # --- PRIMARY metric: dPCC against the naive mean-effects predictor -------
    if naive_ln_ic50 is not None:
        pd_naive = per_drug_correlation(naive_ln_ic50, obs, drug, min_pairs_per_drug).rename(
            columns={"corr": "pcc_naive"}
        )
        per_drug = per_drug.merge(pd_naive[["drug_index", "pcc_naive"]], on="drug_index", how="left")
        delta = per_drug["pcc_excl_undefined"] - per_drug["pcc_naive"]
        per_drug["delta_pcc"] = delta
        metrics["delta_pcc_vs_naive"] = float(np.nanmean(delta))
        point, lo, hi = bootstrap_ci(delta.to_numpy(), n_boot=bootstrap_n)
        metrics["delta_pcc_ci_low"] = lo
        metrics["delta_pcc_ci_high"] = hi
        test = paired_test(
            per_drug["pcc_excl_undefined"].to_numpy(), per_drug["pcc_naive"].to_numpy()
        )
        metrics["delta_pcc_wilcoxon_p"] = test["wilcoxon_p"]
        metrics["n_drugs_beating_naive"] = float((delta > 0).sum())
        metrics["n_drugs_evaluated"] = float(delta.notna().sum())
    else:
        metrics["delta_pcc_vs_naive"] = float("nan")

    # --- Emax: unique to the curve formulation -------------------------------
    if np.isfinite(obs_emax).any() and np.isfinite(arrays["emax_pred"]).any():
        pd_emax = per_drug_correlation(
            arrays["emax_pred"].astype(float), obs_emax, drug, min_pairs_per_drug
        ).rename(columns={"corr": "pcc_emax"})
        per_drug = per_drug.merge(pd_emax[["drug_index", "pcc_emax"]], on="drug_index", how="left")
        metrics["per_drug_pcc_emax"] = float(np.nanmean(pd_emax["pcc_emax"]))
    else:
        metrics["per_drug_pcc_emax"] = float("nan")

    predictions = pd.DataFrame(
        {
            "pair_id": sub["pair_id"].to_numpy(),
            "cell_id": sub["cell_id"].to_numpy(),
            "drug_id": sub["drug_id"].to_numpy(),
            "drug_index": drug,
            "e_inf_pred": arrays["e_inf"],
            "slope_pred": arrays["slope"],
            "midpoint_pred": arrays["midpoint"],
            "ln_ic50_pred": pred,
            "ic50_defined": defined,
            "emax_pred": arrays["emax_pred"],
            "auc_pred": arrays["auc_pred"],
            "ln_ic50_published": obs_published,
            "ln_ic50_refit_m3": obs_refit,
            "emax_refit_m3": obs_emax,
            "censored": censored,
            "log_c_min": arrays["log_c_min"],
            "log_c_max": arrays["log_c_max"],
        }
    )
    if naive_ln_ic50 is not None:
        predictions["ln_ic50_naive"] = naive_ln_ic50

    return EvalResult(metrics=metrics, per_drug=per_drug, predictions=predictions,
                      meta={"reference": reference})


def clinical_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """ROC-AUC / average precision / Brier for the TCGA response head."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(labels, dtype=float)
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": float(y.size), "positive_rate": float(y.mean()) if y.size else float("nan")}
    if y.size < 5 or len(np.unique(y)) < 2:
        out.update({"roc_auc": float("nan"), "average_precision": float("nan"), "brier": float("nan")})
        return out
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["average_precision"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    return out


def save_eval(result: EvalResult, out_dir: str | Path, tag: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.metrics]).to_csv(d / f"{tag}_metrics.csv", index=False)
    result.per_drug.to_csv(d / f"{tag}_per_drug.csv", index=False)
    result.predictions.to_parquet(d / f"{tag}_predictions.parquet", index=False)
