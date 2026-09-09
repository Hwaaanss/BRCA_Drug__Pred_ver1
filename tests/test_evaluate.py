"""Metric policy: per-drug only, naive normalisation, and the global-PCC refusal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hill.evaluate import (
    GlobalPCCForbiddenError, clinical_metrics, evaluate_predictions, guard_metric,
    per_drug_correlation,
)


@pytest.mark.parametrize("name", ["global_pcc", "overall_pcc", "POOLED_PCC"])
def test_global_pcc_is_refused(name):
    with pytest.raises(GlobalPCCForbiddenError):
        guard_metric(name)


def test_per_drug_correlation_is_computed_within_drug():
    """Two drugs with opposite within-drug trends must not average to a high value."""
    drug = np.array([0] * 10 + [1] * 10)
    obs = np.concatenate([np.arange(10), np.arange(10)]).astype(float)
    pred = np.concatenate([np.arange(10), -np.arange(10)]).astype(float)
    out = per_drug_correlation(pred, obs, drug, min_pairs=5)
    assert len(out) == 2
    assert out["corr"].iloc[0] > 0.99 and out["corr"].iloc[1] < -0.99
    assert abs(float(out["corr"].mean())) < 1e-6


def test_min_pairs_threshold_produces_nan_not_a_number():
    drug = np.array([0, 0, 1, 1, 1])
    out = per_drug_correlation(np.arange(5.0), np.arange(5.0), drug, min_pairs=4)
    assert np.isnan(out["corr"]).all()


def test_evaluate_predictions_reports_undefined_ic50_both_ways():
    n = 20
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(n),
            "cell_id": [str(i) for i in range(n)],
            "drug_id": np.repeat([0, 1], n // 2),
            "ln_ic50_published": np.linspace(-2, 2, n),
            "censored": np.zeros(n, dtype=bool),
        }
    )
    arrays = {
        "pair_row": np.arange(n),
        "drug_index": np.repeat([0, 1], n // 2),
        "e_inf": np.linspace(0.1, 0.9, n),
        "slope": np.ones(n),
        "midpoint": np.linspace(-2, 2, n),
        "ln_ic50_pred": np.linspace(-2, 2, n),
        "ic50_defined": np.linspace(0.1, 0.9, n) < 0.5,
        "emax_pred": 1 - np.linspace(0.1, 0.9, n),
        "auc_pred": np.full(n, 0.5),
        "log_c_min": np.full(n, -4.0),
        "log_c_max": np.full(n, 4.0),
        "viability_sse": np.full(n, 0.01),
        "n_points": np.array([n * 5]),
        "loglik_total": np.array([-10.0]),
    }
    res = evaluate_predictions(arrays, pairs, naive_ln_ic50=np.zeros(n), min_pairs_per_drug=3,
                               bootstrap_n=0)
    assert 0 < res.metrics["frac_ic50_undefined_pred"] < 1
    assert "per_drug_pcc" in res.metrics and "per_drug_pcc_incl_undefined" in res.metrics
    assert "delta_pcc_vs_naive" in res.metrics
    assert len(res.predictions) == n


def test_clinical_metrics_handle_a_degenerate_label():
    out = clinical_metrics(np.zeros(10), np.ones(10))
    assert np.isnan(out["roc_auc"])
