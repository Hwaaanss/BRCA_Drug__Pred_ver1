"""IC50 / AUC / Emax derivations (guide §5.2)."""

from __future__ import annotations

import torch

from hill.derive import auc_logspace, auc_numeric, emax, ln_ic50, ln_ic50_scalar
from hill.models.curve_head import hill_curve


def test_ic50_roundtrip(curve_params):
    """parameters -> IC50 -> r(IC50) must return exactly 0.5."""
    e, s, m = curve_params
    ic50, defined = ln_ic50(e, s, m)
    assert bool(defined.any())
    sel = defined.squeeze(-1)
    r = hill_curve(ic50[sel].unsqueeze(-1), e[sel].unsqueeze(-1), s[sel].unsqueeze(-1),
                   m[sel].unsqueeze(-1))
    assert torch.allclose(r, torch.full_like(r, 0.5), atol=1e-5)


def test_ic50_undefined_when_einf_ge_05():
    """E_inf >= 0.5 means the curve never crosses 50%: no IC50 exists."""
    e = torch.tensor([[0.5], [0.6], [0.99]])
    s = torch.ones(3, 1)
    m = torch.zeros(3, 1)
    values, defined = ln_ic50(e, s, m)
    assert not bool(defined.any())
    assert bool(torch.isnan(values).all()), "undefined IC50 must be NaN, never a sentinel value"
    assert ln_ic50_scalar(0.5, 1.0, 0.0) is None
    assert ln_ic50_scalar(0.7, 2.0, 1.0) is None
    assert ln_ic50_scalar(0.0, 1.0, 3.0) == 3.0


def test_ic50_matches_closed_form_when_einf_zero():
    """With a floor at zero the midpoint *is* the ln IC50."""
    e = torch.full((5, 1), 1e-9)
    s = torch.rand(5, 1) + 0.5
    m = torch.randn(5, 1)
    values, defined = ln_ic50(e, s, m)
    assert bool(defined.all())
    assert torch.allclose(values, m, atol=1e-4)


def test_auc_closed_form_vs_numeric(curve_params):
    """The closed form must agree with numeric integration to < 1e-4 relative."""
    e, s, m = curve_params
    lo = torch.full_like(e, -6.0)
    hi = torch.full_like(e, 4.0)
    closed = auc_logspace(e, s, m, lo, hi, normalized=True).double()
    numeric = auc_numeric(e.double(), s.double(), m.double(), lo.double(), hi.double(), n_grid=4096)
    rel = ((closed - numeric).abs() / numeric.abs().clamp_min(1e-9)).max()
    assert float(rel) < 1e-4, f"relative error {float(rel):.2e}"


def test_auc_bounds(curve_params):
    e, s, m = curve_params
    lo, hi = torch.full_like(e, -5.0), torch.full_like(e, 5.0)
    auc = auc_logspace(e, s, m, lo, hi, normalized=True)
    assert bool(((auc >= e - 1e-6) & (auc <= 1.0 + 1e-6)).all())


def test_emax_is_one_minus_einf():
    e = torch.rand(20, 1)
    assert torch.allclose(emax(e), 1.0 - e)
