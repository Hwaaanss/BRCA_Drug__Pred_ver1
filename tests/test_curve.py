"""The structural guarantees of the Hill head (guide §5.1)."""

from __future__ import annotations

import torch

from hill.models.curve_head import CurveHead, constrain_parameters, hill_curve


def test_curve_monotone_decreasing(curve_params):
    """s > 0 must make r(c) monotone decreasing for every admissible parameter."""
    e, s, m = curve_params
    grid = torch.linspace(-12, 12, 128).expand(e.shape[0], 128)
    r = hill_curve(grid, e, s, m)
    diffs = r[:, 1:] - r[:, :-1]
    assert bool((diffs <= 1e-6).all()), f"max increase {float(diffs.max()):.3e}"


def test_curve_limits(curve_params):
    """c -> 0 gives full survival; c -> inf gives the efficacy ceiling E_inf."""
    e, s, m = curve_params
    at_zero = hill_curve(torch.full_like(e, -1e4), e, s, m)
    at_inf = hill_curve(torch.full_like(e, 1e4), e, s, m)
    assert torch.allclose(at_zero, torch.ones_like(at_zero), atol=1e-6)
    assert torch.allclose(at_inf, e, atol=1e-6)


def test_curve_bounded_between_einf_and_one(curve_params):
    e, s, m = curve_params
    grid = torch.linspace(-8, 8, 32).expand(e.shape[0], 32)
    r = hill_curve(grid, e, s, m)
    assert bool((r <= 1.0 + 1e-6).all())
    assert bool((r >= e - 1e-6).all())


def test_constrained_parameter_ranges():
    raw = torch.randn(500, 1) * 10
    e, s, m = constrain_parameters(raw, raw, raw)
    assert bool(((e > 0) & (e < 1)).all())
    assert bool((s > 0).all())
    assert torch.equal(m, raw)


def test_curve_head_outputs_are_admissible():
    head = CurveHead(16, dropout=0.0)
    head.eval()
    params = head(torch.randn(8, 16))
    assert params.e_inf.shape == (8, 1)
    assert bool(((params.e_inf > 0) & (params.e_inf < 1)).all())
    assert bool((params.slope > 0).all())


def test_curve_head_distinguishes_inputs():
    """A zero-weight head would make every pair identical and kill the gradient."""
    head = CurveHead(16, dropout=0.0)
    head.eval()
    with torch.no_grad():
        p = head(torch.randn(32, 16))
    assert float(p.e_inf.std()) > 0
    assert float(p.midpoint.std()) > 0
