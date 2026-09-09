"""The nesting guarantee that makes the gamma likelihood-ratio test valid.

With gamma_E = gamma_m = 0 the histology-gated model must produce *identical*
numbers to the model without histology — not merely close ones.  If this fails,
the two models are not nested and the LRT in guide §7.3 is invalid.
"""

from __future__ import annotations

import torch

from hill.models.hill import build_model


def _models(small_config, token_spec):
    cfg_h = small_config.copy_with(
        ["model.histology.enabled=true", "model.histology.feature_dim=48",
         "model.histology.n_prototypes=4", "model.histology.gate_hidden=16"]
    )
    torch.manual_seed(0)
    gated = build_model(cfg_h, token_spec, drug_feature_dim=6, n_drugs=6)
    gated.eval()
    return gated


def test_gamma_starts_at_exactly_zero(small_config, token_spec):
    gated = _models(small_config, token_spec)
    for name, value in gated.gammas().items():
        assert value == 0.0, f"{name} must be initialised to exactly 0.0, got {value!r}"


def test_gamma_zero_equivalence(small_config, token_spec):
    """Bitwise identical outputs with and without the histology input."""
    gated = _models(small_config, token_spec)
    torch.manual_seed(1)
    omics = torch.randn(6, token_spec.n_features)
    drug = torch.randn(6, 6)
    patches = torch.randn(6, 12, 48)
    mask = torch.ones(6, 12, dtype=torch.bool)

    with torch.no_grad():
        without = gated(omics, drug)
        with_histo = gated(omics, drug, patches, mask, torch.ones(6))

    for key in ("e_inf", "slope", "midpoint", "raw_e", "raw_s", "raw_m", "z"):
        a, b = without[key], with_histo[key]
        assert torch.equal(a, b), f"{key} differs with gamma = 0 (max |d| = {float((a - b).abs().max()):.3e})"
        # bit-level check, normalising the -0.0 / +0.0 pair which compares equal
        ai = (a + 0.0).contiguous().view(torch.int32)
        bi = (b + 0.0).contiguous().view(torch.int32)
        assert torch.equal(ai, bi), f"{key} is not bit-identical with gamma = 0"


def test_gamma_zero_equivalence_with_missing_slides(small_config, token_spec):
    """A patient without a slide must also be unaffected while gamma = 0."""
    gated = _models(small_config, token_spec)
    torch.manual_seed(2)
    omics = torch.randn(4, token_spec.n_features)
    drug = torch.randn(4, 6)
    patches = torch.randn(4, 8, 48)
    mask = torch.ones(4, 8, dtype=torch.bool)
    with torch.no_grad():
        base = gated(omics, drug)
        partial = gated(omics, drug, patches, mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert torch.equal(base["e_inf"], partial["e_inf"])
    assert torch.equal(base["midpoint"], partial["midpoint"])


def test_nonzero_gamma_does_change_output(small_config, token_spec):
    """Sanity: the gate is wired up, so a non-zero gamma must move the prediction."""
    gated = _models(small_config, token_spec)
    with torch.no_grad():
        gated.gate_e.gamma.fill_(0.5)
        for p in gated.gate_e.mlp.parameters():
            p.add_(0.1)
    torch.manual_seed(3)
    omics = torch.randn(4, token_spec.n_features)
    drug = torch.randn(4, 6)
    patches = torch.randn(4, 8, 48)
    mask = torch.ones(4, 8, dtype=torch.bool)
    with torch.no_grad():
        base = gated(omics, drug)
        gatedout = gated(omics, drug, patches, mask, torch.ones(4))
    assert not torch.equal(base["e_inf"], gatedout["e_inf"])
    assert torch.equal(base["midpoint"], gatedout["midpoint"]), "gamma_m was still 0; m must not move"
