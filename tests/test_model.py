"""End-to-end model behaviour on a tiny synthetic batch."""

from __future__ import annotations

import pytest
import torch

from hill.losses import ViabilityLikelihood
from hill.models.drug import DrugEncoder
from hill.models.hill import build_model


def _batch(spec, n: int = 8, drug_dim: int = 6):
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(n, spec.n_features, generator=g),
        torch.randn(n, drug_dim, generator=g),
    )


def test_forward_shapes_curve_head(small_config, token_spec):
    model = build_model(small_config, token_spec, drug_feature_dim=6, n_drugs=6)
    omics, drug = _batch(token_spec)
    out = model(omics, drug)
    assert out["e_inf"].shape == (8, 1)
    assert out["pool_attention"].shape[1] == model.n_tokens
    assert model.n_tokens == token_spec.n_tokens + 1, "one drug token is appended"


def test_forward_shapes_scalar_head(small_config, token_spec):
    cfg = small_config.copy_with(["model.head=scalar"])
    model = build_model(cfg, token_spec, drug_feature_dim=6, n_drugs=6)
    omics, drug = _batch(token_spec)
    out = model(omics, drug)
    assert out["ln_ic50_pred"].shape == (8, 1)
    assert "e_inf" not in out


def test_gradients_flow_to_every_trainable_part(small_config, token_spec):
    model = build_model(small_config, token_spec, drug_feature_dim=6, n_drugs=6)
    crit = ViabilityLikelihood(small_config.loss, n_drugs=6, drug_feature_dim=6)
    omics, drug = _batch(token_spec)
    out = model(omics, drug)
    log_conc = torch.linspace(-3, 3, 5).expand(8, 5)
    viability = torch.rand(8, 5)
    mask = torch.ones(8, 5, dtype=torch.bool)
    pred = model.curve_params(out).viability(log_conc)
    loss, _ = crit(pred, viability, mask, drug_features=drug)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_training_step_reduces_the_loss(small_config, token_spec):
    """A handful of steps on a fixed batch must lower the objective."""
    torch.manual_seed(0)
    model = build_model(small_config, token_spec, drug_feature_dim=6, n_drugs=6)
    crit = ViabilityLikelihood(small_config.loss, n_drugs=6, drug_feature_dim=6)
    omics, drug = _batch(token_spec)
    log_conc = torch.linspace(-3, 3, 5).expand(8, 5)
    e, s, m = 0.3, 1.5, 0.5
    viability = e + (1 - e) / (1 + torch.exp(s * (log_conc - m)))
    mask = torch.ones(8, 5, dtype=torch.bool)
    opt = torch.optim.AdamW(list(model.parameters()) + list(crit.parameters()), lr=1e-2)

    losses = []
    for _ in range(25):
        opt.zero_grad()
        pred = model.curve_params(model(omics, drug)).viability(log_conc)
        loss, _ = crit(pred, viability, mask, drug_features=drug)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_parameter_groups_exclude_norms_from_weight_decay(small_config, token_spec):
    model = build_model(small_config, token_spec, drug_feature_dim=6, n_drugs=6)
    groups = model.param_groups(0.01)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.01 and groups[1]["weight_decay"] == 0.0
    total = sum(len(g["params"]) for g in groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_drug_encoder_rejects_wrong_dimension():
    enc = DrugEncoder(feature_dim=8, d_model=16)
    with pytest.raises(ValueError, match="drug feature dim mismatch"):
        enc(torch.randn(2, 9))


def test_onehot_drug_features_cannot_generalise_to_new_drugs():
    """LDO requires a feature-based encoder; the one-hot fallback says so."""
    assert DrugEncoder(4, 8, mode="fingerprint").supports_unseen_drugs
    assert not DrugEncoder(4, 8, mode="onehot").supports_unseen_drugs


def test_eval_mode_is_deterministic(small_config, token_spec):
    model = build_model(small_config, token_spec, drug_feature_dim=6, n_drugs=6)
    model.eval()
    omics, drug = _batch(token_spec)
    with torch.no_grad():
        a = model(omics, drug)["e_inf"]
        b = model(omics, drug)["e_inf"]
    assert torch.equal(a, b)
