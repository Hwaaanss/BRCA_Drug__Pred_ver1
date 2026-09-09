"""Likelihoods: shapes, masking, gradients, and the no-IC50-loss guarantee."""

from __future__ import annotations

import torch

from hill.config import Config
from hill.losses import ClinicalResponseLoss, ScalarRegressionLoss, ViabilityLikelihood


def _loss(kind: str, hetero: bool = True, from_features: bool = True) -> ViabilityLikelihood:
    cfg = Config().copy_with(
        [f"loss.kind={kind}", f"loss.heteroscedastic={str(hetero).lower()}",
         f"loss.sigma_from_drug_features={str(from_features).lower()}"]
    )
    return ViabilityLikelihood(cfg.loss, n_drugs=4, drug_feature_dim=6)


def test_masked_points_do_not_contribute():
    crit = _loss("gaussian")
    pred = torch.rand(3, 5)
    target = torch.rand(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[:, 3:] = False
    feats = torch.randn(3, 6)
    a, _ = crit(pred, target, mask, drug_features=feats)
    target_changed = target.clone()
    target_changed[:, 3:] = 99.0  # only masked entries change
    b, _ = crit(pred, target_changed, mask, drug_features=feats)
    assert torch.allclose(a, b)


def test_gaussian_nll_decreases_when_prediction_improves():
    crit = _loss("gaussian")
    target = torch.full((4, 6), 0.4)
    mask = torch.ones(4, 6, dtype=torch.bool)
    feats = torch.randn(4, 6)
    with torch.no_grad():
        good, _ = crit(torch.full((4, 6), 0.41), target, mask, drug_features=feats)
        bad, _ = crit(torch.full((4, 6), 0.90), target, mask, drug_features=feats)
    assert float(good) < float(bad)


def test_beta_likelihood_tracks_clipping():
    crit = _loss("beta")
    crit.train()
    target = torch.tensor([[0.0, 1.0, 0.5, 0.5]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    crit(torch.full((1, 4), 0.5), target, mask, drug_features=torch.randn(1, 6))
    assert crit.clip_rate == 0.5, "two of four observations sit on the boundary"


def test_heteroscedastic_scale_is_learnable_and_bounded():
    crit = _loss("gaussian")
    loss, parts = crit(torch.rand(2, 3), torch.rand(2, 3), torch.ones(2, 3, dtype=torch.bool),
                       drug_features=torch.randn(2, 6))
    loss.backward()
    grads = [p.grad for p in crit.parameters() if p.grad is not None]
    assert grads, "the per-drug scale must receive a gradient"
    assert torch.isfinite(parts["mean_log_scale"])


def test_embedding_scale_requires_a_drug_index():
    crit = _loss("gaussian", from_features=False)
    out, _ = crit(torch.rand(2, 3), torch.rand(2, 3), torch.ones(2, 3, dtype=torch.bool),
                  drug_index=torch.tensor([0, 1]))
    assert torch.isfinite(out)


def test_scalar_loss_ignores_missing_targets():
    crit = ScalarRegressionLoss("mse")
    pred = torch.zeros(4, 1)
    target = torch.tensor([1.0, float("nan"), 1.0, float("nan")])
    loss, parts = crit(pred, target)
    assert float(parts["n_pairs"]) == 2
    assert abs(float(loss) - 0.5) < 1e-6


def test_clinical_head_is_monotone_in_predicted_kill():
    head = ClinicalResponseLoss(init_w=1.0, init_b=0.0)
    with torch.no_grad():
        strong_kill = head.logits(torch.tensor([0.1]))   # low viability at Cmax
        weak_kill = head.logits(torch.tensor([0.9]))
    assert float(strong_kill) > float(weak_kill)
