"""Training objectives.

Source domain (GDSC)
    L_source = sum_ijk  NLL( v_ijk ; r_ij(c_k), sigma_j )
    No loss is ever placed on IC50.  That is precisely where the censoring
    problem disappears: we never regress onto a censored quantity, so no Tobit
    likelihood is needed and pairs whose curve never crosses 50% still
    contribute full information through their observed points.

Target domain (TCGA)
    p(responder | i, j) = sigmoid( w * (1 - r_ij(Cmax_j)) + b )
    L_target = BCE
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hill.constants import LOG_2PI, VIABILITY_EPS


class DrugScale(nn.Module):
    """Per-drug observation scale, either an embedding or a map of drug features.

    ``from_features=True`` is required for LDO (leave-drugs-out) to be valid:
    an embedding table has no entry for an unseen drug.
    """

    def __init__(
        self,
        n_drugs: int,
        drug_feature_dim: int | None,
        from_features: bool,
        init_value: float,
        min_value: float,
        max_value: float,
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.from_features = bool(from_features and drug_feature_dim)
        if self.from_features:
            self.proj = nn.Linear(int(drug_feature_dim), 1)
            nn.init.zeros_(self.proj.weight)
            nn.init.constant_(self.proj.bias, init_value)
            self.table = None
        else:
            self.table = nn.Embedding(max(n_drugs, 1), 1)
            nn.init.constant_(self.table.weight, init_value)
            self.proj = None

    def forward(
        self, drug_index: torch.Tensor | None, drug_features: torch.Tensor | None
    ) -> torch.Tensor:
        """-> (B, 1) raw (unclamped-then-clamped) log scale."""
        if self.from_features:
            if drug_features is None:
                raise ValueError("DrugScale configured from drug features but none were given")
            raw = self.proj(drug_features)
        else:
            if drug_index is None:
                raise ValueError("DrugScale configured as an embedding but no drug index was given")
            raw = self.table(drug_index.long()).squeeze(-1).unsqueeze(-1)
        return raw.clamp(self.min_value, self.max_value)


class ViabilityLikelihood(nn.Module):
    """Negative log-likelihood of observed viability under the predicted curve.

    ``kind='gaussian'``: heteroscedastic Gaussian with a learned per-drug sigma.
    ``kind='beta'``:     Beta likelihood with a learned per-drug precision; the
                         observations must be clipped into (0, 1) and the clip
                         rate is reported so it can be logged (guide §6.1).
    """

    def __init__(
        self,
        cfg: Any,
        n_drugs: int,
        drug_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.kind = cfg.kind
        self.heteroscedastic = cfg.heteroscedastic
        init = cfg.init_log_sigma if self.kind == "gaussian" else cfg.beta_init_log_phi
        lo, hi = (cfg.min_log_sigma, cfg.max_log_sigma) if self.kind == "gaussian" else (-4.0, 8.0)
        if self.heteroscedastic:
            self.scale = DrugScale(n_drugs, drug_feature_dim, cfg.sigma_from_drug_features, init, lo, hi)
            self.global_scale = None
        else:
            self.scale = None
            self.global_scale = nn.Parameter(torch.tensor(float(init)))
        self.register_buffer("_clip_count", torch.zeros((), dtype=torch.double), persistent=False)
        self.register_buffer("_total_count", torch.zeros((), dtype=torch.double), persistent=False)

    # -- scale ---------------------------------------------------------------
    def log_scale(
        self,
        batch_size: int,
        device: torch.device,
        drug_index: torch.Tensor | None,
        drug_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.heteroscedastic:
            return self.scale(drug_index, drug_features)  # (B, 1)
        return self.global_scale.expand(batch_size, 1).to(device)

    # -- likelihood ----------------------------------------------------------
    def pointwise_loglik(
        self,
        pred: torch.Tensor,        # (B, K) predicted viability
        target: torch.Tensor,      # (B, K) observed viability
        mask: torch.Tensor,        # (B, K) True where a measurement exists
        log_scale: torch.Tensor,   # (B, 1)
    ) -> torch.Tensor:
        """Per-observation log-likelihood, zero where masked out. Shape (B, K)."""
        if self.kind == "gaussian":
            log_sigma = log_scale
            inv_var = torch.exp(-2.0 * log_sigma)
            ll = -0.5 * ((target - pred) ** 2 * inv_var + 2.0 * log_sigma + LOG_2PI)
        else:
            eps = VIABILITY_EPS
            mu = pred.clamp(eps, 1.0 - eps)
            y = target.clamp(eps, 1.0 - eps)
            if self.training:
                with torch.no_grad():
                    clipped = ((target < eps) | (target > 1.0 - eps)) & mask
                    self._clip_count += clipped.sum().double()
                    self._total_count += mask.sum().double()
            phi = F.softplus(log_scale) + 1.0
            a = mu * phi
            b = (1.0 - mu) * phi
            ll = (
                torch.lgamma(phi)
                - torch.lgamma(a)
                - torch.lgamma(b)
                + (a - 1.0) * torch.log(y)
                + (b - 1.0) * torch.log1p(-y)
            )
        return torch.where(mask, ll, torch.zeros_like(ll))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        drug_index: torch.Tensor | None = None,
        drug_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Mean NLL per observed point, plus diagnostics."""
        log_scale = self.log_scale(pred.shape[0], pred.device, drug_index, drug_features)
        ll = self.pointwise_loglik(pred, target, mask, log_scale)
        n_obs = mask.sum().clamp_min(1)
        nll = -ll.sum() / n_obs
        with torch.no_grad():
            se = ((pred - target) ** 2 * mask).sum() / n_obs
        return nll, {
            "nll": nll.detach(),
            "viability_mse": se.detach(),
            "n_points": n_obs.detach(),
            "mean_log_scale": log_scale.mean().detach(),
        }

    @property
    def clip_rate(self) -> float:
        """Fraction of observations clipped into (0, 1) for the Beta likelihood."""
        total = float(self._total_count)
        return float(self._clip_count) / total if total > 0 else 0.0

    def reset_clip_counter(self) -> None:
        self._clip_count.zero_()
        self._total_count.zero_()


class ScalarRegressionLoss(nn.Module):
    """ScalarHILL objective: MSE (or Huber) on ln IC50 — the field's status quo."""

    def __init__(self, kind: str = "mse", huber_delta: float = 1.0) -> None:
        super().__init__()
        self.kind = kind
        self.huber_delta = huber_delta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        if mask is None:
            mask = torch.isfinite(target)
        mask = mask.reshape(-1) & torch.isfinite(target)
        n = mask.sum().clamp_min(1)
        diff = torch.where(mask, pred - torch.nan_to_num(target), torch.zeros_like(pred))
        if self.kind == "huber":
            d = self.huber_delta
            absd = diff.abs()
            per = torch.where(absd <= d, 0.5 * diff**2, d * (absd - 0.5 * d))
        else:
            per = 0.5 * diff**2
        loss = per.sum() / n
        return loss, {"scalar_loss": loss.detach(), "n_pairs": n.detach()}


class ClinicalResponseLoss(nn.Module):
    """TCGA objective: predicted kill at the clinical Cmax drives response odds.

        p(responder) = sigmoid( w * (1 - r_ij(Cmax_j)) + b )

    ``w`` is initialised positive so that "more killing at Cmax" starts out
    meaning "more likely to respond"; it remains free to be estimated.
    """

    def __init__(self, init_w: float = 1.0, init_b: float = 0.0) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def logits(self, viability_at_cmax: torch.Tensor) -> torch.Tensor:
        """(B,) logits from (B,) predicted viability at Cmax."""
        return self.w * (1.0 - viability_at_cmax.reshape(-1)) + self.b

    def forward(
        self,
        viability_at_cmax: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.logits(viability_at_cmax)
        y = response.reshape(-1).float()
        if mask is None:
            mask = torch.isfinite(y)
        mask = mask.reshape(-1)
        n = mask.sum().clamp_min(1)
        per = F.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(y), reduction="none", pos_weight=pos_weight
        )
        loss = (per * mask).sum() / n
        return loss, {"clinical_bce": loss.detach(), "n_clinical": n.detach()}


def build_viability_loss(cfg: Any, n_drugs: int, drug_feature_dim: int | None) -> ViabilityLikelihood:
    return ViabilityLikelihood(cfg.loss, n_drugs=n_drugs, drug_feature_dim=drug_feature_dim)
