"""HILL — the assembled model.

    omics vector ──► group tokens (pathway-wise, independent projections)
                  └► latent tokens (ungrouped features)
    drug features ─► drug token
                     │
                     ▼
        transformer encoder + attention pooling  ──►  z_ij
                     │
        ┌────────────┴─────────────┐
        │  curve head (HILL)       │   raw_e, raw_s, raw_m  ->  E_inf, s, m
        │  scalar head (ScalarHILL)│   ln IC50
        └──────────────────────────┘
                     ▲
        histology (optional)  raw_e += gamma_E * h_E(H_i, z),
                              raw_m += gamma_m * h_m(H_i, z)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from hill.data.tokenize import TokenSpec
from hill.models.curve_head import CurveHead, CurveParams, ScalarHead, constrain_parameters
from hill.models.drug import DrugEncoder
from hill.models.encoder import GroupTokenizer, LatentQueryTokenizer, OmicsDrugEncoder
from hill.models.histology import HistologyBranch, HistologyGate


class HILL(nn.Module):
    """Full model. ``cfg.model.head`` selects HILL (``curve``) or ScalarHILL (``scalar``)."""

    def __init__(
        self,
        cfg: Any,
        token_spec: TokenSpec,
        drug_feature_dim: int,
        n_drugs: int = 0,
    ) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.head_kind = m.head
        self.n_drugs = int(n_drugs)
        d = m.d_model

        self.group_tokenizer = GroupTokenizer(
            gene_index=token_spec.gene_index,
            gene_mask=token_spec.gene_mask,
            modality_id=token_spec.modality_id,
            d_model=d,
            n_modalities=len(token_spec.modality_names),
            dropout=m.token_dropout,
        )
        self.latent_tokenizer = LatentQueryTokenizer(
            feature_index=token_spec.ungrouped_index,
            d_model=d,
            n_latent=token_spec.n_latent,
            n_heads=max(1, m.n_heads // 2),
            dropout=m.dropout,
        )
        self.drug_encoder = DrugEncoder(
            feature_dim=drug_feature_dim,
            d_model=d,
            hidden=m.drug_hidden,
            dropout=m.dropout,
            mode=cfg.data.drug_features,
        )
        self.encoder = OmicsDrugEncoder(
            d_model=d,
            n_layers=m.n_layers,
            n_heads=m.n_heads,
            ffn_mult=m.ffn_mult,
            dropout=m.dropout,
            attn_dropout=m.attn_dropout,
        )

        if self.head_kind == "curve":
            self.head: nn.Module = CurveHead(
                d, dropout=m.head_dropout, init_e_logit=m.init_e_logit, init_slope_raw=m.init_slope_raw
            )
        else:
            self.head = ScalarHead(d, dropout=m.head_dropout)

        # --- histology gates (zero-initialised, frozen in Stage 0) ----------
        h = m.histology
        self.use_histology = bool(h.enabled)
        if self.use_histology:
            self.histology = HistologyBranch(
                feature_dim=h.feature_dim, d_model=d, n_prototypes=h.n_prototypes, dropout=m.dropout
            )
            self.gate_e = HistologyGate(d, h.gate_hidden, m.dropout) if h.gate_efficacy else None
            self.gate_m = HistologyGate(d, h.gate_hidden, m.dropout) if h.gate_potency else None
            self.set_gamma_trainable(h.train_gamma)
        else:
            self.histology = None
            self.gate_e = None
            self.gate_m = None

        n_group = token_spec.n_groups
        n_latent = token_spec.n_latent if token_spec.ungrouped_index.size else 0
        types = torch.cat(
            [
                torch.zeros(n_group, dtype=torch.long),
                torch.ones(n_latent, dtype=torch.long),
                torch.full((1,), 2, dtype=torch.long),
            ]
        )
        self.register_buffer("token_types", types, persistent=False)
        self.n_tokens = int(types.numel())

    # -- gamma control -------------------------------------------------------
    def set_gamma_trainable(self, flag: bool) -> None:
        """Stage 0 keeps gamma frozen at 0 (cell lines have no slides)."""
        for gate in (self.gate_e, self.gate_m):
            if gate is not None:
                gate.set_trainable(flag)

    @torch.no_grad()
    def zero_gammas(self) -> None:
        for gate in (self.gate_e, self.gate_m):
            if gate is not None:
                gate.zero_gamma()

    def gammas(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.gate_e is not None:
            out["gamma_e"] = float(self.gate_e.gamma.detach())
        if self.gate_m is not None:
            out["gamma_m"] = float(self.gate_m.gamma.detach())
        return out

    # -- forward -------------------------------------------------------------
    def encode(
        self,
        omics: torch.Tensor,          # (B, F)
        drug_features: torch.Tensor,  # (B, F_drug)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [self.group_tokenizer(omics)]
        latent = self.latent_tokenizer(omics)
        if latent is not None:
            tokens.append(latent)
        tokens.append(self.drug_encoder(drug_features))
        seq = torch.cat(tokens, dim=1)  # (B, N, d)
        return self.encoder(seq, self.token_types)

    def forward(
        self,
        omics: torch.Tensor,
        drug_features: torch.Tensor,
        histology: torch.Tensor | None = None,
        histo_mask: torch.Tensor | None = None,
        slide_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z, pool_attn = self.encode(omics, drug_features)
        out: dict[str, torch.Tensor] = {"z": z, "pool_attention": pool_attn}

        histo_repr = None
        if self.use_histology and (self.gate_e is not None or self.gate_m is not None):
            histo_repr, histo_attn = self.histology(histology, histo_mask, z, slide_available)
            if histo_attn is not None:
                out["histo_attention"] = histo_attn

        if self.head_kind == "curve":
            raw_e, raw_s, raw_m = self.head.raw(z)
            if histo_repr is not None:
                if self.gate_e is not None:
                    raw_e = raw_e + self.gate_e(histo_repr, z)
                if self.gate_m is not None:
                    raw_m = raw_m + self.gate_m(histo_repr, z)
            e_inf, slope, midpoint = constrain_parameters(raw_e, raw_s, raw_m)
            out.update({"e_inf": e_inf, "slope": slope, "midpoint": midpoint,
                        "raw_e": raw_e, "raw_s": raw_s, "raw_m": raw_m})
        else:
            pred = self.head(z)
            if histo_repr is not None and self.gate_e is not None:
                pred = pred + self.gate_e(histo_repr, z)
            out["ln_ic50_pred"] = pred
        return out

    def curve_params(self, out: dict[str, torch.Tensor]) -> CurveParams:
        return CurveParams(out["e_inf"], out["slope"], out["midpoint"])

    # -- optimiser helper ----------------------------------------------------
    def param_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """No weight decay on norms, biases, embeddings or the gamma scalars."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "gamma" in name or "embed" in name or "queries" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def n_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def build_model(cfg: Any, token_spec: TokenSpec, drug_feature_dim: int, n_drugs: int = 0) -> HILL:
    return HILL(cfg, token_spec, drug_feature_dim=drug_feature_dim, n_drugs=n_drugs)
