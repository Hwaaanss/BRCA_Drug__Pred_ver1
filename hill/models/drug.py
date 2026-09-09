"""Drug encoder.

A drug must be represented by *features*, not by an identity, otherwise the
leave-drugs-out (LDO) split is impossible: an embedding table has no row for a
drug it has never seen.  Default features are Morgan fingerprints plus a few
physicochemical descriptors, precomputed offline (see ``hill/data/drugs.py``).
The one-hot fallback exists only for LPO/LCO debugging and refuses LDO.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrugEncoder(nn.Module):
    """(B, F_drug) drug features -> (B, 1, d) drug token."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.1,
        mode: str = "fingerprint",
    ) -> None:
        super().__init__()
        if mode not in {"fingerprint", "onehot"}:
            raise ValueError(f"drug encoder mode must be fingerprint|onehot, got {mode!r}")
        self.mode = mode
        self.feature_dim = int(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    @property
    def supports_unseen_drugs(self) -> bool:
        return self.mode == "fingerprint"

    def forward(self, drug_features: torch.Tensor) -> torch.Tensor:
        if drug_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"drug feature dim mismatch: expected {self.feature_dim}, got {drug_features.shape[-1]}"
            )
        return self.norm(self.mlp(drug_features)).unsqueeze(1)  # (B, 1, d)
