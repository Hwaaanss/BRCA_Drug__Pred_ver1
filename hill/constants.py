"""Project-wide constants.

Everything that is a *modelling choice* belongs in a config file
(``configs/*.yaml``).  This module holds only immutable facts about the data
sources and hard numerical guards.
"""

from __future__ import annotations

from typing import Final

# --- numerical guards -------------------------------------------------------
SIGMOID_CLAMP: Final[float] = 30.0   # |arg| clamp for the Hill sigmoid
SLOPE_EPS: Final[float] = 1e-4       # s = softplus(raw_s) + SLOPE_EPS  > 0
VIABILITY_EPS: Final[float] = 1e-4   # Beta-likelihood clipping bound
# E_inf is kept strictly inside (0, 1): float32 sigmoid saturates to exactly 1.0
# for |raw_e| > ~17, which would make (1 - E_inf) exactly zero and collapse the
# curve to a constant with no gradient path back to the encoder.
E_INF_EPS: Final[float] = 1e-6
LOG_2PI: Final[float] = 1.8378770664093453

# --- GDSC raw plate layout --------------------------------------------------
# One row of the raw file is one well of a screening plate.
RAW_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "BARCODE",
    "SCAN_ID",
    "COSMIC_ID",
    "DRUG_ID",
    "CONC",
    "TAG",
    "INTENSITY",
)
# Some GDSC releases name the readout FLUORESCENCE instead of INTENSITY.
RAW_INTENSITY_ALIASES: Final[tuple[str, ...]] = ("INTENSITY", "FLUORESCENCE")

# Control tags, following gdscIC50::normalizeData defaults.
#   negative control = untreated / DMSO wells  -> viability 1
#   positive control = blank wells (no cells)  -> viability 0
DEFAULT_NEG_CONTROL_TAGS: Final[tuple[str, ...]] = ("NC-1", "NC-0")
DEFAULT_POS_CONTROL_TAGS: Final[tuple[str, ...]] = ("B",)
# Tags that are never treatment wells.
NON_TREATMENT_TAGS: Final[frozenset[str]] = frozenset(
    {"B", "NC-0", "NC-1", "DMSO", "SC", "UN-USED", "PC-1", "PC1-D1-S", "FAIL", "EMPTY"}
)

# --- splits -----------------------------------------------------------------
SPLIT_NAMES: Final[tuple[str, ...]] = ("LPO", "LCO", "LDO", "XDOM")
PRIMARY_SPLIT: Final[str] = "LCO"

# --- evaluation -------------------------------------------------------------
# Global (drug-pooled) Pearson correlation is *forbidden* -- see reports and
# hill/evaluate.py::GlobalPCCForbiddenError.
FORBIDDEN_METRICS: Final[frozenset[str]] = frozenset({"global_pcc", "overall_pcc", "pooled_pcc"})

