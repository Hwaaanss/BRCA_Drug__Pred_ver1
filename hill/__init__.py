"""HILL — Hill-curve Inference for Learned dose-response Landscapes.

The project predicts a *dose-response function* per (sample, drug) pair rather
than a scalar summary such as IC50.  A constrained three-parameter Hill head
outputs

    r(c) = E_inf + (1 - E_inf) * sigmoid(-s * (log c - m))

and the model is trained on raw, control-normalised viability measurements.
IC50 / AUC / Emax are *derived* from the predicted curve (see :mod:`hill.derive`).

See ``README.md`` for the full rationale and ``reports/gate0.md`` for the data
validity audit that must pass before any model is trained.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
