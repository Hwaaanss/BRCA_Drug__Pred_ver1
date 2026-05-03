"""Render all v8 figures in one pass."""
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = PROJECT_ROOT / "research" / "figures" / "figures_v8"
os.makedirs(OUT, exist_ok=True)
from . import fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8
for mod in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
    mod.make(OUT)
print('All v8 figures generated →', OUT)
