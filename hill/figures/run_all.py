"""Render every figure the manuscript needs.

    python -m hill.figures.run_all --config configs/base.yaml

Each figure is independent and returns ``None`` (with a warning) when its input
results are not on disk yet, so this can be run at any point in the pipeline.
Outputs go to ``paths.figures_dir`` as both PDF (vector, 400 dpi raster
fallback) and PNG.  Figures built from SYNTHETIC data are watermarked and
suffixed ``_SYNTHETIC``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from hill.config import load_config
from hill.data.synthetic import is_synthetic
from hill.figures import gate0_figs, model_figs, performance_figs
from hill.figures.panels import init
from hill.utils.logging import get_logger

log = get_logger("figures.run_all")

FIGURES: list[tuple[str, Callable[..., Any]]] = [
    ("Fig1_concept", gate0_figs.fig01_concept),
    ("Fig2_gate0_refit", gate0_figs.fig02_refit),
    ("Fig3_censoring", gate0_figs.fig03_censoring),
    ("Fig4_main_performance", performance_figs.fig04_main_performance),
    ("Fig5_ablation_ladder", performance_figs.fig05_ladder),
    ("Fig6_per_drug", performance_figs.fig06_per_drug),
    ("Fig7_predicted_curves", model_figs.fig07_curves),
    ("Fig8_potency_efficacy", model_figs.fig08_decoupling),
    ("Fig9_clinical_and_gates", model_figs.fig09_clinical_gates),
    ("Fig10_hpo", performance_figs.fig10_hpo),
    ("FigS1_variance_decomposition", gate0_figs.figS1_variance),
    ("FigS2_dataflow", model_figs.figS2_dataflow),
]


def render_all(cfg: Any, only: list[str] | None = None) -> dict[str, str | None]:
    init()
    out_dir = Path(cfg.paths.figures_dir)
    synthetic = is_synthetic(cfg.paths.data_root)
    if synthetic:
        log.warning("data root is SYNTHETIC — every figure will be watermarked")
    results: dict[str, str | None] = {}
    for name, fn in FIGURES:
        if only and name not in only:
            continue
        try:
            path = fn(cfg, out_dir, synthetic)
        except Exception as exc:  # noqa: BLE001 - one broken figure must not stop the rest
            log.error("figure %s failed: %s", name, exc, exc_info=True)
            path = None
        results[name] = str(path) if path else None
    made = sum(1 for v in results.values() if v)
    log.info("rendered %d / %d figures into %s", made, len(results), out_dir)
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render all figures")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="figure names to render")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)
    cfg.paths.ensure()
    results = render_all(cfg, args.only)
    for name, path in results.items():
        print(f"{'OK   ' if path else 'SKIP '} {name}{'' if path else '  (missing inputs)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
