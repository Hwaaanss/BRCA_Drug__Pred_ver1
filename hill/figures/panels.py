"""Shared plumbing for figure modules: result loading and saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hill.figures.style import apply_rc, watermark_synthetic
from hill.utils.logging import get_logger

log = get_logger("figures")


def save(fig, out_dir: str | Path, name: str, synthetic: bool = False) -> Path:
    """Write PDF (vector, for the manuscript) and PNG (for slides/preview)."""
    watermark_synthetic(fig, synthetic)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_SYNTHETIC" if synthetic else name
    pdf = d / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(d / f"{stem}.png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log.info("wrote %s", pdf)
    return pdf


def load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        log.warning("missing input for figure: %s", p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_table(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        log.warning("missing input for figure: %s", p)
        return None
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def mean_sd(df: pd.DataFrame, group: list[str], col: str) -> pd.DataFrame:
    g = df.groupby(group, dropna=False)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def init() -> None:
    apply_rc()
