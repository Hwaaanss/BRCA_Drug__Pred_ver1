"""Shared loading + reporting helpers for the Gate-0 audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hill.config import Config
from hill.utils.logging import get_logger

log = get_logger("audit")


def audit_dir(cfg: Config) -> Path:
    d = Path(cfg.paths.results_dir) / "gate0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_synthetic(cfg: Config) -> bool:
    from hill.data.synthetic import is_synthetic as _is

    return _is(cfg.paths.data_root)


def load_tables(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Return (points, pairs, padded arrays) produced by ``hill.data.prepare``."""
    proc = Path(cfg.paths.processed_dir)
    missing = [f for f in ("gdsc_points.parquet", "gdsc_pairs.parquet", "padded.npz")
               if not (proc / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found in {proc}. Run `python -m hill.data.prepare --config <cfg>` first."
        )
    points = pd.read_parquet(proc / "gdsc_points.parquet")
    pairs = pd.read_parquet(proc / "gdsc_pairs.parquet")
    with np.load(proc / "padded.npz", allow_pickle=False) as z:
        padded = {k: z[k] for k in z.files}
    return points, pairs, padded


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    log.info("wrote %s", p)


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def summarise(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0.0}
    return {
        "n": float(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q05": float(np.percentile(v, 5)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "q95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }
