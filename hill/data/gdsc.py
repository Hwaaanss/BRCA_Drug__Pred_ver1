"""GDSC raw viability ingestion.

One row of a GDSC raw data file is one *well* of a screening plate:

    PROJECT_ID, BARCODE, SCAN_ID, SEEDING_DATE, MEASUREMENT_DATE, CELL_ID,
    MASTER_CELL_ID, COSMIC_ID, CELL_NAME, CELLS_PLATED, DRUGSET_ID, ASSAY,
    ASSAY_DURATION, POSITION, TAG, DRUG_ID, CONC, INTENSITY

Normalisation follows ``gdscIC50::normalizeData``:

    viability = (intensity - mu_pos) / (mu_neg - mu_pos)      # per plate

with ``mu_neg`` from the untreated/DMSO negative-control wells (viability 1) and
``mu_pos`` from the blank wells (viability 0).  Values are optionally trimmed to
[0, 1].  The result is the point-level table the model is trained on — no curve
fitting happens here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from hill.constants import (
    DEFAULT_NEG_CONTROL_TAGS,
    DEFAULT_POS_CONTROL_TAGS,
    NON_TREATMENT_TAGS,
    RAW_INTENSITY_ALIASES,
)
from hill.utils.logging import get_logger

log = get_logger("data.gdsc")

_DOSE_RE = re.compile(r"D(\d+)")
_USECOLS = ["BARCODE", "SCAN_ID", "COSMIC_ID", "DRUG_ID", "CONC", "TAG"]


@dataclass
class IngestQC:
    """Everything a reviewer needs to know about how the raw table was reduced."""

    n_wells_read: int = 0
    n_plates: int = 0
    n_plates_failed_qc: int = 0
    n_control_wells: int = 0
    n_treatment_wells: int = 0
    n_combination_wells_dropped: int = 0
    n_wells_nonpositive_conc: int = 0
    n_points_before_pair_filter: int = 0
    n_points: int = 0
    n_pairs: int = 0
    n_pairs_dropped_too_few_points: int = 0
    trimmed_low_fraction: float = 0.0
    trimmed_high_fraction: float = 0.0
    points_per_pair: dict[str, float] = field(default_factory=dict)
    tag_counts: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _find_intensity_column(columns: Iterable[str]) -> str:
    cols = {c.upper(): c for c in columns}
    for alias in RAW_INTENSITY_ALIASES:
        if alias in cols:
            return cols[alias]
    raise KeyError(
        f"no intensity column among {RAW_INTENSITY_ALIASES}; got columns {sorted(cols)}"
    )


def _dose_level(tag: pd.Series) -> pd.Series:
    """Extract the dose index from a TAG such as ``L12-D3-S`` -> 3."""
    return tag.str.extract(_DOSE_RE, expand=False).astype("float32")


def _is_combination(tag: pd.Series) -> pd.Series:
    """Combination wells carry two library positions; single-agent wells one."""
    return tag.str.contains(r"\+", regex=True, na=False) | (
        tag.str.count(r"D\d+") > 1
    )


def read_raw_wells(path: str | Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Read a GDSC raw well-level file, keeping only the columns we need."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"GDSC raw data not found at {p}. Download it with "
            "`python -m hill.data.download --what gdsc-raw` (see reports/DATA_NOTES.md)."
        )
    header = pd.read_csv(p, nrows=0)
    intensity_col = _find_intensity_column(header.columns)
    usecols = [c for c in _USECOLS if c in header.columns] + [intensity_col]
    missing = set(_USECOLS) - set(usecols)
    if missing:
        raise KeyError(f"{p} is missing required columns: {sorted(missing)}")
    chunks = []
    for chunk in pd.read_csv(
        p,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"BARCODE": "str", "SCAN_ID": "str", "TAG": "str"},
        low_memory=False,
    ):
        chunk = chunk.rename(columns={intensity_col: "INTENSITY"})
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    log.info("read %s: %d wells", p.name, len(df))
    return df


def normalize_wells(
    wells: pd.DataFrame,
    neg_control_tags: Sequence[str] = DEFAULT_NEG_CONTROL_TAGS,
    pos_control_tags: Sequence[str] = DEFAULT_POS_CONTROL_TAGS,
    trim: bool = True,
    qc: IngestQC | None = None,
) -> pd.DataFrame:
    """Plate-wise control normalisation. Returns treatment wells with ``viability``."""
    qc = qc or IngestQC()
    df = wells.copy()
    df["TAG"] = df["TAG"].astype("string").fillna("")
    qc.n_wells_read += len(df)
    qc.tag_counts = (
        df["TAG"].str.replace(r"L\d+-", "", regex=True).value_counts().head(30).to_dict()
    )

    plate_key = ["BARCODE", "SCAN_ID"]
    is_neg = df["TAG"].isin(list(neg_control_tags))
    is_pos = df["TAG"].isin(list(pos_control_tags))
    qc.n_control_wells += int((is_neg | is_pos).sum())

    mu_neg = df[is_neg].groupby(plate_key, dropna=False)["INTENSITY"].mean().rename("mu_neg")
    mu_pos = df[is_pos].groupby(plate_key, dropna=False)["INTENSITY"].mean().rename("mu_pos")
    plates = pd.concat([mu_neg, mu_pos], axis=1)
    if plates.empty:
        raise ValueError(
            "no control wells found — check neg/pos control TAGs "
            f"({list(neg_control_tags)} / {list(pos_control_tags)}) against the file's TAG values"
        )
    # A blank-only plate has no positive control: fall back to 0 intensity.
    plates["mu_pos"] = plates["mu_pos"].fillna(0.0)
    qc.n_plates += len(plates)
    good = plates["mu_neg"] > plates["mu_pos"]
    qc.n_plates_failed_qc += int((~good).sum())
    plates = plates[good]

    treat = df[~df["TAG"].isin(NON_TREATMENT_TAGS) & df["DRUG_ID"].notna()].copy()
    combo = _is_combination(treat["TAG"])
    qc.n_combination_wells_dropped += int(combo.sum())
    treat = treat[~combo]
    treat["dose_level"] = _dose_level(treat["TAG"])

    nonpos = ~(treat["CONC"] > 0)
    qc.n_wells_nonpositive_conc += int(nonpos.sum())
    treat = treat[~nonpos]

    treat = treat.merge(plates, left_on=plate_key, right_index=True, how="inner")
    denom = treat["mu_neg"] - treat["mu_pos"]
    treat["viability"] = (treat["INTENSITY"] - treat["mu_pos"]) / denom
    qc.n_treatment_wells += len(treat)

    if trim:
        low = float((treat["viability"] < 0).mean()) if len(treat) else 0.0
        high = float((treat["viability"] > 1).mean()) if len(treat) else 0.0
        qc.trimmed_low_fraction = low
        qc.trimmed_high_fraction = high
        treat["viability"] = treat["viability"].clip(0.0, 1.0)

    return treat[["COSMIC_ID", "DRUG_ID", "CONC", "dose_level", "viability"]]


def build_points_table(
    raw_files: dict[str, str | Path],
    cfg: Any,
    qc: IngestQC | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, IngestQC]:
    """Normalise every raw file and reduce to point- and pair-level tables.

    Returns ``(points, pairs, qc)`` where ``points`` has one row per
    (pair, concentration) and ``pairs`` one row per (cell line, drug).
    """
    qc = qc or IngestQC()
    frames = []
    for source, path in raw_files.items():
        wells = read_raw_wells(path)
        norm = normalize_wells(
            wells,
            neg_control_tags=cfg.data.neg_control_tags,
            pos_control_tags=cfg.data.pos_control_tags,
            trim=cfg.data.trim_viability,
            qc=qc,
        )
        norm["source"] = source
        qc.sources[source] = len(norm)
        frames.append(norm)
        del wells
    if not frames:
        raise ValueError("no raw GDSC files were provided")

    points = pd.concat(frames, ignore_index=True)
    points = points[points["viability"].notna()]
    points["cell_id"] = points["COSMIC_ID"].astype("Int64").astype("string")
    points["drug_id"] = points["DRUG_ID"].astype("int64")
    points = points.rename(columns={"CONC": "conc"})
    points = points[points["conc"] > cfg.data.min_conc_um]

    if cfg.data.aggregate_replicates:
        grouped = points.groupby(["source", "cell_id", "drug_id", "conc"], observed=True)
        points = grouped.agg(
            viability=("viability", "mean"),
            n_replicates=("viability", "size"),
            dose_level=("dose_level", "min"),
        ).reset_index()
    else:
        points["n_replicates"] = 1

    qc.n_points_before_pair_filter = len(points)

    # GDSC2 supersedes GDSC1 wherever both screened the same pair.
    if points["source"].nunique() > 1:
        priority = {src: i for i, src in enumerate(sorted(points["source"].unique(), reverse=True))}
        points["_prio"] = points["source"].map(priority)
        winner = (
            points.groupby(["cell_id", "drug_id"], observed=True)["_prio"].min().rename("_win")
        )
        points = points.merge(winner, on=["cell_id", "drug_id"], how="left")
        points = points[points["_prio"] == points["_win"]].drop(columns=["_prio", "_win"])

    pair_stats = points.groupby(["cell_id", "drug_id"], observed=True).agg(
        n_points=("conc", "size"),
        min_conc=("conc", "min"),
        max_conc=("conc", "max"),
        source=("source", "first"),
    ).reset_index()

    too_few = pair_stats["n_points"] < cfg.data.min_points_per_pair
    qc.n_pairs_dropped_too_few_points = int(too_few.sum())
    pair_stats = pair_stats[~too_few].reset_index(drop=True)
    pair_stats["pair_id"] = np.arange(len(pair_stats), dtype=np.int64)

    points = points.merge(
        pair_stats[["cell_id", "drug_id", "pair_id", "max_conc", "n_points"]],
        on=["cell_id", "drug_id"],
        how="inner",
    )
    points["log_conc"] = np.log(points["conc"].to_numpy(dtype=np.float64))
    points = points.sort_values(["pair_id", "conc"]).reset_index(drop=True)

    qc.n_points = len(points)
    qc.n_pairs = len(pair_stats)
    ppp = pair_stats["n_points"]
    if len(ppp):
        qc.points_per_pair = {
            "mean": float(ppp.mean()),
            "median": float(ppp.median()),
            "min": float(ppp.min()),
            "max": float(ppp.max()),
            **{f"q{q}": float(ppp.quantile(q / 100)) for q in (5, 25, 75, 95)},
            **{f"n_with_{k}_points": int((ppp == k).sum()) for k in sorted(ppp.unique())[:20]},
        }
    else:
        qc.points_per_pair = {"mean": float("nan")}
    log.info(
        "GDSC points: %d measurements over %d (cell, drug) pairs; %.2f points/pair",
        qc.n_points, qc.n_pairs, qc.points_per_pair.get("mean", float("nan")),
    )
    return points, pair_stats, qc


# ---------------------------------------------------------------------------
# Published fitted curves (for comparison only — never a training target)
# ---------------------------------------------------------------------------

_FITTED_RENAME = {
    "COSMIC_ID": "cell_id",
    "DRUG_ID": "drug_id",
    "LN_IC50": "ln_ic50_published",
    "AUC": "auc_published",
    "RMSE": "rmse_published",
    "Z_SCORE": "z_score_published",
    "MAX_CONC": "max_conc_published",
    "DATASET": "dataset",
    "DRUG_NAME": "drug_name",
    "CELL_LINE_NAME": "cell_line_name",
    "TCGA_DESC": "tcga_desc",
    "PUTATIVE_TARGET": "putative_target",
    "PATHWAY_NAME": "pathway_name",
}


def load_fitted(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Read GDSC fitted dose-response files (xlsx or csv) into one frame."""
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            log.warning("fitted file missing: %s", p)
            continue
        df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
        df.columns = [c.strip().upper() for c in df.columns]
        keep = {k: v for k, v in _FITTED_RENAME.items() if k in df.columns}
        df = df[list(keep)].rename(columns=keep)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "no GDSC fitted dose-response file found; run "
            "`python -m hill.data.download --what gdsc-fitted`"
        )
    out = pd.concat(frames, ignore_index=True)
    out["cell_id"] = out["cell_id"].astype("Int64").astype("string")
    out["drug_id"] = out["drug_id"].astype("int64")
    return out


def attach_published(pairs: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """Join published LN_IC50 / AUC onto the pair table and flag censoring.

    ``censored`` means the published IC50 lies beyond the maximum tested
    concentration, i.e. it was extrapolated by the two-parameter fit.
    """
    cols = [c for c in fitted.columns if c not in {"dataset"}]
    merged = pairs.merge(
        fitted[cols].drop_duplicates(["cell_id", "drug_id"]),
        on=["cell_id", "drug_id"],
        how="left",
        suffixes=("", "_pub"),
    )
    max_conc = merged["max_conc_published"].fillna(merged["max_conc"])
    with np.errstate(invalid="ignore", divide="ignore"):
        merged["censored"] = merged["ln_ic50_published"] > np.log(max_conc.to_numpy(dtype=float))
    merged["censored"] = merged["censored"].fillna(False)
    return merged


def points_to_padded(
    points: pd.DataFrame, pairs: pd.DataFrame, max_points: int
) -> dict[str, np.ndarray]:
    """Pack the point table into dense (n_pairs, K) arrays for fast batching."""
    pair_order = {pid: i for i, pid in enumerate(pairs["pair_id"].to_numpy())}
    n_pairs = len(pair_order)
    log_conc = np.zeros((n_pairs, max_points), dtype=np.float32)
    viability = np.zeros((n_pairs, max_points), dtype=np.float32)
    mask = np.zeros((n_pairs, max_points), dtype=bool)

    pid = points["pair_id"].to_numpy()
    row = np.array([pair_order.get(p, -1) for p in pid])
    keep = row >= 0
    row = row[keep]
    lc = points["log_conc"].to_numpy(dtype=np.float32)[keep]
    vb = points["viability"].to_numpy(dtype=np.float32)[keep]

    # position of each point inside its pair (points are sorted by pair, conc)
    order = np.lexsort((lc, row))
    row, lc, vb = row[order], lc[order], vb[order]
    starts = np.searchsorted(row, np.arange(n_pairs), side="left")
    pos = np.arange(row.size) - starts[row]
    ok = pos < max_points
    log_conc[row[ok], pos[ok]] = lc[ok]
    viability[row[ok], pos[ok]] = vb[ok]
    mask[row[ok], pos[ok]] = True

    return {"log_conc": log_conc, "viability": viability, "mask": mask}
