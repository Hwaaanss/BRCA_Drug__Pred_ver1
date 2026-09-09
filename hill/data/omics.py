"""Cell-line omics assembly.

Builds one aligned matrix ``(n_cell_lines, n_features)`` from the modalities
requested in ``cfg.data.omics_modalities``, together with the per-feature
modality label the tokeniser needs.

Every loader is defensive about file layout: if a release changes the column
names, the adapter raises with the columns it actually found instead of
silently producing a wrong matrix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hill.utils.logging import get_logger

log = get_logger("data.omics")

_SAMPLE_COLS = ("cosmic_id", "cosmic", "model_id", "model_name", "cell_line_name", "sample", "sample_name")
_GENE_COLS = ("gene_symbol", "gene", "symbol", "gene_name", "hugo_symbol")


@dataclass
class OmicsMatrix:
    values: np.ndarray            # (C, F) float32
    cell_ids: list[str]           # COSMIC ids as strings
    feature_names: list[str]
    feature_modality: list[str]

    def variance(self) -> np.ndarray:
        return np.nanvar(self.values, axis=0)

    def subset_cells(self, cell_ids: Sequence[str]) -> "OmicsMatrix":
        pos = {c: i for i, c in enumerate(self.cell_ids)}
        rows = [pos[c] for c in cell_ids if c in pos]
        return OmicsMatrix(
            values=self.values[rows],
            cell_ids=[self.cell_ids[r] for r in rows],
            feature_names=list(self.feature_names),
            feature_modality=list(self.feature_modality),
        )


def _lower_map(columns: Sequence[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c) for c in columns}


def _pick(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lm = _lower_map(columns)
    for c in candidates:
        if c in lm:
            return lm[c]
    return None


def load_expression(path: str | Path) -> pd.DataFrame:
    """GDSC RMA basal expression: genes x ``DATA.<COSMIC_ID>`` columns.

    Returns a cells x genes frame indexed by COSMIC id (as string).
    """
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(p, sep=sep, low_memory=False)
    gene_col = _pick(df.columns, ("gene_symbols", *_GENE_COLS))
    if gene_col is None:
        raise KeyError(f"{p.name}: no gene symbol column; found {list(df.columns)[:8]}")
    data_cols = [c for c in df.columns if str(c).upper().startswith("DATA.")]
    if not data_cols:
        # Already samples x genes?
        idx_col = _pick(df.columns, _SAMPLE_COLS)
        if idx_col is None:
            raise KeyError(
                f"{p.name}: neither DATA.<COSMIC> columns nor a sample id column were found"
            )
        out = df.set_index(idx_col)
        out.index = out.index.astype(str).str.replace(r"\.0$", "", regex=True)
        return out.select_dtypes("number")
    df = df.dropna(subset=[gene_col])
    df = df.drop_duplicates(subset=[gene_col], keep="first").set_index(gene_col)
    expr = df[data_cols].T
    expr.index = [re.sub(r"^DATA\.", "", str(c)) for c in data_cols]
    expr.index.name = "cell_id"
    return expr.astype(np.float32)


def load_mutations(path: str | Path, cell_id_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Long-format mutation table -> binary cells x genes matrix."""
    p = Path(path)
    df = pd.read_csv(p, low_memory=False)
    sample_col = _pick(df.columns, _SAMPLE_COLS)
    gene_col = _pick(df.columns, _GENE_COLS)
    if sample_col is None or gene_col is None:
        raise KeyError(
            f"{p.name}: need a sample column {_SAMPLE_COLS} and a gene column {_GENE_COLS}; "
            f"found {list(df.columns)[:12]}"
        )
    df = df[[sample_col, gene_col]].dropna()
    df[sample_col] = df[sample_col].astype(str).str.replace(r"\.0$", "", regex=True)
    if cell_id_map:
        df[sample_col] = df[sample_col].map(lambda s: cell_id_map.get(s, s))
    df["value"] = 1.0
    mat = df.pivot_table(
        index=sample_col, columns=gene_col, values="value", aggfunc="max", fill_value=0.0
    )
    mat.index.name = "cell_id"
    return mat.astype(np.float32)


def load_cnv(path: str | Path, cell_id_map: dict[str, str] | None = None) -> pd.DataFrame:
    """GISTIC-style copy number: auto-detects genes x cells vs cells x genes."""
    p = Path(path)
    sep = "\t" if p.suffix.lower() in {".txt", ".tsv"} else ","
    df = pd.read_csv(p, sep=sep, low_memory=False)
    gene_col = _pick(df.columns, _GENE_COLS)
    sample_col = _pick(df.columns, _SAMPLE_COLS)
    if gene_col is not None and sample_col is not None and "value" in _lower_map(df.columns):
        val_col = _lower_map(df.columns)["value"]
        mat = df.pivot_table(index=sample_col, columns=gene_col, values=val_col, aggfunc="mean")
    elif gene_col is not None:
        df = df.drop_duplicates(subset=[gene_col]).set_index(gene_col)
        mat = df.select_dtypes("number").T
    elif sample_col is not None:
        mat = df.set_index(sample_col).select_dtypes("number")
    else:
        raise KeyError(f"{p.name}: cannot infer orientation; columns {list(df.columns)[:8]}")
    mat.index = mat.index.astype(str).str.replace(r"^DATA\.|\.0$", "", regex=True)
    if cell_id_map:
        mat.index = [cell_id_map.get(i, i) for i in mat.index]
    mat.index.name = "cell_id"
    return mat.astype(np.float32)


def build_cell_id_map(annotation_path: str | Path) -> dict[str, str]:
    """Map cell-line names / model ids to COSMIC ids using Cell_Lines_Details."""
    p = Path(annotation_path)
    if not p.exists():
        log.warning("cell line annotation missing (%s); identifiers will not be harmonised", p)
        return {}
    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    cosmic_col = next((c for c in df.columns if "cosmic" in c.lower()), None)
    name_cols = [c for c in df.columns if "sample name" in c.lower() or "cell line name" in c.lower()]
    if cosmic_col is None or not name_cols:
        log.warning("%s: no COSMIC/name columns (%s)", p.name, list(df.columns)[:8])
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        cosmic = row[cosmic_col]
        if pd.isna(cosmic):
            continue
        cid = str(int(cosmic))
        for c in name_cols:
            val = row[c]
            if isinstance(val, str) and val.strip():
                mapping[val.strip()] = cid
                mapping[val.strip().upper().replace("-", "").replace(" ", "")] = cid
    log.info("cell id map: %d aliases -> COSMIC", len(mapping))
    return mapping


def _resolve_path(cfg: Any, modality: str, raw_dir: Path) -> Path | None:
    override = cfg.data.omics_files.get(modality)
    if override:
        return Path(override)
    patterns = {
        "expression": ("*RMA_proc_basalExp*", "*expression*"),
        "mutation": ("*mutations_all*", "*mutation*"),
        "cnv": ("*cnv*", "*gistic*"),
        "proteomic": ("*rppa*", "*proteom*"),
    }[modality]
    for pat in patterns:
        hits = sorted(
            q for q in (raw_dir / "omics").glob(pat)
            if q.is_file() and q.suffix.lower() in {".csv", ".txt", ".tsv", ".parquet"}
        )
        if hits:
            return hits[0]
    return None


def assemble_cell_omics(cfg: Any) -> OmicsMatrix:
    """Load every requested modality, align on COSMIC id, and concatenate."""
    raw_dir = Path(cfg.paths.raw_dir)
    cell_map = build_cell_id_map(raw_dir / "gdsc_annotation" / "Cell_Lines_Details.xlsx")

    frames: list[tuple[str, pd.DataFrame]] = []
    for modality in cfg.data.omics_modalities:
        path = _resolve_path(cfg, modality, raw_dir)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"no file found for omics modality {modality!r} under {raw_dir / 'omics'}. "
                "Run `python -m hill.data.download --what omics` or set "
                f"data.omics_files.{modality} in the config."
            )
        if modality == "expression":
            df = load_expression(path)
        elif modality == "mutation":
            df = load_mutations(path, cell_map)
        elif modality == "cnv":
            df = load_cnv(path, cell_map)
        else:
            df = load_cnv(path, cell_map)
        df = df.loc[~df.index.duplicated(keep="first")]
        log.info("%s: %d cells x %d features (%s)", modality, df.shape[0], df.shape[1], path.name)
        frames.append((modality, df))

    common = set(frames[0][1].index)
    for _, df in frames[1:]:
        common &= set(df.index)
    cells = sorted(common)
    if not cells:
        raise ValueError(
            "no cell line is present in every requested modality — check identifier harmonisation "
            f"({[ (m, list(df.index[:3])) for m, df in frames ]})"
        )

    blocks, names, modalities = [], [], []
    for modality, df in frames:
        sub = df.loc[cells]
        sub = sub.loc[:, sub.notna().mean() >= 0.5]
        values = sub.to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=float(np.nanmedian(values)) if values.size else 0.0)
        if modality == "expression" and cfg.data.expression_log1p and values.min() >= 0:
            values = np.log1p(values)
        blocks.append(values)
        names.extend([f"{c}_{modality[:3]}" if modality != "expression" else str(c) for c in sub.columns])
        modalities.extend([modality] * sub.shape[1])

    matrix = np.concatenate(blocks, axis=1)

    # variance filter, applied per modality so a modality is never wiped out
    if cfg.data.max_genes and matrix.shape[1] > cfg.data.max_genes:
        var = np.nanvar(matrix, axis=0)
        mod_arr = np.array(modalities)
        keep = np.zeros(matrix.shape[1], dtype=bool)
        for m in dict.fromkeys(modalities):
            idx = np.flatnonzero(mod_arr == m)
            quota = max(1, int(round(cfg.data.max_genes * idx.size / matrix.shape[1])))
            keep[idx[np.argsort(-var[idx])[:quota]]] = True
        matrix = matrix[:, keep]
        names = [n for n, k in zip(names, keep) if k]
        modalities = [m for m, k in zip(modalities, keep) if k]
        log.info("variance filter: kept %d features", matrix.shape[1])

    log.info("omics matrix: %d cells x %d features", matrix.shape[0], matrix.shape[1])
    return OmicsMatrix(
        values=np.ascontiguousarray(matrix, dtype=np.float32),
        cell_ids=[str(c) for c in cells],
        feature_names=names,
        feature_modality=modalities,
    )


def save_omics(matrix: OmicsMatrix, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        values=matrix.values,
        cell_ids=np.array(matrix.cell_ids),
        feature_names=np.array(matrix.feature_names),
        feature_modality=np.array(matrix.feature_modality),
    )


def load_omics(path: str | Path) -> OmicsMatrix:
    with np.load(Path(path), allow_pickle=False) as z:
        return OmicsMatrix(
            values=z["values"],
            cell_ids=[str(c) for c in z["cell_ids"]],
            feature_names=[str(c) for c in z["feature_names"]],
            feature_modality=[str(c) for c in z["feature_modality"]],
        )
