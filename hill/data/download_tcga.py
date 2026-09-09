"""Download the TCGA target-domain data from the GDC.

    python -m hill.data.download_tcga --project TCGA-BRCA --what clinical expression mutation

Whole-slide images are *not* downloaded here: the full TCGA-BRCA slide set is
~600 GB, far beyond this project's 30 GB disk budget.  The pipeline consumes
pre-extracted UNI patch features instead (``hill/data/uni_features.py``), which
are ~4 MB per slide.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from hill.utils.logging import get_logger

log = get_logger("data.tcga_download")

GDC_API = "https://api.gdc.cancer.gov"
_CLINICAL_FIELDS = [
    "case_id", "submitter_id",
    "demographic.gender", "demographic.race", "demographic.vital_status",
    "demographic.days_to_death", "demographic.age_at_index",
    "diagnoses.primary_diagnosis", "diagnoses.ajcc_pathologic_stage",
    "diagnoses.age_at_diagnosis", "diagnoses.days_to_last_follow_up",
    "diagnoses.treatments.treatment_type", "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.treatment_outcome", "diagnoses.treatments.treatment_intent_type",
    "diagnoses.treatments.days_to_treatment_start",
]


def _paged_cases(project: str, fields: Iterable[str], page: int = 500) -> list[dict[str, Any]]:
    import requests

    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "filters": json.dumps(
                {"op": "=", "content": {"field": "project.project_id", "value": project}}
            ),
            "fields": ",".join(fields),
            "size": page,
            "from": offset,
            "format": "JSON",
        }
        resp = requests.get(f"{GDC_API}/cases", params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()["data"]
        out.extend(data["hits"])
        total = data["pagination"]["total"]
        offset += page
        log.info("  clinical: %d / %d cases", len(out), total)
        if len(out) >= total:
            return out


def download_clinical(project: str, out_dir: Path) -> pd.DataFrame:
    """Case-level clinical table plus one row per recorded drug treatment."""
    cases = _paged_cases(project, _CLINICAL_FIELDS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases_raw.json").write_text(json.dumps(cases), encoding="utf-8")

    rows, treatments = [], []
    for case in cases:
        demo = case.get("demographic") or {}
        diags = case.get("diagnoses") or []
        diag = diags[0] if diags else {}
        rows.append(
            {
                "patient_id": case.get("submitter_id"),
                "gender": demo.get("gender"),
                "race": demo.get("race"),
                "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                "age_at_index": demo.get("age_at_index"),
                "primary_diagnosis": diag.get("primary_diagnosis"),
                "stage": diag.get("ajcc_pathologic_stage"),
                "days_to_last_follow_up": diag.get("days_to_last_follow_up"),
            }
        )
        for d in diags:
            for t in d.get("treatments") or []:
                agents = t.get("therapeutic_agents")
                if not agents:
                    continue
                treatments.append(
                    {
                        "patient_id": case.get("submitter_id"),
                        "treatment_type": t.get("treatment_type"),
                        "therapeutic_agents": agents,
                        "treatment_outcome": t.get("treatment_outcome"),
                        "treatment_intent": t.get("treatment_intent_type"),
                        "days_to_treatment_start": t.get("days_to_treatment_start"),
                    }
                )
    clinical = pd.DataFrame(rows).drop_duplicates("patient_id")
    clinical.to_csv(out_dir / "clinical.csv", index=False)
    pd.DataFrame(treatments).to_csv(out_dir / "treatments.csv", index=False)
    log.info("clinical: %d patients, %d treatment records", len(clinical), len(treatments))
    return clinical


def _query_files(filters: dict[str, Any], fields: str, size: int = 5000) -> list[dict[str, Any]]:
    import requests

    resp = requests.get(
        f"{GDC_API}/files",
        params={"filters": json.dumps(filters), "fields": fields, "size": size, "format": "JSON"},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


def _download_bundle(file_ids: list[str], dest_dir: Path, batch: int = 100) -> None:
    """POST /data returns a tar.gz bundle; extract it as we go to save disk."""
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(file_ids), batch):
        chunk = file_ids[i : i + batch]
        resp = requests.post(
            f"{GDC_API}/data",
            data=json.dumps({"ids": chunk}),
            headers={"Content-Type": "application/json"},
            timeout=1800,
        )
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.name.endswith("MANIFEST.txt"):
                    continue
                target = dest_dir / Path(member.name).name
                if target.exists():
                    continue
                extracted = tar.extractfile(member)
                if extracted is not None:
                    target.write_bytes(extracted.read())
        log.info("  files %d / %d", min(i + batch, len(file_ids)), len(file_ids))


def _dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def _report_reclaimable(files_dir: Path, matrix_path: Path) -> None:
    """The per-file downloads are redundant once the matrix exists (30 GB budget)."""
    size = _dir_size_gb(files_dir)
    if size > 0.5:
        log.info(
            "%s now holds everything from %s (%.1f GB). Reclaim it with: rm -rf %s",
            matrix_path.name, files_dir.name, size, files_dir,
        )


def download_expression(project: str, out_dir: Path) -> Path:
    """STAR-Counts gene expression; assembled into a patients x genes TPM matrix."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }
    hits = _query_files(filters, "file_id,file_name,cases.submitter_id,cases.samples.sample_type")
    manifest = pd.DataFrame(
        [
            {
                "file_id": h["file_id"],
                "file_name": h["file_name"],
                "patient_id": (h.get("cases") or [{}])[0].get("submitter_id"),
                "sample_type": (((h.get("cases") or [{}])[0].get("samples") or [{}])[0]).get("sample_type"),
            }
            for h in hits
        ]
    )
    manifest = manifest[manifest["sample_type"].astype(str).str.contains("Primary", na=False)]
    manifest = manifest.drop_duplicates("patient_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "expression_manifest.csv", index=False)
    log.info("expression: %d primary-tumour files", len(manifest))

    files_dir = out_dir / "expression_files"
    _download_bundle(manifest["file_id"].tolist(), files_dir)

    series: dict[str, pd.Series] = {}
    for _, row in manifest.iterrows():
        path = files_dir / row["file_name"]
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        df = df[~df["gene_id"].astype(str).str.startswith("N_")]
        value_col = "tpm_unstranded" if "tpm_unstranded" in df.columns else "unstranded"
        s = df.groupby("gene_name")[value_col].max()
        series[row["patient_id"]] = s
    matrix = pd.DataFrame(series).T
    matrix.index.name = "patient_id"
    out_path = out_dir / "expression_tpm.parquet"
    matrix.to_parquet(out_path)
    log.info("expression matrix: %s -> %s", matrix.shape, out_path.name)
    _report_reclaimable(files_dir, out_path)
    return out_path


def download_mutations(project: str, out_dir: Path) -> Path:
    """Open-access MAF files reduced to a binary patient x gene matrix."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }
    hits = _query_files(filters, "file_id,file_name")
    out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "maf_files"
    _download_bundle([h["file_id"] for h in hits], files_dir)

    frames = []
    for path in list(files_dir.glob("*.maf.gz")) + list(files_dir.glob("*.maf")):
        df = pd.read_csv(path, sep="\t", comment="#", low_memory=False,
                         usecols=lambda c: c in {"Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"})
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no MAF files extracted into {files_dir}")
    maf = pd.concat(frames, ignore_index=True)
    silent = {"Silent", "Intron", "3'UTR", "5'UTR", "RNA", "IGR", "5'Flank", "3'Flank"}
    maf = maf[~maf["Variant_Classification"].isin(silent)]
    maf["patient_id"] = maf["Tumor_Sample_Barcode"].astype(str).str.slice(0, 12)
    maf["value"] = 1.0
    matrix = maf.pivot_table(index="patient_id", columns="Hugo_Symbol", values="value",
                             aggfunc="max", fill_value=0.0)
    out_path = out_dir / "mutations_binary.parquet"
    matrix.to_parquet(out_path)
    log.info("mutation matrix: %s -> %s", matrix.shape, out_path.name)
    _report_reclaimable(files_dir, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download TCGA data from the GDC")
    ap.add_argument("--project", default="TCGA-BRCA")
    ap.add_argument("--out-dir", default="data/raw/tcga")
    ap.add_argument("--what", nargs="+", default=["clinical", "expression", "mutation"],
                    choices=["clinical", "expression", "mutation"])
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    failures = []
    for what in args.what:
        try:
            {"clinical": download_clinical, "expression": download_expression,
             "mutation": download_mutations}[what](args.project, out_dir)
        except Exception as exc:  # noqa: BLE001 - report honestly
            log.error("%s download FAILED: %s", what, exc)
            failures.append(what)
    if failures:
        log.error("failed: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
