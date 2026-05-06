#!/usr/bin/env python3
"""Download external validation cohorts used by PathOmicDRP analyses.

This script creates the file layout expected by the existing analysis code:

  data/08_metabric/
    data_clinical_patient.txt
    data_clinical_sample.txt
    data_mrna_illumina_microarray.txt

  data/09_depmap/
    Model.csv
    CRISPRGeneDependency.csv

  data/10_cptac/
    data_clinical_patient.txt
    data_clinical_sample.txt
    data_mrna_seq_fpkm.txt
    data_mutations.txt

METABRIC and CPTAC-BRCA are downloaded from cBioPortal DataHub.  The direct
cBioPortal study tarballs are tried first; if that route is unavailable, the
script downloads the required Git LFS-backed files through GitHub's media URL.
A bounded Git LFS sparse checkout remains as a final fallback.
DepMap 22Q4 files are resolved through the public Figshare API so file IDs and
checksums do not have to be hard-coded except for the article version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
CACHE_DIR = DATA_ROOT / "_downloads" / "external_validation"

CBIO_REPO = "https://github.com/cBioPortal/datahub.git"
CBIO_RAW_BASE = "https://raw.githubusercontent.com/cBioPortal/datahub/master/public"
CBIO_MEDIA_BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
CBIO_STUDY_URLS = {
    "brca_metabric": [
        "https://download.cbioportal.org/brca_metabric.tar.gz",
        "http://download.cbioportal.org/brca_metabric.tar.gz",
    ],
    "brca_cptac_2020": [
        "https://download.cbioportal.org/brca_cptac_2020.tar.gz",
        "http://download.cbioportal.org/brca_cptac_2020.tar.gz",
    ],
}

CBIO_STUDIES = {
    "brca_metabric": {
        "out_dir": DATA_ROOT / "08_metabric",
        "required": [
            "data_clinical_patient.txt",
            "data_clinical_sample.txt",
            "data_mrna_illumina_microarray.txt",
        ],
        "optional": [
            "LICENSE",
            "Readme.txt",
            "meta_clinical_patient.txt",
            "meta_clinical_sample.txt",
            "meta_mrna_illumina_microarray.txt",
            "meta_study.txt",
        ],
    },
    "brca_cptac_2020": {
        "out_dir": DATA_ROOT / "10_cptac",
        "required": [
            "data_clinical_patient.txt",
            "data_clinical_sample.txt",
            "data_mrna_seq_fpkm.txt",
            "data_mutations.txt",
        ],
        "optional": [
            "LICENSE",
            "README.md",
            "meta_clinical_patient.txt",
            "meta_clinical_sample.txt",
            "meta_mrna_seq_fpkm.txt",
            "meta_mutations.txt",
            "meta_study.txt",
        ],
    },
}

DEPMAP_ARTICLE_API = "https://api.figshare.com/v2/articles/21637199"
DEPMAP_OUT = DATA_ROOT / "09_depmap"
DEPMAP_REQUIRED = ["Model.csv", "CRISPRGeneDependency.csv"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def existing_ok(paths: Iterable[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 1024 for p in paths)


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        first = path.read_text(errors="ignore").splitlines()[:1]
    except OSError:
        return False
    return bool(first and first[0].startswith("version https://git-lfs.github.com/spec/v1"))


def validate_not_pointer(paths: Iterable[Path]) -> None:
    pointers = [str(p) for p in paths if is_git_lfs_pointer(p)]
    if pointers:
        joined = "\n  ".join(pointers)
        raise RuntimeError(f"Downloaded Git LFS pointer files instead of data:\n  {joined}")


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path, force: bool = False, expected_md5: str | None = None) -> Path:
    ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        if expected_md5 and md5sum(out_path) != expected_md5:
            log(f"  checksum mismatch for existing {out_path.name}; re-downloading")
        else:
            log(f"  exists: {out_path}")
            return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"  downloading {url}")
    req = Request(url, headers={"User-Agent": "PathOmicDRP/1.0"})
    with urlopen(req, timeout=120) as resp, tmp_path.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded >= next_report:
                pct = 100.0 * downloaded / total
                log(f"    {out_path.name}: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.1f}%)")
                next_report = downloaded + 100 * 1024 * 1024

    if expected_md5:
        got = md5sum(tmp_path)
        if got != expected_md5:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 mismatch for {out_path.name}: expected {expected_md5}, got {got}")

    tmp_path.replace(out_path)
    log(f"  saved: {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return out_path


def safe_tar_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in tar.getmembers():
        name = Path(member.name)
        if member.isdir() or name.is_absolute() or ".." in name.parts:
            continue
        members.append(member)
    return members


def extract_selected_from_tar(tar_path: Path, study: str, wanted: list[str], out_dir: Path, force: bool) -> None:
    ensure_dir(out_dir)
    wanted_set = set(wanted)
    found = set()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in safe_tar_members(tar):
            filename = Path(member.name).name
            if filename not in wanted_set:
                continue
            out_path = out_dir / filename
            if out_path.exists() and not force:
                found.add(filename)
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            found.add(filename)
            log(f"  extracted {study}/{filename} -> {out_path}")

    missing = wanted_set - found
    if missing:
        raise RuntimeError(f"{study} tarball did not contain: {', '.join(sorted(missing))}")


def download_cbio_tarball(study: str, wanted: list[str], out_dir: Path, force: bool) -> bool:
    archive_path = CACHE_DIR / f"{study}.tar.gz"
    for url in CBIO_STUDY_URLS[study]:
        try:
            download_file(url, archive_path, force=force)
            extract_selected_from_tar(archive_path, study, wanted, out_dir, force=force)
            validate_not_pointer(out_dir / name for name in wanted)
            return True
        except (HTTPError, URLError, TimeoutError, tarfile.TarError, RuntimeError, OSError) as exc:
            log(f"  cBioPortal tarball failed for {study}: {exc}")
    return False


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> None:
    log("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout)


def git_lfs_available() -> bool:
    return shutil.which("git") is not None and shutil.which("git-lfs") is not None


def download_cbio_with_github_media(study: str, filenames: list[str], out_dir: Path, force: bool) -> bool:
    """Download Git LFS-backed cBioPortal files without invoking git-lfs."""
    ensure_dir(out_dir)
    try:
        for filename in filenames:
            url = f"{CBIO_MEDIA_BASE}/{study}/{filename}"
            download_file(url, out_dir / filename, force=force)
        validate_not_pointer(out_dir / name for name in filenames)
        return True
    except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
        log(f"  GitHub media download failed for {study}: {exc}")
        return False


def download_cbio_with_git_lfs(study: str, wanted: list[str], out_dir: Path, force: bool) -> bool:
    if not git_lfs_available():
        log("  git/git-lfs not available; skipping DataHub Git LFS fallback")
        return False

    ensure_dir(out_dir)
    with tempfile.TemporaryDirectory(prefix=f"{study}_", dir=str(CACHE_DIR)) as tmp:
        repo_dir = Path(tmp) / "datahub"
        study_path = f"public/{study}"
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", CBIO_REPO, str(repo_dir)])
        run(["git", "sparse-checkout", "set", study_path], cwd=repo_dir)
        try:
            run(["git", "-c", "lfs.fetchexclude=", "lfs", "pull", "-I", study_path], cwd=repo_dir, timeout=900)
        except subprocess.TimeoutExpired as exc:
            log(f"  git-lfs timed out after {exc.timeout} seconds for {study}")
            return False

        for filename in wanted:
            src = repo_dir / study_path / filename
            dst = out_dir / filename
            if not src.exists():
                raise RuntimeError(f"DataHub Git LFS checkout missing {study}/{filename}")
            if dst.exists() and not force:
                continue
            shutil.copy2(src, dst)
            log(f"  copied {study}/{filename} -> {dst}")

    validate_not_pointer(out_dir / name for name in wanted)
    return True


def download_cbio_study(study: str, force: bool) -> None:
    spec = CBIO_STUDIES[study]
    out_dir = spec["out_dir"]
    wanted = spec["required"] + spec["optional"]
    required_paths = [out_dir / name for name in spec["required"]]

    log(f"== cBioPortal: {study} -> {out_dir} ==")
    if existing_ok(required_paths) and not force:
        log("  required files already exist; skipping")
        validate_not_pointer(required_paths)
        return

    ensure_dir(CACHE_DIR)
    if download_cbio_tarball(study, wanted, out_dir, force=force):
        return
    if download_cbio_with_github_media(study, spec["required"], out_dir, force=force):
        return
    if download_cbio_with_git_lfs(study, wanted, out_dir, force=force):
        return

    raw_hint = "\n".join(f"    {CBIO_MEDIA_BASE}/{study}/{name}" for name in spec["required"])
    raise RuntimeError(
        f"Could not download {study}. Try the GitHub media URLs manually or install/update git-lfs.\n"
        f"Required files:\n{raw_hint}"
    )


def load_depmap_article() -> dict:
    req = Request(DEPMAP_ARTICLE_API, headers={"User-Agent": "PathOmicDRP/1.0"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_depmap(force: bool) -> None:
    log(f"== DepMap 22Q4 -> {DEPMAP_OUT} ==")
    ensure_dir(DEPMAP_OUT)
    required_paths = [DEPMAP_OUT / name for name in DEPMAP_REQUIRED]
    if existing_ok(required_paths) and not force:
        log("  required files already exist; skipping")
        return

    article = load_depmap_article()
    files = {f["name"]: f for f in article.get("files", [])}
    missing = [name for name in DEPMAP_REQUIRED if name not in files]
    if missing:
        raise RuntimeError(f"DepMap Figshare article missing files: {', '.join(missing)}")

    manifest = {
        "article": DEPMAP_ARTICLE_API,
        "figshare_url": article.get("figshare_url"),
        "doi": article.get("doi"),
        "version": article.get("version"),
        "files": {},
    }

    for name in DEPMAP_REQUIRED:
        info = files[name]
        out_path = DEPMAP_OUT / name
        download_file(info["download_url"], out_path, force=force, expected_md5=info.get("computed_md5"))
        manifest["files"][name] = {
            "download_url": info.get("download_url"),
            "size": info.get("size"),
            "md5": info.get("computed_md5"),
        }

    manifest_path = DEPMAP_OUT / "figshare_manifest_22Q4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"  wrote: {manifest_path}")


def verify_layout() -> None:
    checks = {
        "METABRIC": [CBIO_STUDIES["brca_metabric"]["out_dir"] / name for name in CBIO_STUDIES["brca_metabric"]["required"]],
        "DepMap": [DEPMAP_OUT / name for name in DEPMAP_REQUIRED],
        "CPTAC-BRCA": [CBIO_STUDIES["brca_cptac_2020"]["out_dir"] / name for name in CBIO_STUDIES["brca_cptac_2020"]["required"]],
    }
    problems = []
    for cohort, paths in checks.items():
        for path in paths:
            if not path.exists():
                problems.append(f"{cohort}: missing {path}")
            elif path.stat().st_size <= 1024:
                problems.append(f"{cohort}: suspiciously small {path} ({path.stat().st_size} bytes)")
            elif is_git_lfs_pointer(path):
                problems.append(f"{cohort}: Git LFS pointer instead of data {path}")
    if problems:
        raise RuntimeError("Layout verification failed:\n  " + "\n  ".join(problems))
    log("All required external validation files are present.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and overwrite existing files.")
    parser.add_argument("--skip-metabric", action="store_true", help="Skip METABRIC download.")
    parser.add_argument("--skip-depmap", action="store_true", help="Skip DepMap download.")
    parser.add_argument("--skip-cptac", action="store_true", help="Skip CPTAC-BRCA download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(CACHE_DIR)

    if not args.skip_metabric:
        download_cbio_study("brca_metabric", force=args.force)
    if not args.skip_depmap:
        download_depmap(force=args.force)
    if not args.skip_cptac:
        download_cbio_study("brca_cptac_2020", force=args.force)

    verify_layout()
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
