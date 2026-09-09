"""Fetch the external files the pipeline needs.

Nothing here invents data.  A failed download is reported as a failure and the
downstream step refuses to run, rather than falling back to a substitute file
(project rule R2).

    python -m hill.data.download --what all
    python -m hill.data.download --what gdsc-raw pathways
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

import yaml

from hill.utils.logging import get_logger

log = get_logger("data.download")

_GROUPS = {
    "gdsc-raw": "gdsc_raw",
    "gdsc-fitted": "gdsc_fitted",
    "gdsc-annotation": "gdsc_annotation",
    "omics": "omics",
    "pathways": "pathways",
}


def load_sources(path: str | Path = "configs/data_sources.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data source manifest not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def download_file(url: str, dest: Path, timeout: int = 300, chunk: int = 1 << 20) -> bool:
    """Stream ``url`` to ``dest``.  Returns True on success, False on failure."""
    import requests

    if dest.exists() and dest.stat().st_size > 0:
        log.info("exists, skipping: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            written = 0
            with tmp.open("wb") as fh:
                for block in resp.iter_content(chunk_size=chunk):
                    fh.write(block)
                    written += len(block)
                    if total and written % (50 * chunk) < chunk:
                        log.info("  %s: %.0f%%", dest.name, 100 * written / total)
        tmp.replace(dest)
        log.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:  # noqa: BLE001 - report, never mask
        log.error("FAILED to download %s -> %s: %s", url, dest.name, exc)
        tmp.unlink(missing_ok=True)
        return False


def maybe_unzip(path: Path) -> list[Path]:
    """Extract a zip next to itself; returns the extracted paths."""
    if path.suffix.lower() != ".zip":
        return [path]
    out_dir = path.parent
    extracted: list[Path] = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            target = out_dir / Path(member).name
            if not target.exists():
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            extracted.append(target)
    log.info("unzipped %s -> %s", path.name, [p.name for p in extracted])
    return extracted


def fetch_group(group: str, sources: dict[str, Any], raw_dir: Path) -> dict[str, dict[str, Any]]:
    key = _GROUPS[group]
    entries = sources.get(key, {})
    out: dict[str, dict[str, Any]] = {}
    sub_dir = raw_dir / key
    for name, spec in entries.items():
        dest = sub_dir / spec["filename"]
        ok = download_file(spec["url"], dest)
        files = maybe_unzip(dest) if ok and spec.get("unzip") else ([dest] if ok else [])
        out[name] = {
            "ok": ok,
            "url": spec["url"],
            "path": str(dest),
            "extracted": [str(f) for f in files],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download external data for HILL")
    ap.add_argument("--what", nargs="+", default=["all"], choices=list(_GROUPS) + ["all"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--sources", default="configs/data_sources.yaml")
    args = ap.parse_args(argv)

    groups: Iterable[str] = list(_GROUPS) if "all" in args.what else args.what
    sources = load_sources(args.sources)
    raw_dir = Path(args.raw_dir)

    report: dict[str, Any] = {}
    for g in groups:
        report[g] = fetch_group(g, sources, raw_dir)

    failures = [
        f"{g}/{n}" for g, entries in report.items() for n, r in entries.items() if not r["ok"]
    ]
    if failures:
        log.error("%d download(s) FAILED: %s", len(failures), ", ".join(failures))
        log.error("Fix the URLs in %s or place the files manually under %s", args.sources, raw_dir)
        return 1
    log.info("all requested downloads completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
