"""Fetch the external files the pipeline needs.

Nothing here invents data.  A failed download is reported as a failure and the
downstream step refuses to run, rather than falling back to a substitute file
(project rule R2).

    python -m hill.data.download --what all
    python -m hill.data.download --what gdsc-raw pathways
    python -m hill.data.download --probe                # which candidate URLs resolve?
    python -m hill.data.download --discover GDSC_release8.5   # list what the bucket holds

GDSC moves files between release folders and renames them with each release, so
every entry may list several candidate URLs; the first one that responds wins.
When they all 404, ``--discover`` lists the bucket so the real key can be found
in seconds, and ``--probe`` checks candidates without downloading anything.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

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

# S3-compatible bucket that hosts the GDSC bulk downloads.
GDSC_BUCKET = "https://cog.sanger.ac.uk/cancerrxgene/"


def load_sources(path: str | Path = "configs/data_sources.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data source manifest not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def candidate_urls(spec: dict[str, Any]) -> list[str]:
    """An entry may carry a single ``url`` or an ordered list of ``urls``."""
    urls = spec.get("urls") or ([spec["url"]] if spec.get("url") else [])
    if not urls:
        raise KeyError(f"source entry has neither 'url' nor 'urls': {spec}")
    return list(urls)


def download_file(
    urls: str | Iterable[str],
    dest: Path,
    timeout: int = 300,
    chunk: int = 1 << 20,
    retries: int = 4,
) -> tuple[bool, str | None]:
    """Stream the first working URL to ``dest``, resuming after a broken transfer.

    Returns ``(ok, url_that_worked)``.  A truncated transfer (the
    ``IncompleteRead`` that GDSC's server produces on large files) is retried
    with an HTTP ``Range`` request so the bytes already on disk are kept.
    """
    import requests

    if isinstance(urls, str):
        urls = [urls]
    urls = list(urls)

    if dest.exists() and dest.stat().st_size > 0:
        log.info("exists, skipping: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True, None

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for url in urls:
        for attempt in range(retries):
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={have}-"} if (have and attempt > 0) else {}
            try:
                with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                    if resp.status_code == 416 and have:  # already complete
                        tmp.replace(dest)
                        log.info("resumed to completion: %s", dest.name)
                        return True, url
                    resp.raise_for_status()
                    resuming = resp.status_code == 206
                    if not resuming:
                        have = 0  # server ignored Range: start over
                    total = int(resp.headers.get("content-length", 0)) + have
                    written = have
                    with tmp.open("ab" if resuming else "wb") as fh:
                        for block in resp.iter_content(chunk_size=chunk):
                            fh.write(block)
                            written += len(block)
                            if total and written % (50 * chunk) < chunk:
                                log.info("  %s: %.0f%%", dest.name, 100 * written / total)
                if total and written < total:
                    raise OSError(f"truncated: {written} of {total} bytes")
                tmp.replace(dest)
                log.info("downloaded %s (%.1f MB) from %s", dest.name, dest.stat().st_size / 1e6, url)
                return True, url
            except Exception as exc:  # noqa: BLE001 - report, never mask
                status = getattr(getattr(exc, "response", None), "status_code", None)
                fatal = status in {403, 404, 410}
                log.warning(
                    "%s attempt %d/%d failed (%s): %s",
                    dest.name, attempt + 1, retries, url, exc,
                )
                if fatal:
                    tmp.unlink(missing_ok=True)
                    break  # a missing file will not appear on retry: try the next URL
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
    log.error("FAILED to download %s from %d candidate URL(s)", dest.name, len(urls))
    return False, None


def probe_url(url: str, timeout: int = 30) -> tuple[int | None, int | None]:
    """HEAD a URL (GET fallback). Returns ``(status_code, content_length)``."""
    import requests

    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code in {403, 405}:  # some buckets refuse HEAD
            resp = requests.get(url, timeout=timeout, stream=True, headers={"Range": "bytes=0-0"})
        size = resp.headers.get("content-length")
        if resp.status_code == 206:
            crange = resp.headers.get("content-range", "")
            size = crange.split("/")[-1] if "/" in crange else size
        return resp.status_code, int(size) if size and size.isdigit() else None
    except Exception as exc:  # noqa: BLE001
        log.debug("probe failed for %s: %s", url, exc)
        return None, None


def probe_sources(sources: dict[str, Any], groups: Iterable[str]) -> dict[str, Any]:
    """Check every candidate URL without downloading. Prints a table."""
    report: dict[str, Any] = {}
    print(f"{'entry':<28} {'status':>7} {'size':>10}  url")
    print("-" * 110)
    for group in groups:
        for name, spec in (sources.get(_GROUPS[group], {}) or {}).items():
            key = f"{group}/{name}"
            rows = []
            for url in candidate_urls(spec):
                status, size = probe_url(url)
                mb = f"{size / 1e6:.1f} MB" if size else "-"
                ok = status is not None and 200 <= status < 300
                print(f"{key:<28} {str(status or 'ERR'):>7} {mb:>10}  {'OK  ' if ok else '    '}{url}")
                rows.append({"url": url, "status": status, "bytes": size, "ok": ok})
                if ok:
                    break
            report[key] = rows
    working = sum(1 for rows in report.values() if any(r["ok"] for r in rows))
    print("-" * 110)
    print(f"{working} / {len(report)} entries have at least one working URL")
    return report


def discover(prefix: str = "", pattern: str = "", bucket: str = GDSC_BUCKET,
             max_keys: int = 1000) -> list[tuple[str, int]]:
    """List keys in the GDSC bucket so the real filename can be found.

    The bucket is S3-compatible and normally allows anonymous listing.  Use this
    when a download 404s: it shows exactly what the release folder contains.
    """
    import requests

    keys: list[tuple[str, int]] = []
    token = None
    rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if token:
            params["continuation-token"] = token
        try:
            resp = requests.get(bucket, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log.error("bucket listing failed (%s): %s", bucket, exc)
            log.error("Open the download page in a browser instead: "
                      "https://www.cancerrxgene.org/downloads/bulk_download")
            return keys
        root = ElementTree.fromstring(resp.text)
        ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        find = (lambda el, tag: el.findall(f"s3:{tag}", ns)) if ns else (lambda el, tag: el.findall(tag))
        for content in find(root, "Contents"):
            key_el = find(content, "Key")
            size_el = find(content, "Size")
            if not key_el:
                continue
            key = key_el[0].text or ""
            size = int(size_el[0].text or 0) if size_el else 0
            if rx is None or rx.search(key):
                keys.append((key, size))
        trunc = find(root, "IsTruncated")
        next_tok = find(root, "NextContinuationToken")
        if trunc and (trunc[0].text or "").lower() == "true" and next_tok:
            token = next_tok[0].text
        else:
            break
    for key, size in keys:
        print(f"{size / 1e6:10.1f} MB  {bucket}{key}")
    print(f"\n{len(keys)} key(s) matched prefix={prefix!r} pattern={pattern!r}")
    return keys


def autofix_urls(
    sources: dict[str, Any],
    groups: Iterable[str],
    bucket: str = GDSC_BUCKET,
    prefix: str = "",
    write_to: str | Path | None = None,
) -> dict[str, list[str]]:
    """Find the real URL for every entry by listing the bucket and matching.

    GDSC keeps files in the release folder where they were introduced, so a
    hard-coded release path rots. Each manifest entry may carry a ``match``
    regex; this lists the bucket once and reports which key actually satisfies
    it. With ``write_to`` the discovered URL is written to the head of the
    entry's ``urls`` list, so the manifest self-heals.
    """
    listing = discover(prefix=prefix, pattern="", bucket=bucket)
    if not listing:
        log.error("bucket listing empty or unavailable — cannot autofix")
        return {}

    found: dict[str, list[str]] = {}
    for group in groups:
        key = _GROUPS[group]
        for name, spec in (sources.get(key, {}) or {}).items():
            pattern = spec.get("match")
            if not pattern:
                continue
            rx = re.compile(pattern, re.IGNORECASE)
            hits = [(k, size) for k, size in listing if rx.search(k)]
            if not hits:
                log.warning("%s/%s: nothing in the bucket matches %r", group, name, pattern)
                continue
            # newest release folder first, then largest file
            hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
            urls = [f"{bucket}{k}" for k, _ in hits[:3]]
            found[f"{group}/{name}"] = urls
            log.info("%s/%s -> %s", group, name, urls[0])
            if write_to:
                existing = [u for u in candidate_urls(spec) if u not in urls]
                spec.pop("url", None)
                spec["urls"] = urls + existing

    if write_to and found:
        Path(write_to).write_text(yaml.safe_dump(sources, sort_keys=False), encoding="utf-8")
        log.info("rewrote %s with %d discovered URL(s)", write_to, len(found))
    return found


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
    entries = sources.get(key, {}) or {}
    out: dict[str, dict[str, Any]] = {}
    sub_dir = raw_dir / key
    for name, spec in entries.items():
        dest = sub_dir / spec["filename"]
        urls = candidate_urls(spec)
        ok, used = download_file(urls, dest)
        files = maybe_unzip(dest) if ok and spec.get("unzip") else ([dest] if ok else [])
        out[name] = {
            "ok": ok,
            "url": used,
            "candidates": urls,
            "path": str(dest),
            "extracted": [str(f) for f in files],
            "optional": bool(spec.get("optional", False)),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download external data for HILL")
    ap.add_argument("--what", nargs="+", default=["all"], choices=list(_GROUPS) + ["all"])
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--sources", default="configs/data_sources.yaml")
    ap.add_argument("--probe", action="store_true",
                    help="check which candidate URLs resolve, download nothing")
    ap.add_argument("--discover", metavar="PREFIX", default=None,
                    help="list keys in the GDSC bucket under PREFIX (e.g. GDSC_release8.5)")
    ap.add_argument("--pattern", default="", help="regex filter for --discover")
    ap.add_argument("--autofix", action="store_true",
                    help="list the bucket and report the URL that really matches each entry")
    ap.add_argument("--write", action="store_true",
                    help="with --autofix, write the discovered URLs back into the manifest")
    ap.add_argument("--bucket-prefix", default="GDSC",
                    help="prefix used when listing the bucket for --autofix")
    args = ap.parse_args(argv)

    if args.discover is not None:
        discover(prefix=args.discover, pattern=args.pattern)
        return 0

    groups: Iterable[str] = list(_GROUPS) if "all" in args.what else args.what
    sources = load_sources(args.sources)

    if args.autofix:
        found = autofix_urls(sources, groups, prefix=args.bucket_prefix,
                             write_to=args.sources if args.write else None)
        if not args.write and found:
            print("\nRun again with --write to store these in the manifest.")
        return 0 if found else 1

    if args.probe:
        probe_sources(sources, groups)
        return 0

    raw_dir = Path(args.raw_dir)
    report: dict[str, Any] = {}
    for g in groups:
        report[g] = fetch_group(g, sources, raw_dir)

    required_failures = [
        f"{g}/{n}" for g, entries in report.items()
        for n, r in entries.items() if not r["ok"] and not r["optional"]
    ]
    optional_failures = [
        f"{g}/{n}" for g, entries in report.items()
        for n, r in entries.items() if not r["ok"] and r["optional"]
    ]
    if optional_failures:
        log.warning("%d optional download(s) failed (safe to ignore with the default "
                    "config): %s", len(optional_failures), ", ".join(optional_failures))
    if required_failures:
        log.error("%d REQUIRED download(s) FAILED: %s", len(required_failures),
                  ", ".join(required_failures))
        log.error("Next step: `python -m hill.data.download --probe` to see which candidate URLs "
                  "resolve, then `--discover GDSC_release8.5 --pattern raw_data` to list what the "
                  "bucket actually holds, and put the working URL in %s.", args.sources)
        return 1
    log.info("all required downloads completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
