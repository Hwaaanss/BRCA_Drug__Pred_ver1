"""Downloader robustness: candidate URLs, resume after truncation, probe, discover.

The real GDSC server 404s when a release folder changes and truncates large
transfers (``IncompleteRead``); both are exercised here against a local server.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hill.data.download import (
    candidate_urls, discover, download_file, load_sources, probe_url,
)

PAYLOAD = bytes(range(256)) * 400          # 102_400 bytes
_state = {"flaky_served": 0}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test server
        pass

    def _range_start(self) -> int:
        rng = self.headers.get("Range", "")
        return int(rng.split("=")[1].split("-")[0]) if rng.startswith("bytes=") else 0

    def do_HEAD(self):
        if self.path.split("?", 1)[0] in {"/full.bin", "/flaky.bin"}:
            self.send_response(200)
            self.send_header("content-length", str(len(PAYLOAD)))
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/listing":
            body = (
                '<?xml version="1.0"?>'
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                "<Contents><Key>GDSC_release8.2/GDSC2_public_raw_data.csv.zip</Key>"
                "<Size>1234</Size></Contents>"
                "<Contents><Key>GDSC_release8.2/other.txt</Key><Size>7</Size></Contents>"
                "<IsTruncated>false</IsTruncated></ListBucketResult>"
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/xml")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/full.bin":
            start = self._range_start()
            body = PAYLOAD[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("content-range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/flaky.bin":
            start = self._range_start()
            _state["flaky_served"] += 1
            if _state["flaky_served"] == 1:
                # announce the full size, deliver half, then hang up: the exact
                # failure mode that killed the GDSC1 fitted download
                self.send_response(200)
                self.send_header("content-length", str(len(PAYLOAD)))
                self.end_headers()
                self.wfile.write(PAYLOAD[: len(PAYLOAD) // 2])
                return
            body = PAYLOAD[start:]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("content-range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_candidate_urls_accepts_both_shapes():
    assert candidate_urls({"url": "a"}) == ["a"]
    assert candidate_urls({"urls": ["a", "b"]}) == ["a", "b"]
    with pytest.raises(KeyError):
        candidate_urls({"filename": "x"})


def test_downloads_a_working_url(server, tmp_path):
    dest = tmp_path / "full.bin"
    ok, used = download_file(f"{server}/full.bin", dest)
    assert ok and used.endswith("/full.bin")
    assert dest.read_bytes() == PAYLOAD


def test_falls_through_to_the_next_candidate_on_404(server, tmp_path):
    dest = tmp_path / "candidate.bin"
    ok, used = download_file([f"{server}/missing.zip", f"{server}/full.bin"], dest)
    assert ok, "a 404 on the first candidate must not fail the entry"
    assert used.endswith("/full.bin")
    assert dest.read_bytes() == PAYLOAD


def test_resumes_after_a_truncated_transfer(server, tmp_path):
    """The retry must keep the bytes already on disk and finish the file."""
    _state["flaky_served"] = 0
    dest = tmp_path / "flaky.bin"
    ok, _ = download_file(f"{server}/flaky.bin", dest, retries=3)
    assert ok
    assert dest.read_bytes() == PAYLOAD
    assert _state["flaky_served"] >= 2, "the truncated transfer should have been retried"
    assert not dest.with_suffix(".bin.part").exists()


def test_existing_file_is_not_refetched(server, tmp_path):
    dest = tmp_path / "full.bin"
    dest.write_bytes(b"already here")
    ok, used = download_file(f"{server}/full.bin", dest)
    assert ok and used is None
    assert dest.read_bytes() == b"already here"


def test_all_candidates_missing_reports_failure(server, tmp_path):
    ok, used = download_file([f"{server}/nope1", f"{server}/nope2"], tmp_path / "x.bin")
    assert not ok and used is None


def test_probe_reports_status_and_size(server):
    status, size = probe_url(f"{server}/full.bin")
    assert status == 200 and size == len(PAYLOAD)
    status, _ = probe_url(f"{server}/missing.zip")
    assert status == 404


def test_discover_parses_a_bucket_listing(server, capsys):
    keys = discover(prefix="", pattern="raw_data", bucket=f"{server}/listing")
    assert [k for k, _ in keys] == ["GDSC_release8.2/GDSC2_public_raw_data.csv.zip"]


def test_shipped_manifest_is_well_formed():
    sources = load_sources("configs/data_sources.yaml")
    for group, entries in sources.items():
        for name, spec in entries.items():
            assert "filename" in spec, f"{group}/{name} has no filename"
            urls = candidate_urls(spec)
            assert urls and all(u.startswith("https://") for u in urls), f"{group}/{name}"


def test_autofix_finds_the_real_key(server, tmp_path):
    """A rotted release path must be repairable from the bucket listing."""
    import yaml

    from hill.data.download import autofix_urls

    manifest = {
        "gdsc_raw": {
            "GDSC2": {
                "urls": [f"{server}/cancerrxgene/GDSC_release8.5/GDSC2_public_raw_data.csv.zip"],
                "filename": "GDSC2_public_raw_data.csv.zip",
                "match": r"GDSC2_public_raw_data.*\.zip$",
                "unzip": True,
            }
        }
    }
    out = tmp_path / "sources.yaml"
    out.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    found = autofix_urls(manifest, ["gdsc-raw"], bucket=f"{server}/listing", write_to=out)
    assert "gdsc-raw/GDSC2" in found
    assert found["gdsc-raw/GDSC2"][0].endswith("GDSC_release8.2/GDSC2_public_raw_data.csv.zip")

    written = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert written["gdsc_raw"]["GDSC2"]["urls"][0] == found["gdsc-raw/GDSC2"][0]
    assert len(written["gdsc_raw"]["GDSC2"]["urls"]) == 2, "the original URL is kept as a fallback"
