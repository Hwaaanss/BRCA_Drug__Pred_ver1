#!/usr/bin/env python3
"""Parallel download of WSI files for 3-modal intersection patients."""
import os
import csv
import sys
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WSI_DIR = "./data/05_morphology/wsi"
TARGET_CSV = "./data/05_morphology/wsi_target_3modal.csv"
FEATURE_DIR = "./data/05_morphology/features"
BASE = "https://api.gdc.cancer.gov"
N_WORKERS = 6  # parallel threads
os.makedirs(WSI_DIR, exist_ok=True)
COMPLETE_RATIO = 0.99


def read_patient_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required feature file: {path}\n"
            "Run scripts/10_extract_genomic_features.py, "
            "scripts/11_extract_transcriptomic_features.py, and "
            "scripts/12_extract_proteomic_features.py first."
        )
    with open(path) as f:
        reader = csv.DictReader(f)
        if "patient_id" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a patient_id column.")
        return {row["patient_id"] for row in reader if row.get("patient_id")}


def ensure_target_csv():
    if os.path.exists(TARGET_CSV):
        return

    print(f"{TARGET_CSV} not found. Creating WSI target manifest from 3-modal patients...")
    integrated = "./data/07_integrated"
    gen_ids = read_patient_ids(os.path.join(integrated, "X_genomic.csv"))
    tra_ids = read_patient_ids(os.path.join(integrated, "X_transcriptomic.csv"))
    pro_ids = read_patient_ids(os.path.join(integrated, "X_proteomic.csv"))
    patient_ids = sorted(gen_ids & tra_ids & pro_ids)
    if not patient_ids:
        raise RuntimeError("No common genomic/transcriptomic/proteomic patients found.")

    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "cases.submitter_id", "value": patient_ids}},
        ],
    }
    fields = "file_id,file_name,file_size,cases.submitter_id"

    session = create_session()
    rows = {}
    page_size = 1000
    offset = 0
    while True:
        resp = session.post(
            f"{BASE}/files",
            json={
                "filters": filters,
                "fields": fields,
                "format": "JSON",
                "size": page_size,
                "from": offset,
            },
            timeout=120,
        )
        resp.raise_for_status()
        hits = resp.json().get("data", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            cases = hit.get("cases") or []
            if not cases:
                continue
            rows[hit["file_id"]] = {
                "patient_id": cases[0].get("submitter_id", ""),
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size_MB": f"{hit.get('file_size', 0) / (1024 ** 2):.3f}",
            }
        offset += page_size

    if not rows:
        raise RuntimeError("GDC returned no SVS slide images for the 3-modal patient set.")

    os.makedirs(os.path.dirname(TARGET_CSV), exist_ok=True)
    fieldnames = ["patient_id", "file_id", "file_name", "file_size_MB"]
    ordered = sorted(rows.values(), key=lambda r: (r["patient_id"], r["file_name"]))
    with open(TARGET_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    total_gb = sum(float(r["file_size_MB"]) for r in ordered) / 1024
    n_patients = len({r["patient_id"] for r in ordered})
    print(f"Created {TARGET_CSV}: {len(ordered)} files, {n_patients} patients, {total_gb:.1f} GB")


def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=N_WORKERS + 2))
    return session

def is_complete_file(path, expected_mb):
    if not os.path.exists(path):
        return False
    actual_mb = os.path.getsize(path) / (1024**2)
    return actual_mb >= expected_mb * COMPLETE_RATIO

def download_one(file_id, file_name, file_size_mb):
    out_path = os.path.join(WSI_DIR, file_name)
    part_path = out_path + ".part"
    if is_complete_file(out_path, file_size_mb):
        actual = os.path.getsize(out_path) / (1024**2)
        return file_name, "skip", actual

    if os.path.exists(part_path):
        os.remove(part_path)

    session = create_session()
    try:
        resp = session.get(f"{BASE}/data/{file_id}", timeout=600, stream=True)
        resp.raise_for_status()
        with open(part_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=131072):
                if chunk:
                    f.write(chunk)
        actual = os.path.getsize(part_path) / (1024**2)
        if actual < file_size_mb * COMPLETE_RATIO:
            os.remove(part_path)
            return file_name, f"error: incomplete {actual:.1f}/{file_size_mb:.1f}MB", 0
        os.replace(part_path, out_path)
        actual = os.path.getsize(out_path) / (1024**2)
        return file_name, "ok", actual
    except Exception as e:
        if os.path.exists(part_path):
            os.remove(part_path)
        return file_name, f"error: {e}", 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N remaining files.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many remaining files before applying --limit.")
    parser.add_argument("--max_gb", type=float, default=None,
                        help="Download complete patient groups up to this approximate GB limit.")
    parser.add_argument("--redownload_featured", action="store_true",
                        help="Also download slides for patients that already have a .pt feature file.")
    args = parser.parse_args()

    ensure_target_csv()

    # Load target list
    with open(TARGET_CSV) as f:
        reader = csv.DictReader(f)
        files = list(reader)

    featured = {
        f.replace(".pt", "")
        for f in os.listdir(FEATURE_DIR)
        if f.endswith(".pt")
    } if os.path.isdir(FEATURE_DIR) else set()

    patient_files = {}
    for f in files:
        patient_files.setdefault(f["patient_id"], []).append(f)

    patient_todo = []
    skipped_featured = 0
    complete_files = 0
    for pid, rows in patient_files.items():
        if pid in featured and not args.redownload_featured:
            skipped_featured += len(rows)
            continue

        missing_or_incomplete = [
            r for r in rows
            if not is_complete_file(os.path.join(WSI_DIR, r["file_name"]), float(r["file_size_MB"]))
        ]
        complete_files += len(rows) - len(missing_or_incomplete)
        if missing_or_incomplete:
            patient_todo.append((pid, missing_or_incomplete))

    total_remaining = sum(len(rows) for _, rows in patient_todo)
    if args.offset:
        patient_todo = patient_todo[args.offset:]

    selected = []
    selected_mb = 0.0
    max_mb = args.max_gb * 1024 if args.max_gb is not None else None
    for pid, rows in patient_todo:
        patient_mb = sum(float(f["file_size_MB"]) for f in rows)
        if selected and max_mb is not None and selected_mb + patient_mb > max_mb:
            break
        selected.extend(rows)
        selected_mb += patient_mb

    todo = selected
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"Total target: {len(files)}, complete files: {complete_files}, "
          f"feature-ready skipped: {skipped_featured}, remaining files: {total_remaining}")
    if args.offset or args.limit is not None or args.max_gb is not None:
        print(f"Current batch: offset={args.offset}, limit={args.limit}, max_gb={args.max_gb}, files={len(todo)}")
    print(f"Estimated remaining size: {sum(float(f['file_size_MB']) for f in todo)/1024:.1f} GB")
    print(f"Using {N_WORKERS} parallel workers")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    completed = 0
    total_mb = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(download_one, f['file_id'], f['file_name'], float(f['file_size_MB'])): f
            for f in todo
        }

        for future in as_completed(futures):
            fname, status, size_mb = future.result()
            completed += 1
            total_mb += size_mb

            elapsed = time.time() - start_time
            speed = total_mb / elapsed if elapsed > 0 else 0
            remaining_mb = sum(float(f['file_size_MB']) for f in todo) - total_mb
            eta_min = remaining_mb / speed / 60 if speed > 0 else 0

            if completed % 10 == 0 or status != "ok":
                print(f"[{completed}/{len(todo)}] {fname[:50]:50s} | {status:5s} | "
                      f"{size_mb:.0f}MB | {speed:.1f}MB/s | ETA: {eta_min:.0f}min")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone! {completed} files, {total_mb/1024:.1f} GB in {elapsed/3600:.1f} hours")
    print(f"Final count: {len(os.listdir(WSI_DIR))} files in {WSI_DIR}")

if __name__ == '__main__':
    main()
