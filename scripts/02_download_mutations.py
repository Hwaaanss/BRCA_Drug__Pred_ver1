#!/usr/bin/env python3
"""Download TCGA-BRCA somatic mutation data (MAF) from GDC."""
import requests, json, os

OUT_DIR = "./data/02_genomic/mutations"
BASE = "https://api.gdc.cancer.gov"
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_all_hits(filters, fields, page_size=1000):
    """Fetch every matching GDC file hit using explicit pagination."""
    hits = []
    offset = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "size": page_size,
            "from": offset,
            "format": "JSON",
        }
        resp = requests.get(f"{BASE}/files", params=params, timeout=120)
        resp.raise_for_status()
        batch = resp.json().get("data", {}).get("hits", [])
        if not batch:
            break
        hits.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return hits


def download_file(file_id, file_name, timeout=600):
    out_path = os.path.join(OUT_DIR, file_name)
    part_path = out_path + ".part"

    if os.path.exists(out_path) or os.path.exists(out_path.replace('.gz', '')):
        print(f"  {file_name} already exists, skipping")
        return

    print(f"Downloading {file_name}...")
    with requests.get(f"{BASE}/data/{file_id}", timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(part_path, out_path)
    print(f"  Saved ({os.path.getsize(out_path)/1024/1024:.1f} MB)")

# Query for open-access MAF files (Masked Somatic Mutation)
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Simple Nucleotide Variation"}},
        {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}

fields = "file_id,file_name,file_size,data_type,analysis.workflow_type"
hits = fetch_all_hits(filters, fields)
print(f"Found {len(hits)} MAF files")

for h in hits:
    print(f"  {h['file_name']} ({h['file_size']/1024/1024:.1f} MB) - {h.get('analysis',{}).get('workflow_type','N/A')}")

for h in hits:
    download_file(h["file_id"], h["file_name"], timeout=300)

# Also download aggregated mutation data via MC3 or similar
print("\n=== Downloading TCGA MC3 public MAF (pan-cancer, will filter BRCA) ===")
# GDC aggregated somatic mutations for TCGA-BRCA
agg_filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Simple Nucleotide Variation"}},
        {"op": "=", "content": {"field": "data_type", "value": "Aggregated Somatic Mutation"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}
agg_hits = fetch_all_hits(agg_filters, fields)
print(f"Found {len(agg_hits)} aggregated MAF files")
for h in agg_hits:
    print(f"  {h['file_name']} ({h['file_size']/1024/1024:.1f} MB)")
    download_file(h["file_id"], h["file_name"], timeout=600)

print("\nDone!")
