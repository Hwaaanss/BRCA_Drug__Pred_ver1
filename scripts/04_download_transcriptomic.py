#!/usr/bin/env python3
"""Download TCGA-BRCA RNA-seq gene expression data from GDC."""
import requests, json, os, gzip, shutil

OUT_DIR = "./data/03_transcriptomic"
BASE = "https://api.gdc.cancer.gov"
os.makedirs(OUT_DIR, exist_ok=True)

# Query for STAR-Counts (current GDC pipeline)
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
        {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}

# Get file list
params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,file_size,cases.submitter_id,cases.samples.sample_type",
    "size": 2000,
    "format": "JSON"
}
resp = requests.get(f"{BASE}/files", params=params)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]
total = resp.json()["data"]["pagination"]["total"]
print(f"Found {total} RNA-seq STAR-Counts files")

# Save file manifest
file_ids = [h["file_id"] for h in hits]
manifest = []
for h in hits:
    cases = h.get("cases", [{}])
    sid = cases[0].get("submitter_id", "unknown") if cases else "unknown"
    manifest.append({"file_id": h["file_id"], "file_name": h["file_name"], 
                      "submitter_id": sid, "file_size": h["file_size"]})

import csv
with open(os.path.join(OUT_DIR, "manifest_rnaseq.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_id","file_name","submitter_id","file_size"])
    writer.writeheader()
    writer.writerows(manifest)
print(f"Saved manifest with {len(manifest)} files")

# Download via GDC data endpoint (batch download as tar.gz)
# For bulk, we download in batches of 50
batch_size = 50
all_downloaded = 0

for i in range(0, len(file_ids), batch_size):
    batch = file_ids[i:i+batch_size]
    batch_num = i // batch_size + 1
    tar_path = os.path.join(OUT_DIR, f"rnaseq_batch_{batch_num}.tar.gz")
    
    if os.path.exists(tar_path):
        print(f"  Batch {batch_num} already exists, skipping")
        all_downloaded += len(batch)
        continue
    
    print(f"Downloading batch {batch_num} ({len(batch)} files, {all_downloaded}/{len(file_ids)})...")
    payload = {"ids": batch}
    resp = requests.post(f"{BASE}/data", json=payload, headers={"Content-Type": "application/json"}, timeout=600)
    resp.raise_for_status()
    
    with open(tar_path, "wb") as f:
        f.write(resp.content)
    all_downloaded += len(batch)
    print(f"  Saved batch {batch_num} ({len(resp.content)/1024/1024:.1f} MB)")

print(f"\nTotal downloaded: {all_downloaded} files")
print("Done!")
