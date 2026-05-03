#!/usr/bin/env python3
"""Download TCGA-BRCA RPPA proteomic data from GDC and TCPA."""
import requests, json, os, csv

OUT_DIR = "./data/04_proteomic"
BASE = "https://api.gdc.cancer.gov"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Try GDC for RPPA data
print("=== Querying GDC for RPPA data ===")
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "data_category", "value": "Proteome Profiling"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,file_size,data_type,analysis.workflow_type",
    "size": 2000,
    "format": "JSON"
}
resp = requests.get(f"{BASE}/files", params=params)
resp.raise_for_status()
hits = resp.json()["data"]["hits"]
total = resp.json()["data"]["pagination"]["total"]
print(f"Found {total} proteome profiling files on GDC")

if hits:
    for h in hits[:5]:
        print(f"  {h['file_name']} ({h['file_size']/1024:.1f} KB)")
    
    # Download all
    file_ids = [h["file_id"] for h in hits]
    
    # Batch download
    print(f"\nDownloading {len(file_ids)} RPPA files...")
    tar_path = os.path.join(OUT_DIR, "rppa_batch.tar.gz")
    if not os.path.exists(tar_path):
        payload = {"ids": file_ids[:500]}  # GDC max
        resp = requests.post(f"{BASE}/data", json=payload, 
                           headers={"Content-Type": "application/json"}, timeout=300)
        resp.raise_for_status()
        with open(tar_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved ({len(resp.content)/1024/1024:.1f} MB)")

# 2. Download from TCPA (The Cancer Proteome Atlas) - direct link
print("\n=== Downloading TCPA RPPA data ===")
tcpa_url = "https://tcpaportal.org/tcpa/download/TCGA-BRCA-L4.csv"
try:
    resp = requests.get(tcpa_url, timeout=60)
    if resp.status_code == 200:
        with open(os.path.join(OUT_DIR, "TCPA_BRCA_RPPA_L4.csv"), "wb") as f:
            f.write(resp.content)
        print(f"  Saved TCPA L4 data ({len(resp.content)/1024:.1f} KB)")
    else:
        print(f"  TCPA direct download returned {resp.status_code}, trying alternative...")
        # Try the pan-cancer RPPA file
        alt_url = "https://tcpaportal.org/tcpa/download/TCGA-ALL-L4.csv"
        resp2 = requests.get(alt_url, timeout=60)
        if resp2.status_code == 200:
            with open(os.path.join(OUT_DIR, "TCPA_pancan_RPPA_L4.csv"), "wb") as f:
                f.write(resp2.content)
            print(f"  Saved TCPA pan-cancer L4 ({len(resp2.content)/1024:.1f} KB)")
        else:
            print(f"  Alternative also failed: {resp2.status_code}")
except Exception as e:
    print(f"  TCPA download error: {e}")

# 3. Try cBioPortal for RPPA
print("\n=== Trying cBioPortal for RPPA ===")
try:
    cbio_url = "https://www.cbioportal.org/api/molecular-profiles/brca_tcga_rppa/molecular-data?sampleListId=brca_tcga_all"
    resp = requests.get(cbio_url, timeout=30, headers={"Accept": "application/json"})
    if resp.status_code == 200:
        data = resp.json()
        with open(os.path.join(OUT_DIR, "cBioPortal_BRCA_RPPA.json"), "w") as f:
            json.dump(data, f)
        print(f"  Saved cBioPortal RPPA data ({len(data)} records)")
    else:
        print(f"  cBioPortal returned {resp.status_code}")
except Exception as e:
    print(f"  cBioPortal error: {e}")

print("\nDone!")
