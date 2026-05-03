#!/usr/bin/env python3
"""Download GDSC drug sensitivity data for breast cancer cell lines."""
import requests, os

OUT_DIR = "./data/06_drug_response/gdsc"
os.makedirs(OUT_DIR, exist_ok=True)

# GDSC2 fitted dose-response data (IC50, AUC)
urls = {
    "GDSC2_fitted_dose_response": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx",
    "GDSC1_fitted_dose_response": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
}

for name, url in urls.items():
    out_path = os.path.join(OUT_DIR, f"{name}.xlsx")
    if os.path.exists(out_path):
        print(f"  {name} already exists, skipping")
        continue
    print(f"Downloading {name}...")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved: {out_path} ({len(resp.content)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"  Error downloading {name}: {e}")
        # Try alternative URL format
        alt_url = url.replace("27Oct23", "25Jul22").replace("release8.5", "release8.4")
        print(f"  Trying alternative: {alt_url}")
        try:
            resp = requests.get(alt_url, timeout=120)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  Saved: {out_path} ({len(resp.content)/1024/1024:.1f} MB)")
        except Exception as e2:
            print(f"  Alternative also failed: {e2}")

# Cell line annotations
anno_url = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx"
out_path = os.path.join(OUT_DIR, "Cell_Lines_Details.xlsx")
print(f"Downloading cell line annotations...")
try:
    resp = requests.get(anno_url, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved: {out_path} ({len(resp.content)/1024/1024:.1f} MB)")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone!")
