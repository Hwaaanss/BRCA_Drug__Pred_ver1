#!/usr/bin/env python3
"""Download TCGA-BRCA clinical and drug treatment data from GDC API."""
import requests, json, os, csv

BASE = "https://api.gdc.cancer.gov"
OUT_DIR = "./data/01_clinical"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Clinical data (demographics, diagnoses, treatments, exposures)
def download_clinical():
    cases = []
    size = 500
    offset = 0
    
    while True:
        params = {
            "filters": json.dumps({
                "op": "=",
                "content": {"field": "project.project_id", "value": "TCGA-BRCA"}
            }),
            "fields": ",".join([
                "case_id", "submitter_id",
                "demographic.gender", "demographic.race", "demographic.ethnicity",
                "demographic.vital_status", "demographic.days_to_death", "demographic.year_of_birth",
                "demographic.age_at_index",
                "diagnoses.primary_diagnosis", "diagnoses.tumor_stage", "diagnoses.tumor_grade",
                "diagnoses.age_at_diagnosis", "diagnoses.days_to_last_follow_up",
                "diagnoses.morphology", "diagnoses.site_of_resection_or_biopsy",
                "diagnoses.tissue_or_organ_of_origin",
                "diagnoses.treatments.treatment_type", "diagnoses.treatments.therapeutic_agents",
                "diagnoses.treatments.treatment_outcome", "diagnoses.treatments.days_to_treatment_start",
                "diagnoses.treatments.days_to_treatment_end",
                "exposures.alcohol_history", "exposures.bmi",
                "project.project_id"
            ]),
            "size": size,
            "from": offset,
            "format": "JSON"
        }
        
        resp = requests.get(f"{BASE}/cases", params=params)
        resp.raise_for_status()
        data = resp.json()
        hits = data["data"]["hits"]
        if not hits:
            break
        cases.extend(hits)
        offset += size
        print(f"  Fetched {len(cases)} / {data['data']['pagination']['total']} cases")
        if len(cases) >= data["data"]["pagination"]["total"]:
            break
    
    # Save raw JSON
    with open(os.path.join(OUT_DIR, "TCGA_BRCA_clinical_raw.json"), "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {len(cases)} cases to clinical_raw.json")
    
    # Flatten to CSV
    rows = []
    for c in cases:
        base = {
            "case_id": c.get("case_id"),
            "submitter_id": c.get("submitter_id"),
        }
        demo = c.get("demographic", {}) or {}
        base.update({
            "gender": demo.get("gender"),
            "race": demo.get("race"),
            "ethnicity": demo.get("ethnicity"),
            "vital_status": demo.get("vital_status"),
            "days_to_death": demo.get("days_to_death"),
            "age_at_index": demo.get("age_at_index"),
        })
        
        diags = c.get("diagnoses", []) or []
        if diags:
            d = diags[0]
            base.update({
                "primary_diagnosis": d.get("primary_diagnosis"),
                "tumor_stage": d.get("tumor_stage"),
                "tumor_grade": d.get("tumor_grade"),
                "age_at_diagnosis": d.get("age_at_diagnosis"),
                "days_to_last_follow_up": d.get("days_to_last_follow_up"),
            })
            treatments = d.get("treatments", []) or []
            for i, t in enumerate(treatments):
                base[f"treatment_{i+1}_type"] = t.get("treatment_type")
                base[f"treatment_{i+1}_agents"] = t.get("therapeutic_agents")
                base[f"treatment_{i+1}_outcome"] = t.get("treatment_outcome")
        
        rows.append(base)
    
    # Write CSV
    if rows:
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(os.path.join(OUT_DIR, "TCGA_BRCA_clinical.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to clinical.csv with {len(all_keys)} columns")
    
    return cases

# 2. Drug treatment-specific query
def download_drug_treatments():
    """Also get treatment info from diagnoses.treatments endpoint."""
    params = {
        "filters": json.dumps({
            "op": "=",
            "content": {"field": "project.project_id", "value": "TCGA-BRCA"}
        }),
        "fields": "submitter_id,diagnoses.treatments.treatment_type,diagnoses.treatments.therapeutic_agents,diagnoses.treatments.treatment_outcome,diagnoses.treatments.treatment_intent_type,diagnoses.treatments.days_to_treatment_start",
        "size": 2000,
        "format": "JSON"
    }
    resp = requests.get(f"{BASE}/cases", params=params)
    resp.raise_for_status()
    data = resp.json()["data"]["hits"]
    
    drug_rows = []
    for c in data:
        sid = c.get("submitter_id")
        for diag in (c.get("diagnoses") or []):
            for t in (diag.get("treatments") or []):
                if t.get("therapeutic_agents"):
                    drug_rows.append({
                        "submitter_id": sid,
                        "treatment_type": t.get("treatment_type"),
                        "therapeutic_agents": t.get("therapeutic_agents"),
                        "treatment_outcome": t.get("treatment_outcome"),
                        "treatment_intent": t.get("treatment_intent_type"),
                        "days_to_treatment_start": t.get("days_to_treatment_start"),
                    })
    
    with open(os.path.join(OUT_DIR, "TCGA_BRCA_drug_treatments.csv"), "w", newline="") as f:
        if drug_rows:
            writer = csv.DictWriter(f, fieldnames=drug_rows[0].keys())
            writer.writeheader()
            writer.writerows(drug_rows)
    print(f"Saved {len(drug_rows)} drug treatment records")
    return drug_rows

if __name__ == "__main__":
    print("=== Downloading TCGA-BRCA Clinical Data ===")
    download_clinical()
    print("\n=== Downloading Drug Treatment Records ===")
    download_drug_treatments()
    print("\nDone!")
