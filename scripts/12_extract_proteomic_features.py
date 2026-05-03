#!/usr/bin/env python3
"""Extract proteomic feature matrix from TCGA-BRCA RPPA data."""
import os, tarfile, csv
from collections import defaultdict

RPPA_PATH = "./data/04_proteomic/rppa_batch.tar.gz"
OUT_DIR = "./data/07_integrated"
os.makedirs(OUT_DIR, exist_ok=True)

print("Extracting RPPA data...")

# patient_id -> {protein: expression}
patient_protein = {}
all_proteins = set()

with tarfile.open(RPPA_PATH, 'r:gz') as tar:
    members = [m for m in tar.getmembers() if m.name.endswith('.tsv') and m.name != 'MANIFEST.txt']
    print(f"Found {len(members)} RPPA files")
    
    for member in members:
        # Extract patient ID from filename: TCGA-XX-XXXX-01A-...
        fname = os.path.basename(member.name)
        parts = fname.split('-')
        if len(parts) >= 3 and parts[0] == 'TCGA':
            patient_id = '-'.join(parts[:3])
        else:
            continue
        
        f = tar.extractfile(member)
        if f is None:
            continue
        
        content = f.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        expression = {}
        for line in lines[1:]:  # skip header
            fields = line.split('\t')
            if len(fields) >= 6:
                protein = fields[4]  # peptide_target
                try:
                    value = float(fields[5])  # protein_expression
                except ValueError:
                    continue
                expression[protein] = value
                all_proteins.add(protein)
        
        # Keep first occurrence per patient (prefer 01A = primary tumor)
        if patient_id not in patient_protein:
            patient_protein[patient_id] = expression

print(f"Patients: {len(patient_protein)}, Proteins: {len(all_proteins)}")

# Build matrix
proteins = sorted(all_proteins)
patients = sorted(patient_protein.keys())

out_path = os.path.join(OUT_DIR, "X_proteomic.csv")
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['patient_id'] + proteins)
    for patient in patients:
        row = [patient] + [f"{patient_protein[patient].get(p, 0):.6f}" for p in proteins]
        writer.writerow(row)

print(f"Saved X_proteomic.csv: {len(patients)} patients x {len(proteins)+1} columns")
