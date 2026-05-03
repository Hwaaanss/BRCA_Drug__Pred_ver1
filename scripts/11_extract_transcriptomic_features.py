#!/usr/bin/env python3
"""Extract transcriptomic feature matrix from TCGA-BRCA RNA-seq STAR-Counts."""
import os, tarfile, csv, io
from collections import defaultdict

RNASEQ_DIR = "./data/03_transcriptomic"
OUT_DIR = "./data/07_integrated"
MANIFEST = os.path.join(RNASEQ_DIR, "manifest_rnaseq.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# Read manifest to map file_id -> submitter_id
file_to_patient = {}
with open(MANIFEST) as f:
    reader = csv.DictReader(f)
    for row in reader:
        file_to_patient[row['file_id']] = row['submitter_id'][:12]  # patient-level

# Parse all tar.gz batches
print("Extracting TPM from RNA-seq batches...")
batch_files = sorted([f for f in os.listdir(RNASEQ_DIR) if f.startswith('rnaseq_batch_') and f.endswith('.tar.gz')])
print(f"Found {len(batch_files)} batch files")

# gene_id -> gene_name mapping (from first file)
gene_id_to_name = {}
# patient_id -> {gene_name: tpm}
patient_expression = {}

for bi, batch_file in enumerate(batch_files):
    batch_path = os.path.join(RNASEQ_DIR, batch_file)
    with tarfile.open(batch_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if not member.name.endswith('.tsv') or member.name == 'MANIFEST.txt':
                continue
            
            # Extract file_id from path (directory name)
            file_id = member.name.split('/')[0]
            patient_id = file_to_patient.get(file_id, None)
            if patient_id is None:
                continue
            
            f = tar.extractfile(member)
            if f is None:
                continue
            
            content = f.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            expression = {}
            for line in lines:
                if line.startswith('#') or line.startswith('N_') or line.startswith('gene_id'):
                    continue
                parts = line.split('\t')
                if len(parts) < 7:
                    continue
                gene_id = parts[0]
                gene_name = parts[1]
                gene_type = parts[2]
                
                # Only protein coding genes
                if gene_type != 'protein_coding':
                    continue
                
                try:
                    tpm = float(parts[6])  # tpm_unstranded column
                except (ValueError, IndexError):
                    continue
                
                gene_id_to_name[gene_id] = gene_name
                expression[gene_name] = tpm
            
            # If same patient has multiple samples, keep the one with higher total expression (primary tumor preferred)
            if patient_id not in patient_expression or sum(expression.values()) > sum(patient_expression[patient_id].values()):
                patient_expression[patient_id] = expression
    
    print(f"  Batch {bi+1}/{len(batch_files)}: {len(patient_expression)} patients so far")

print(f"\nTotal patients: {len(patient_expression)}")

# Select top 2000 most variable genes
import statistics

all_genes = set()
for exp in patient_expression.values():
    all_genes.update(exp.keys())

# Calculate variance for each gene
gene_variance = {}
patients_list = sorted(patient_expression.keys())
for gene in all_genes:
    values = [patient_expression[p].get(gene, 0) for p in patients_list]
    if len(set(values)) > 1:
        gene_variance[gene] = statistics.variance(values)

# Top 2000 by variance
top_genes = sorted(gene_variance.keys(), key=lambda g: -gene_variance[g])[:2000]
print(f"Selected top 2000 variable genes (variance range: {gene_variance[top_genes[0]]:.1f} - {gene_variance[top_genes[-1]]:.1f})")

# Build matrix
out_path = os.path.join(OUT_DIR, "X_transcriptomic.csv")
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['patient_id'] + top_genes)
    for patient in patients_list:
        row = [patient] + [f"{patient_expression[patient].get(g, 0):.4f}" for g in top_genes]
        writer.writerow(row)

print(f"Saved X_transcriptomic.csv: {len(patients_list)} patients x {len(top_genes)+1} columns")

# Also save full gene list for reference
with open(os.path.join(OUT_DIR, "gene_list_all_protein_coding.txt"), 'w') as f:
    for gene in sorted(all_genes):
        f.write(gene + '\n')
print(f"Saved gene list: {len(all_genes)} protein-coding genes")
