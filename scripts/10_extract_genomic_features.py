#!/usr/bin/env python3
"""Extract genomic feature matrix from TCGA-BRCA MAF files."""
import os, gzip, glob, csv
from collections import defaultdict

MAF_DIR = "./data/02_genomic/mutations"
OUT_DIR = "./data/07_integrated"
os.makedirs(OUT_DIR, exist_ok=True)

# Parse all MAF files
print("Parsing MAF files...")
# sample -> {gene -> set of variant classifications}
sample_mutations = defaultdict(lambda: defaultdict(set))
sample_total_mutations = defaultdict(int)

# Nonsynonymous variant types for binary matrix
NONSYNONYMOUS = {
    'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins',
    'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site', 'Translation_Start_Site',
    'Nonstop_Mutation'
}

maf_files = sorted(glob.glob(os.path.join(MAF_DIR, "*.maf.gz")))
print(f"Found {len(maf_files)} MAF files")

for i, maf_path in enumerate(maf_files):
    try:
        with gzip.open(maf_path, 'rt') as f:
            header = None
            for line in f:
                if line.startswith('#'):
                    continue
                if header is None:
                    header = line.strip().split('\t')
                    hugo_idx = header.index('Hugo_Symbol')
                    barcode_idx = header.index('Tumor_Sample_Barcode')
                    vc_idx = header.index('Variant_Classification')
                    continue
                fields = line.strip().split('\t')
                gene = fields[hugo_idx]
                barcode = fields[barcode_idx]
                vc = fields[vc_idx]
                
                # Extract patient ID (first 12 chars of TCGA barcode)
                patient_id = barcode[:12]
                
                sample_total_mutations[patient_id] += 1
                if vc in NONSYNONYMOUS:
                    sample_mutations[patient_id][gene].add(vc)
    except Exception as e:
        print(f"  Error in {os.path.basename(maf_path)}: {e}")
    
    if (i+1) % 100 == 0:
        print(f"  Processed {i+1}/{len(maf_files)} files")

print(f"Parsed {len(sample_mutations)} patients")

# Identify frequently mutated genes (>= 2% of samples)
gene_freq = defaultdict(int)
for patient, genes in sample_mutations.items():
    for gene in genes:
        gene_freq[gene] += 1

n_samples = len(sample_mutations)
min_freq = max(2, int(n_samples * 0.02))  # at least 2% or 2 samples
top_genes = sorted([g for g, c in gene_freq.items() if c >= min_freq],
                   key=lambda g: -gene_freq[g])
print(f"Genes mutated in >= {min_freq} samples: {len(top_genes)}")
print(f"Top 20: {[(g, gene_freq[g]) for g in top_genes[:20]]}")

# Build binary mutation matrix
patients = sorted(sample_mutations.keys())
rows = []
for patient in patients:
    row = {'patient_id': patient, 'TMB': sample_total_mutations[patient]}
    for gene in top_genes:
        row[gene] = 1 if gene in sample_mutations[patient] else 0
    rows.append(row)

# Save
out_path = os.path.join(OUT_DIR, "X_genomic.csv")
fieldnames = ['patient_id', 'TMB'] + top_genes
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved X_genomic.csv: {len(rows)} patients x {len(fieldnames)} features")
