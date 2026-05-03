#!/usr/bin/env python3
"""Harmonize sample IDs across all modalities and create integrated overview."""
import pandas as pd
import os

OUT_DIR = "./data/07_integrated"
CLINICAL = "./data/01_clinical/TCGA_BRCA_clinical.csv"
DRUG_TREAT = "./data/01_clinical/TCGA_BRCA_drug_treatments.csv"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("MULTI-MODAL DATA HARMONIZATION")
print("=" * 60)

# 1. Load all feature matrices
print("\n--- Loading feature matrices ---")

clinical = pd.read_csv(CLINICAL)
clinical['patient_id'] = clinical['submitter_id']
print(f"Clinical: {len(clinical)} patients")

drug_treat = pd.read_csv(DRUG_TREAT)
drug_patients = set(drug_treat['submitter_id'].unique())
print(f"Drug treatments: {len(drug_treat)} records from {len(drug_patients)} patients")

genomic = pd.read_csv(os.path.join(OUT_DIR, "X_genomic.csv"))
genomic_patients = set(genomic['patient_id'])
print(f"Genomic: {len(genomic)} patients x {genomic.shape[1]-1} features")

transcriptomic = pd.read_csv(os.path.join(OUT_DIR, "X_transcriptomic.csv"))
trans_patients = set(transcriptomic['patient_id'])
print(f"Transcriptomic: {len(transcriptomic)} patients x {transcriptomic.shape[1]-1} features")

proteomic = pd.read_csv(os.path.join(OUT_DIR, "X_proteomic.csv"))
prot_patients = set(proteomic['patient_id'])
print(f"Proteomic: {len(proteomic)} patients x {proteomic.shape[1]-1} features")

# Clinical patient set
clinical_patients = set(clinical['patient_id'])

# 2. Drug treatment analysis
print("\n--- Drug treatment analysis ---")
print(f"Patients with drug treatment records: {len(drug_patients)}")
print(f"Unique drugs: {drug_treat['therapeutic_agents'].nunique()}")
print(f"\nTop 15 drugs:")
for drug, count in drug_treat['therapeutic_agents'].value_counts().head(15).items():
    print(f"  {drug}: {count} patients")

print(f"\nTreatment outcomes:")
for outcome, count in drug_treat['treatment_outcome'].value_counts().items():
    print(f"  {outcome}: {count}")

# 3. Set intersections
print("\n--- Sample overlap analysis ---")
all_patients = clinical_patients
print(f"Total TCGA-BRCA patients: {len(all_patients)}")

# Pairwise overlaps
sets = {
    'Clinical': clinical_patients,
    'Drug_treat': drug_patients,
    'Genomic': genomic_patients,
    'Transcriptomic': trans_patients,
    'Proteomic': prot_patients,
}

print("\nPairwise overlaps:")
names = list(sets.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        overlap = len(sets[names[i]] & sets[names[j]])
        print(f"  {names[i]} ∩ {names[j]}: {overlap}")

# Key intersections
core_3modal = genomic_patients & trans_patients & prot_patients
print(f"\nGenomic ∩ Transcriptomic ∩ Proteomic: {len(core_3modal)} patients")

core_3modal_drug = core_3modal & drug_patients
print(f"Genomic ∩ Transcriptomic ∩ Proteomic ∩ Drug_treatment: {len(core_3modal_drug)} patients")

core_2modal = genomic_patients & trans_patients
print(f"Genomic ∩ Transcriptomic (for oncoPredict): {len(core_2modal)} patients")

core_2modal_drug = core_2modal & drug_patients
print(f"Genomic ∩ Transcriptomic ∩ Drug_treatment: {len(core_2modal_drug)} patients")

# 4. Create harmonized master table
print("\n--- Creating master sample table ---")
master_rows = []
for pid in sorted(all_patients):
    row = {
        'patient_id': pid,
        'has_clinical': pid in clinical_patients,
        'has_drug_treatment': pid in drug_patients,
        'has_genomic': pid in genomic_patients,
        'has_transcriptomic': pid in trans_patients,
        'has_proteomic': pid in prot_patients,
        'has_morphology': False,  # WSI still downloading
    }
    # Count modalities
    row['n_modalities'] = sum([row['has_genomic'], row['has_transcriptomic'], 
                                row['has_proteomic'], row['has_drug_treatment']])
    master_rows.append(row)

master = pd.DataFrame(master_rows)
master.to_csv(os.path.join(OUT_DIR, "sample_master_table.csv"), index=False)

print(f"\nModality coverage distribution:")
for n in sorted(master['n_modalities'].unique()):
    count = (master['n_modalities'] == n).sum()
    print(f"  {n} modalities: {count} patients")

# 5. Create merged feature matrix for patients with all available modalities
print("\n--- Creating merged feature matrix (3-modal core) ---")
core_patients = sorted(core_3modal)

genomic_core = genomic[genomic['patient_id'].isin(core_patients)].set_index('patient_id')
trans_core = transcriptomic[transcriptomic['patient_id'].isin(core_patients)].set_index('patient_id')
prot_core = proteomic[proteomic['patient_id'].isin(core_patients)].set_index('patient_id')

# Add prefixes to avoid column name conflicts
genomic_core = genomic_core.add_prefix('GEN_')
trans_core = trans_core.add_prefix('RNA_')
prot_core = prot_core.add_prefix('PROT_')

merged = pd.concat([genomic_core, trans_core, prot_core], axis=1, join='inner')
merged.index.name = 'patient_id'
merged.to_csv(os.path.join(OUT_DIR, "X_merged_3modal.csv"))

print(f"Merged 3-modal matrix: {merged.shape[0]} patients x {merged.shape[1]} features")
print(f"  Genomic features: {sum(1 for c in merged.columns if c.startswith('GEN_'))}")
print(f"  Transcriptomic features: {sum(1 for c in merged.columns if c.startswith('RNA_'))}")
print(f"  Proteomic features: {sum(1 for c in merged.columns if c.startswith('PROT_'))}")

# 6. Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total TCGA-BRCA patients: {len(all_patients)}")
print(f"With drug treatment data: {len(drug_patients)}")
print(f"3-modal core (GEN+RNA+PROT): {len(core_3modal)} patients")
print(f"3-modal + drug treatment: {len(core_3modal_drug)} patients")
print(f"Merged feature dimensions: {merged.shape}")
print(f"\nGDSC cell lines (separate): 51 breast cancer cell lines x 542 drugs")
print(f"oncoPredict can predict IC50 for {len(core_2modal)} patients using RNA-seq")
print(f"\nFiles saved in {OUT_DIR}:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f}: {size/1024/1024:.1f} MB" if size > 1024*1024 else f"  {f}: {size/1024:.1f} KB")
