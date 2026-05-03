#!/usr/bin/env python3
"""Extract GDSC drug response data for breast cancer cell lines."""
import pandas as pd
import os

OUT_DIR = "./data/07_integrated"
GDSC_DIR = "./data/06_drug_response/gdsc"
os.makedirs(OUT_DIR, exist_ok=True)

# Read GDSC1 and GDSC2
print("Reading GDSC data...")
gdsc1 = pd.read_excel(os.path.join(GDSC_DIR, "GDSC1_fitted_dose_response.xlsx"))
gdsc2 = pd.read_excel(os.path.join(GDSC_DIR, "GDSC2_fitted_dose_response.xlsx"))

print(f"GDSC1: {gdsc1.shape}, columns: {list(gdsc1.columns)}")
print(f"GDSC2: {gdsc2.shape}, columns: {list(gdsc2.columns)}")

# Identify breast cancer cell lines
# Check TCGA classification column
for col in gdsc1.columns:
    if 'tcga' in col.lower() or 'tissue' in col.lower() or 'cancer' in col.lower():
        print(f"  Column '{col}': {gdsc1[col].nunique()} unique values")
        if gdsc1[col].nunique() < 50:
            brca_vals = [v for v in gdsc1[col].unique() if v and 'brca' in str(v).lower() or 'breast' in str(v).lower()]
            print(f"    Breast-related: {brca_vals}")

# Filter breast cancer
brca_cols_gdsc1 = [c for c in gdsc1.columns if 'tcga' in c.lower()]
if brca_cols_gdsc1:
    tcga_col = brca_cols_gdsc1[0]
    brca1 = gdsc1[gdsc1[tcga_col].str.contains('BRCA', case=False, na=False)]
    print(f"\nGDSC1 BRCA: {brca1.shape[0]} records, {brca1['CELL_LINE_NAME'].nunique()} cell lines, {brca1['DRUG_NAME'].nunique()} drugs")
else:
    # Try tissue column
    tissue_cols = [c for c in gdsc1.columns if 'tissue' in c.lower()]
    if tissue_cols:
        tcga_col = tissue_cols[0]
        brca1 = gdsc1[gdsc1[tcga_col].str.contains('breast', case=False, na=False)]
        print(f"\nGDSC1 Breast: {brca1.shape[0]} records")
    else:
        brca1 = gdsc1
        print("No tissue/TCGA column found, using all data")

brca_cols_gdsc2 = [c for c in gdsc2.columns if 'tcga' in c.lower()]
if brca_cols_gdsc2:
    tcga_col2 = brca_cols_gdsc2[0]
    brca2 = gdsc2[gdsc2[tcga_col2].str.contains('BRCA', case=False, na=False)]
    print(f"GDSC2 BRCA: {brca2.shape[0]} records, {brca2['CELL_LINE_NAME'].nunique()} cell lines, {brca2['DRUG_NAME'].nunique()} drugs")
else:
    tissue_cols2 = [c for c in gdsc2.columns if 'tissue' in c.lower()]
    if tissue_cols2:
        tcga_col2 = tissue_cols2[0]
        brca2 = gdsc2[gdsc2[tcga_col2].str.contains('breast', case=False, na=False)]
    else:
        brca2 = gdsc2

# Combine GDSC1 and GDSC2 (GDSC2 preferred for overlapping drug-cell pairs)
combined = pd.concat([brca1.assign(source='GDSC1'), brca2.assign(source='GDSC2')], ignore_index=True)
print(f"\nCombined BRCA: {combined.shape[0]} records")

# Save full drug response data
combined.to_csv(os.path.join(OUT_DIR, "GDSC_BRCA_drug_response.csv"), index=False)

# Create IC50 matrix (cell line x drug) using LN_IC50
ic50_col = [c for c in combined.columns if 'ic50' in c.lower()][0]
print(f"Using IC50 column: {ic50_col}")

# Pivot: prefer GDSC2, fallback to GDSC1
gdsc2_pivot = combined[combined['source']=='GDSC2'].pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME', values=ic50_col, aggfunc='first')
gdsc1_pivot = combined[combined['source']=='GDSC1'].pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME', values=ic50_col, aggfunc='first')

# Combine: GDSC2 takes priority
ic50_matrix = gdsc2_pivot.combine_first(gdsc1_pivot)
print(f"IC50 matrix: {ic50_matrix.shape[0]} cell lines x {ic50_matrix.shape[1]} drugs")

ic50_matrix.to_csv(os.path.join(OUT_DIR, "GDSC_BRCA_IC50_matrix.csv"))

# Also save AUC matrix
auc_col = [c for c in combined.columns if c == 'AUC' or 'auc' in c.lower()]
if auc_col:
    auc_col = auc_col[0]
    gdsc2_auc = combined[combined['source']=='GDSC2'].pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME', values=auc_col, aggfunc='first')
    gdsc1_auc = combined[combined['source']=='GDSC1'].pivot_table(
        index='CELL_LINE_NAME', columns='DRUG_NAME', values=auc_col, aggfunc='first')
    auc_matrix = gdsc2_auc.combine_first(gdsc1_auc)
    auc_matrix.to_csv(os.path.join(OUT_DIR, "GDSC_BRCA_AUC_matrix.csv"))
    print(f"AUC matrix: {auc_matrix.shape[0]} cell lines x {auc_matrix.shape[1]} drugs")

# Summary
print(f"\nTop 10 most tested drugs:")
drug_counts = combined['DRUG_NAME'].value_counts().head(10)
for drug, count in drug_counts.items():
    print(f"  {drug}: {count} cell lines tested")

print(f"\nCell lines: {sorted(combined['CELL_LINE_NAME'].unique())[:10]}...")
print(f"\nDone!")
