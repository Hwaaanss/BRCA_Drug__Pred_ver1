# PathOmicDRP

**PathOmicDRP: Integrating Multi-Omics and Histopathology through Cross-Attention Networks for Cancer Drug Response Prediction**

## Overview

PathOmicDRP is a multi-modal deep learning framework that integrates four biological data modalities — genomic mutations, transcriptomic gene expression, proteomic protein abundance, and histopathology whole-slide images — through cross-attention fusion to predict cancer drug response.

### Key Findings

- **Performance Dissociation**: Linear baselines (ElasticNet) achieve near-perfect accuracy on imputed IC50 targets (PCC = 0.925) but fail on real clinical treatment outcomes (AUC = 0.461). PathOmicDRP shows the opposite: modest imputed IC50 accuracy (PCC = 0.455) but strong clinical prediction (AUC = 0.831).
- **Representation Learning**: PathOmicDRP embeddings contain substantially more clinical information than the imputed IC50 training targets (AUC 0.735 vs. 0.423).
- **Cross-Attention Robustness**: Cross-attention fusion retains 55.6% performance with 3/4 modalities missing, vs. 43.0% for self-attention (+12.6 pp).
- **External Validation**: Drug sensitivity patterns align with GDSC cell line pharmacological data (Spearman rho = 0.670, p = 0.012).

## Architecture

```
Input Modalities
├── Genomic (193 mutation features) → Genomic Encoder → 8 tokens
├── Transcriptomic (2,000 genes) → Pathway Tokenizer → 2,000 tokens
├── Proteomic (464 RPPA proteins) → Proteomic Encoder → 16 tokens
└── Histopathology (H&E WSI) → UNI + ABMIL → 16 tokens

Cross-Attention Fusion (2 layers, 8 heads, bidirectional)
    Omics tokens <--> Histology tokens

Attention Pooling → 256-dim global representation
Prediction Head → IC50 for 13 drugs
```

## Project Structure

```
PathOmicDRP/
├── src/
│   ├── model.py                    # PathOmicDRP architecture
│   ├── dataset.py                  # Data loading and preprocessing
│   ├── train.py                    # Phase 1: 3-modal baseline training
│   ├── train_phase2.py             # Phase 2: Ablation studies
│   ├── train_phase3_4modal.py      # Phase 3: 4-modal training with histology
│   ├── interpretability.py         # Modality ablation, gradient attribution, ABMIL attention
│   ├── advanced_analysis.py        # Subtype analysis, drug clustering, survival
│   ├── high_impact_analyses.py     # Clinical outcome, LODO, biomarker concordance
│   ├── architecture_comparison.py  # Cross-attn vs self-attn vs MLP fusion
│   ├── priority_analyses.py        # Statistical tests, GDSC validation, SOTA benchmarking
│   ├── strengthening_analyses.py   # Fair comparison, survival, phenotype analysis
│   ├── strengthening_abc.py        # Modality dropout robustness, representation quality
│   └── w1w2w3_resolution.py        # Clinical AUC comparison, expanded validation
├── scripts/
│   ├── 01_download_clinical.py     # Download TCGA-BRCA clinical data
│   ├── 02_download_mutations.py    # Download somatic mutation data (MAF)
│   ├── 04_download_transcriptomic.py  # Download RNA-seq expression
│   ├── 05_download_proteomic.py    # Download RPPA proteomic data
│   ├── 06c_download_wsi_parallel.py   # Download H&E whole-slide images
│   ├── 07_download_gdsc.py        # Download GDSC drug sensitivity data
│   ├── 07_extract_uni_features.py  # Extract UNI foundation model features from WSIs
│   ├── 10_extract_genomic_features.py    # Process genomic features
│   ├── 11_extract_transcriptomic_features.py  # Process transcriptomic features
│   ├── 12_extract_proteomic_features.py  # Process proteomic features
│   ├── 13_extract_gdsc_brca.py     # Extract BRCA-specific GDSC data
│   └── 14_harmonize_samples.py     # Harmonize multi-modal sample IDs
├── configs/
│   └── default_config.json         # Default model hyperparameters
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/Soonlab/BRCA_Drug_Pred.git
cd BRCA_Drug_Pred

# Create conda environment
conda create -n pathomic python=3.10 -y
conda activate pathomic

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### 1. Download TCGA-BRCA Data
```bash
# Clinical data, mutations, expression, proteomics
python scripts/01_download_clinical.py
python scripts/02_download_mutations.py
python scripts/04_download_transcriptomic.py
python scripts/05_download_proteomic.py

# Whole-slide images (requires ~612GB storage)
python scripts/06c_download_wsi_parallel.py

# GDSC drug sensitivity data
python scripts/07_download_gdsc.py
```

### 2. Feature Extraction
```bash
# Extract and process features
python scripts/10_extract_genomic_features.py
python scripts/11_extract_transcriptomic_features.py
python scripts/12_extract_proteomic_features.py
python scripts/13_extract_gdsc_brca.py
python scripts/14_harmonize_samples.py

# Extract UNI histopathology features
python scripts/07_extract_uni_features.py
```

## Training

### Phase 1: 3-Modal Baseline
```bash
python src/train.py
```

### Phase 2: Modality Ablation
```bash
python src/train_phase2.py
```

### Phase 3: 4-Modal with Histopathology
```bash
python src/train_phase3_4modal.py
```

## Analysis

```bash
# Interpretability (ablation, gradient attribution, attention)
python src/interpretability.py

# Clinical outcome prediction, LODO, biomarker concordance
python src/high_impact_analyses.py

# Architecture comparison (cross-attn vs self-attn vs MLP)
python src/architecture_comparison.py

# SOTA benchmarking, statistical tests, GDSC validation
python src/priority_analyses.py

# Modality dropout robustness, representation quality analysis
python src/strengthening_abc.py
```

## Requirements

- Python 3.10+
- PyTorch 2.11+ (CUDA 12.8 for RTX 5090, or CUDA 12.6+ for other GPUs)
- 32GB+ GPU memory recommended
- ~700GB storage for WSI data

## Data Availability

- **TCGA-BRCA**: Available from [GDC Data Portal](https://portal.gdc.cancer.gov/)
- **GDSC**: Available from [Genomics of Drug Sensitivity in Cancer](https://www.cancerrxgene.org/)
- **UNI Foundation Model**: Available from [Mahmood Lab](https://github.com/mahmoodlab/UNI)

## Citation

If you use this code, please cite:

```
PathOmicDRP: Integrating Multi-Omics and Histopathology through Cross-Attention
Networks for Cancer Drug Response Prediction. (2026)
```

## License

This project is licensed under the MIT License.


## Create reports
```bash
python src/reinforce_cv_ablation.py
python src/reinforce_fair_embedding.py
python src/reinforce_metabric.py
python src/reinforce_drug_heterogeneity.py
python src/oof_predictions.py
python src/clinical_utility_v2.py
python src/biological_validation.py
python src/cptac_validation.py
python src/sota_benchmark.py
python src/reinforce_figures.py
python src/figures_v8.py
python -m src.figures_v8_nature.run_all
python src/generate_manuscript_v6.py
python src/generate_manuscript_v7.py
python src/patch_v7_discussion.py
python src/generate_manuscript_v8.py
```
