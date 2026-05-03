"""Shared data loaders for v8 figures."""
import json, os, numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
BASE = PROJECT_ROOT

def J(p):
    with open(f"{BASE}/{p}") as f: return json.load(f)

def short_drug(name: str) -> str:
    return name.split('_')[0]

def load_phase_cv(tag):
    d = J(f"results/{tag}/cv_results.json")
    return d

def per_drug_pcc(cv):
    """Returns dict drug_short -> (mean, std) across folds."""
    drugs = cv['drugs']
    folds = cv['drug_metrics_per_fold']
    out = {}
    for d in drugs:
        vals = [float(f[d]['pcc']) for f in folds]
        out[short_drug(d)] = (float(np.mean(vals)), float(np.std(vals)))
    return out

def load_interp():
    return J("research/figures/interpretability/modality_importance.json")

def load_clinical_auc():
    return J("results/analysis1_clinical_outcome/results.json")

def load_lodo():
    return J("results/analysis3_lodo/results.json")

def load_multitask():
    return J("results/analysis4_multitask/results.json")

def load_phenotype():
    r = J("results/analysis6_phenotype/results.json")
    cl = np.load(f"{BASE}/results/analysis6_phenotype/cluster_labels.npy")
    um = np.load(f"{BASE}/results/analysis6_phenotype/patch_umap.npy")
    return r, cl, um

def load_advanced():
    return J("results/advanced_analysis/advanced_analysis_results.json")

def load_umap_patient():
    return pd.read_csv(f"{BASE}/results/advanced_analysis/umap_embedding_data.csv")

def load_metabric():
    return J("results/reinforce/metabric_validation.json")

def load_cv_ablation():
    return J("results/reinforce/cv_ablation.json")

def load_fair():
    return J("results/reinforce/fair_embedding_and_bootstrap.json")

def load_heterogeneity():
    return J("results/reinforce/drug_heterogeneity.json")

def load_robustness():
    return J("results/strengthening/analysis_a_robustness.json")

def load_cohort_sizes():
    """Count rows for each modality CSV."""
    out = {}
    for mod, fn in [('Genomic','X_genomic.csv'),('Transcriptomic','X_transcriptomic.csv'),
                    ('Proteomic','X_proteomic.csv')]:
        try:
            out[mod] = sum(1 for _ in open(DATA_ROOT / "07_integrated" / fn)) - 1
        except: out[mod] = 0
    try:
        out['Histology'] = sum(1 for f in os.listdir(DATA_ROOT / "05_morphology" / "features") if f.endswith('.pt'))
    except: out['Histology'] = 431
    out['Clinical'] = 1098
    out['Drug treatment'] = 776
    return out

# Drug MOA assignment
DRUG_MOA = {
    'Cisplatin':'DNA damage','Gemcitabine':'DNA damage',
    'Docetaxel':'Tubulin','Paclitaxel':'Tubulin','Vinblastine':'Tubulin',
    'Tamoxifen':'Hormone','Fulvestrant':'Hormone',
    'Lapatinib':'Kinase','OSI-027':'Kinase',
    'Daporinad':'mTOR/NAD',
    'Venetoclax':'Apoptosis','ABT737':'Apoptosis','AZD5991':'Apoptosis',
}
DRUG_ORDER_13 = ['Cisplatin','Docetaxel','Paclitaxel','Gemcitabine','Tamoxifen','Lapatinib',
                 'Vinblastine','OSI-027','Daporinad','Venetoclax','ABT737','AZD5991','Fulvestrant']
