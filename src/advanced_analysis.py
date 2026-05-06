#!/usr/bin/env python3
"""
PathOmicDRP Advanced Analyses for npj Digital Medicine

1. Clinical Validation: predicted IC50 vs actual treatment response
2. Survival Analysis: KM curves + Cox regression
3. Molecular Subtype (ER/PR/HER2) analysis
4. UMAP embedding visualization
5. Drug mechanism clustering + cross-modal attention
6. WSI attention heatmap overlay
7. Benchmark comparison
8. Publication-quality composite figures
"""

import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.patheffects as pe

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings('ignore')

from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
CLIN_DIR = DATA_ROOT / "01_clinical"
RESULTS = PROJECT_ROOT / "results" / "phase3_4modal_full"
FIG_DIR = PROJECT_ROOT / "research" / "figures"
OUT_DIR = PROJECT_ROOT / "results" / "advanced_analysis"

# Matching TCGA drug names to our model drug columns
DRUG_NAME_MAP = {
    'Tamoxifen': 'Tamoxifen_1199',
    'Paclitaxel': 'Paclitaxel_1080',
    'Docetaxel': 'Docetaxel_1007',
    'Cyclophosphamide': 'Cyclophosphamide_1014',
    'Cisplatin': 'Cisplatin_1005',
    'Gemcitabine': 'Gemcitabine_1190',
    'Lapatinib': 'Lapatinib_1558',
    'Vinblastine': 'Vinblastine_1004',
    'Fulvestrant': 'Fulvestrant_1816',
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_and_data():
    """Load best 4-modal model and full dataset."""
    with open(os.path.join(RESULTS, "cv_results.json")) as f:
        cv = json.load(f)
    config = cv['config']
    drug_cols = cv['drugs']

    model = PathOmicDRP(config).to(DEVICE)
    state = torch.load(os.path.join(RESULTS, "best_model.pt"),
                       map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    gen_ids = set(gen_df['patient_id'])
    tra_ids = set(tra_df['patient_id'])
    pro_ids = set(pro_df['patient_id'])
    ic50_ids = set(ic50_df.index)
    histo_ids = {f.replace('.pt', '') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(gen_ids & tra_ids & pro_ids & ic50_ids & histo_ids)

    dataset = MultiDrugDataset4Modal(
        common, gen_df, tra_df, pro_df, ic50_df, drug_cols,
        histo_dir=HISTO_DIR, fit=True,
    )

    drug_names = [d.rsplit('_', 1)[0] for d in drug_cols]
    gen_names = [c for c in gen_df.columns if c != 'patient_id']
    tra_names = [c for c in tra_df.columns if c != 'patient_id']
    pro_names = [c for c in pro_df.columns if c != 'patient_id']

    return model, dataset, config, drug_cols, drug_names, gen_names, tra_names, pro_names, common, ic50_df, pro_df


def get_predictions(model, dataset):
    """Get model predictions for all patients."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            kwargs = {}
            if 'histology' in batch:
                kwargs['histology'] = batch['histology'].to(DEVICE)
                kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)
            out = model(g, t, p, **kwargs)['prediction'].cpu().numpy()
            all_pred.append(out)
    all_pred = np.concatenate(all_pred)
    # Inverse transform
    all_pred_orig = dataset.scalers['ic50'].inverse_transform(all_pred)
    return all_pred_orig


def get_fused_embeddings(model, dataset):
    """Extract fused multi-modal embeddings before prediction head."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)

    embeddings = []
    attn_weights_list = []

    def hook_fn(module, input, output):
        # After fusion self_attn, before prediction head
        embeddings.append(output.detach().cpu())

    # Register hook on fusion output
    handle = model.fusion.self_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            kwargs = {}
            if 'histology' in batch:
                kwargs['histology'] = batch['histology'].to(DEVICE)
                kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)
            result = model(g, t, p, **kwargs)
            if 'histo_attention' in result:
                attn_weights_list.append(result['histo_attention'].cpu())

    handle.remove()

    # Pool embeddings: mean over tokens
    pooled = []
    for emb in embeddings:
        pooled.append(emb.mean(dim=1))  # (B, hidden_dim)
    pooled = torch.cat(pooled, dim=0).numpy()
    return pooled, attn_weights_list


# ═══════════════════════════════════════════════════════════════════
# 1. CLINICAL VALIDATION
# ═══════════════════════════════════════════════════════════════════

def clinical_validation(pids, pred_ic50, drug_cols, drug_names):
    """Validate predicted IC50 against actual treatment outcomes."""
    log("=== 1. Clinical Validation ===")

    # Load treatment data
    drug_df = pd.read_csv(os.path.join(CLIN_DIR, "TCGA_BRCA_drug_treatments.csv"))

    # Create patient → predicted IC50 mapping
    pred_df = pd.DataFrame(pred_ic50, columns=drug_cols, index=pids)

    results = {}

    for tcga_drug, model_col in DRUG_NAME_MAP.items():
        if model_col not in drug_cols:
            continue

        # Find patients who received this drug
        treated = drug_df[drug_df['therapeutic_agents'].str.contains(tcga_drug, case=False, na=False)]
        treated_pids = set(treated['submitter_id']) & set(pids)

        if len(treated_pids) < 10:
            continue

        # Get outcomes
        outcomes = []
        for _, row in treated[treated['submitter_id'].isin(treated_pids)].iterrows():
            pid = row['submitter_id']
            outcome = row['treatment_outcome']
            ic50_pred = pred_df.loc[pid, model_col]
            outcomes.append({
                'patient_id': pid,
                'outcome': outcome,
                'predicted_ic50': ic50_pred,
            })

        df = pd.DataFrame(outcomes).drop_duplicates('patient_id')

        # Binary: response vs non-response
        df['responded'] = df['outcome'].isin(['Complete Response', 'Partial Response']).astype(int)
        df['progressed'] = df['outcome'].isin(['Progressive Disease']).astype(int)

        responders = df[df['responded'] == 1]['predicted_ic50']
        non_responders = df[df['responded'] == 0]['predicted_ic50']

        # Lower IC50 = more sensitive
        if len(responders) >= 5 and len(non_responders) >= 5:
            stat, pval = mannwhitneyu(responders, non_responders, alternative='less')
            try:
                auc = roc_auc_score(df['responded'], -df['predicted_ic50'])
            except:
                auc = 0.5
        else:
            pval, auc = 1.0, 0.5

        results[tcga_drug] = {
            'n_treated': len(df),
            'n_responded': int(df['responded'].sum()),
            'n_progressed': int(df['progressed'].sum()),
            'auc': float(auc),
            'pval': float(pval),
            'responder_ic50_mean': float(responders.mean()) if len(responders) > 0 else None,
            'nonresponder_ic50_mean': float(non_responders.mean()) if len(non_responders) > 0 else None,
            'data': df,
        }

        log(f"  {tcga_drug:20s} | n={len(df):3d} | resp={df['responded'].sum():3d} | "
            f"AUC={auc:.3f} | p={pval:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def survival_analysis(pids, pred_ic50, drug_cols, drug_names):
    """KM curves and Cox regression for predicted drug sensitivity."""
    log("=== 2. Survival Analysis ===")

    clin = pd.read_csv(os.path.join(CLIN_DIR, "TCGA_BRCA_clinical.csv"))
    clin = clin.set_index('submitter_id')

    # Prepare survival data
    surv_data = []
    for pid in pids:
        if pid not in clin.index:
            continue
        row = clin.loc[pid]
        vital = row.get('vital_status')
        if pd.isna(vital):
            continue

        event = 1 if vital == 'Dead' else 0
        if event == 1:
            time_val = row.get('days_to_death')
        else:
            time_val = row.get('days_to_last_follow_up')

        if pd.isna(time_val) or time_val <= 0:
            continue

        surv_data.append({
            'patient_id': pid,
            'time': float(time_val),
            'event': event,
            'age': row.get('age_at_index', np.nan),
        })

    surv_df = pd.DataFrame(surv_data).set_index('patient_id')
    pred_df = pd.DataFrame(pred_ic50, columns=drug_cols, index=pids)

    log(f"  Survival data: {len(surv_df)} patients, {surv_df['event'].sum()} events")

    results = {}

    # For each clinically relevant drug, stratify by predicted sensitivity
    key_drugs = ['Tamoxifen_1199', 'Paclitaxel_1080', 'Docetaxel_1007',
                 'Cisplatin_1005', 'Cyclophosphamide_1014']

    for drug_col in key_drugs:
        if drug_col not in drug_cols:
            continue
        drug_name = drug_col.rsplit('_', 1)[0]

        # Merge predictions with survival
        common_pids = surv_df.index.intersection(pred_df.index)
        merged = surv_df.loc[common_pids].copy()
        merged['predicted_ic50'] = pred_df.loc[common_pids, drug_col]

        # Stratify by median
        median_ic50 = merged['predicted_ic50'].median()
        merged['sensitive'] = (merged['predicted_ic50'] < median_ic50).astype(int)

        # Log-rank test
        sensitive = merged[merged['sensitive'] == 1]
        resistant = merged[merged['sensitive'] == 0]

        lr_result = logrank_test(
            sensitive['time'], resistant['time'],
            sensitive['event'], resistant['event']
        )

        # Cox regression
        cox_df = merged[['time', 'event', 'predicted_ic50']].copy()
        if 'age' in merged.columns:
            cox_df['age'] = merged['age']
            cox_df = cox_df.dropna()

        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='time', event_col='event')
            hr = float(np.exp(cph.params_['predicted_ic50']))
            hr_ci = cph.confidence_intervals_.loc['predicted_ic50']
            hr_pval = float(cph.summary.loc['predicted_ic50', 'p'])
        except:
            hr, hr_pval = 1.0, 1.0
            hr_ci = pd.Series([0, 0])

        results[drug_name] = {
            'n': len(merged),
            'n_events': int(merged['event'].sum()),
            'logrank_p': float(lr_result.p_value),
            'hr': hr,
            'hr_pval': hr_pval,
            'merged': merged,
        }

        log(f"  {drug_name:20s} | n={len(merged)} | events={merged['event'].sum()} | "
            f"logrank p={lr_result.p_value:.4f} | HR={hr:.3f} (p={hr_pval:.4f})")

    return results, surv_df


# ═══════════════════════════════════════════════════════════════════
# 3. MOLECULAR SUBTYPE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def molecular_subtype_analysis(pids, pred_ic50, drug_cols, drug_names, pro_df):
    """Define ER/PR/HER2 subtypes from RPPA and analyze per-subtype performance."""
    log("=== 3. Molecular Subtype Analysis ===")

    pro = pro_df.set_index('patient_id')
    common_pids = [p for p in pids if p in pro.index]

    # Extract receptor status from RPPA
    er_col = 'ERALPHA' if 'ERALPHA' in pro.columns else None
    pr_col = 'PR' if 'PR' in pro.columns else None
    her2_col = 'HER2' if 'HER2' in pro.columns else None

    if not all([er_col, pr_col, her2_col]):
        log("  Missing receptor columns in RPPA data")
        return {}

    subtype_data = []
    for pid in common_pids:
        er = pro.loc[pid, er_col]
        pr_val = pro.loc[pid, pr_col]
        her2 = pro.loc[pid, her2_col]

        # Use median as threshold
        er_pos = er > pro[er_col].median()
        pr_pos = pr_val > pro[pr_col].median()
        her2_pos = her2 > pro[her2_col].median()

        if er_pos and not her2_pos:
            subtype = 'Luminal'
        elif her2_pos:
            subtype = 'HER2+'
        elif not er_pos and not pr_pos and not her2_pos:
            subtype = 'Triple-Negative'
        else:
            subtype = 'Other'

        subtype_data.append({
            'patient_id': pid,
            'subtype': subtype,
            'ER': float(er), 'PR': float(pr_val), 'HER2': float(her2),
            'ER_pos': bool(er_pos), 'PR_pos': bool(pr_pos), 'HER2_pos': bool(her2_pos),
        })

    sub_df = pd.DataFrame(subtype_data).set_index('patient_id')
    log(f"  Subtypes: {sub_df['subtype'].value_counts().to_dict()}")

    # Per-subtype drug predictions
    pred_df = pd.DataFrame(pred_ic50, columns=drug_cols, index=pids)
    sub_drug = sub_df.join(pred_df, how='inner')

    results = {
        'subtype_counts': sub_df['subtype'].value_counts().to_dict(),
        'subtype_df': sub_df,
        'drug_by_subtype': {},
    }

    for drug_col, drug_name in zip(drug_cols, drug_names):
        subtype_means = {}
        for subtype in ['Luminal', 'HER2+', 'Triple-Negative']:
            mask = sub_drug['subtype'] == subtype
            if mask.sum() > 0:
                subtype_means[subtype] = float(sub_drug.loc[mask, drug_col].mean())
        results['drug_by_subtype'][drug_name] = subtype_means

    # Kruskal-Wallis test per drug
    for drug_col, drug_name in zip(drug_cols, drug_names):
        groups = [sub_drug.loc[sub_drug['subtype'] == s, drug_col].values
                  for s in ['Luminal', 'HER2+', 'Triple-Negative']
                  if (sub_drug['subtype'] == s).sum() > 5]
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            results['drug_by_subtype'][drug_name]['kruskal_p'] = float(pval)
            if pval < 0.05:
                log(f"  {drug_name:15s} | subtype-specific (p={pval:.4f})")

    return results


# ═══════════════════════════════════════════════════════════════════
# 4. UMAP EMBEDDING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def umap_analysis(embeddings, pids, subtype_results, surv_df, pred_ic50, drug_cols, drug_names):
    """UMAP of fused embeddings colored by various clinical features."""
    log("=== 4. UMAP Embedding ===")

    if not HAS_UMAP:
        log("  umap-learn not available, using t-SNE")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = reducer.fit_transform(embeddings)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
        coords = reducer.fit_transform(embeddings)

    log(f"  Reduced {embeddings.shape} → {coords.shape}")

    sub_df = subtype_results.get('subtype_df', pd.DataFrame())
    rows = []
    for i, pid in enumerate(pids):
        row = {
            'patient_id': pid,
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1]),
            'subtype': 'Other',
        }
        if not sub_df.empty and pid in sub_df.index:
            row['subtype'] = sub_df.loc[pid, 'subtype']
        rows.append(row)
    umap_df = pd.DataFrame(rows)
    umap_df.to_csv(os.path.join(OUT_DIR, 'umap_embedding_data.csv'), index=False)
    log(f"  Saved {OUT_DIR}/umap_embedding_data.csv")

    return coords


# ═══════════════════════════════════════════════════════════════════
# 5. DRUG MECHANISM CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def drug_clustering(pred_ic50, drug_cols, drug_names):
    """Cluster drugs by predicted IC50 correlation patterns."""
    log("=== 5. Drug Mechanism Clustering ===")

    pred_df = pd.DataFrame(pred_ic50, columns=drug_names)
    corr = pred_df.corr(method='spearman')

    # Hierarchical clustering
    dist = pdist(corr.values, metric='correlation')
    Z = linkage(dist, method='ward')

    # Drug mechanism annotations
    drug_moa = {
        'Cisplatin': 'DNA crosslinker',
        'Docetaxel': 'Microtubule stabilizer',
        'Paclitaxel': 'Microtubule stabilizer',
        'Gemcitabine': 'Antimetabolite',
        'Tamoxifen': 'ER antagonist',
        'Lapatinib': 'HER2/EGFR TKI',
        'Vinblastine': 'Microtubule destabilizer',
        'OSI-027': 'mTOR inhibitor',
        'Daporinad': 'NAMPT inhibitor',
        'Venetoclax': 'BCL-2 inhibitor',
        'ABT737': 'BCL-2/BCL-XL inhibitor',
        'AZD5991': 'MCL-1 inhibitor',
        'Fulvestrant': 'ER degrader',
    }

    log(f"  Drug correlation matrix computed")
    return corr, Z, drug_moa


# ═══════════════════════════════════════════════════════════════════
# 6. WSI ATTENTION HEATMAP
# ═══════════════════════════════════════════════════════════════════

def wsi_attention_heatmap(model, dataset, pids, n_patients=4):
    """Generate WSI-level attention heatmaps for representative patients."""
    log("=== 6. WSI Attention Heatmap ===")
    import openslide
    from torch.utils.data import DataLoader

    WSI_DIR = DATA_ROOT / "05_morphology" / "wsi"
    TARGET_CSV = DATA_ROOT / "05_morphology" / "wsi_target_3modal.csv"

    # Load WSI target mapping
    wsi_map = defaultdict(list)
    with open(TARGET_CSV) as f:
        import csv
        reader = csv.DictReader(f)
        for r in reader:
            wsi_map[r['patient_id']].append(r['file_name'])

    # Select patients with different subtypes if possible
    selected = []
    for i, pid in enumerate(pids):
        if pid in wsi_map and len(wsi_map[pid]) > 0:
            svs_path = os.path.join(WSI_DIR, wsi_map[pid][0])
            if os.path.exists(svs_path):
                selected.append((i, pid, svs_path))
        if len(selected) >= n_patients:
            break

    heatmap_data = []

    for idx, pid, svs_path in selected:
        # Get model's attention for this patient
        sample = dataset[idx]
        batch = collate_4modal([sample])

        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)

        kwargs = {}
        if 'histology' in batch:
            kwargs['histology'] = batch['histology'].to(DEVICE)
            kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

        with torch.no_grad():
            result = model(g, t, p, **kwargs)

        if 'histo_attention' not in result:
            continue

        attn = result['histo_attention'].cpu().numpy()[0]  # (n_tokens, N_patches)
        n_patches = batch['histo_mask'][0].sum().item()
        attn_avg = attn[:, :n_patches].mean(axis=0)  # Average over tokens

        # Get patch coordinates from WSI
        try:
            slide = openslide.OpenSlide(svs_path)
            mag = float(slide.properties.get("openslide.objective-power", 40))
            ds = mag / 20
            ps_l0 = int(256 * ds)
            w, h = slide.dimensions

            # Generate thumbnail
            thumb = slide.get_thumbnail((1024, 1024))
            thumb_arr = np.array(thumb.convert("RGB"))

            # Reconstruct patch grid coordinates
            coords = []
            thumb_gray = np.mean(np.array(slide.get_thumbnail((512, 512)).convert("RGB")), axis=2)
            sx, sy = 512 / w, 512 / h

            for y in range(0, h - ps_l0 + 1, ps_l0):
                for x in range(0, w - ps_l0 + 1, ps_l0):
                    tx, ty = int(x * sx), int(y * sy)
                    tw, th = max(1, int(ps_l0 * sx)), max(1, int(ps_l0 * sy))
                    tr = thumb_gray[ty:ty+th, tx:tx+tw]
                    if tr.size > 0 and np.mean(tr < 220) > 0.5:
                        arr_check = np.array(slide.read_region(
                            (x, y), 1 if slide.level_count > 1 else 0,
                            (int(ps_l0 / slide.level_downsamples[1 if slide.level_count > 1 else 0]),) * 2
                        ).convert("RGB"))
                        if np.mean(np.mean(arr_check, axis=2) < 220) > 0.7:
                            coords.append((x, y))

            slide.close()

            # Trim to match number of patches
            coords = coords[:n_patches]
            attn_trimmed = attn_avg[:len(coords)]

            heatmap_data.append({
                'pid': pid,
                'thumb': thumb_arr,
                'coords': coords,
                'attention': attn_trimmed,
                'w': w, 'h': h, 'ps_l0': ps_l0,
            })
            log(f"  {pid}: {len(coords)} patches, thumb={thumb_arr.shape}")

        except Exception as e:
            log(f"  {pid}: error - {e}")

    return heatmap_data


# ═══════════════════════════════════════════════════════════════════
# 7. BENCHMARK COMPARISON
# ═══════════════════════════════════════════════════════════════════

def benchmark_comparison(dataset, drug_cols, drug_names, pids):
    """Compare PathOmicDRP against traditional ML baselines."""
    log("=== 7. Benchmark Comparison ===")

    # Collect features and targets from dataset
    gen_all, tra_all, pro_all, y_all = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        gen_all.append(s['genomic'].numpy())
        tra_all.append(s['transcriptomic'].numpy())
        pro_all.append(s['proteomic'].numpy())
        y_all.append(s['target'].numpy())

    gen_all = np.array(gen_all)
    tra_all = np.array(tra_all)
    pro_all = np.array(pro_all)
    y_all = np.array(y_all)

    # Concatenate omics features
    X = np.hstack([gen_all, tra_all, pro_all])
    y = y_all  # Already scaled

    log(f"  Features: {X.shape}, Targets: {y.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    baselines = {
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    results = {}
    for name, model_base in baselines.items():
        fold_pcc_global = []
        fold_pcc_drug = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Scale X
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

            # Fit per-drug models
            preds = np.zeros_like(y_val)
            for d in range(y.shape[1]):
                m = type(model_base)(**model_base.get_params())
                m.fit(X_tr, y_tr[:, d])
                preds[:, d] = m.predict(X_val)

            # Inverse transform for metrics
            preds_orig = dataset.scalers['ic50'].inverse_transform(preds)
            y_val_orig = dataset.scalers['ic50'].inverse_transform(y_val)

            pcc_global, _ = pearsonr(preds_orig.flatten(), y_val_orig.flatten())
            fold_pcc_global.append(pcc_global)

            drug_pccs = []
            for d in range(y.shape[1]):
                try:
                    pcc, _ = pearsonr(preds_orig[:, d], y_val_orig[:, d])
                except:
                    pcc = 0
                drug_pccs.append(pcc)
            fold_pcc_drug.append(np.mean(drug_pccs))

        results[name] = {
            'pcc_global': (np.mean(fold_pcc_global), np.std(fold_pcc_global)),
            'pcc_drug': (np.mean(fold_pcc_drug), np.std(fold_pcc_drug)),
        }
        log(f"  {name:20s} | PCC_global={np.mean(fold_pcc_global):.4f}±{np.std(fold_pcc_global):.4f} | "
            f"PCC_drug={np.mean(fold_pcc_drug):.4f}±{np.std(fold_pcc_drug):.4f}")

    # Add our model results
    with open(os.path.join(RESULTS, "cv_results.json")) as f:
        r4 = json.load(f)
    with open(PROJECT_ROOT / "results" / "phase3_3modal_baseline" / "cv_results.json") as f:
        r3 = json.load(f)

    results['PathOmicDRP (3-modal)'] = {
        'pcc_global': (r3['avg']['pcc_global']['mean'], r3['avg']['pcc_global']['std']),
        'pcc_drug': (r3['avg']['pcc_per_drug_mean']['mean'], r3['avg']['pcc_per_drug_mean']['std']),
    }
    results['PathOmicDRP (4-modal)'] = {
        'pcc_global': (r4['avg']['pcc_global']['mean'], r4['avg']['pcc_global']['std']),
        'pcc_drug': (r4['avg']['pcc_per_drug_mean']['mean'], r4['avg']['pcc_per_drug_mean']['std']),
    }

    return results


# ═══════════════════════════════════════════════════════════════════
# 8. PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════

def plot_clinical_validation(clin_results, save_dir):
    """Figure 6: Clinical validation waterfall + ROC."""
    drugs_with_data = [(k, v) for k, v in clin_results.items() if v['n_treated'] >= 20]
    if not drugs_with_data:
        log("  Not enough clinical data for figure")
        return

    n_drugs = len(drugs_with_data)
    fig = plt.figure(figsize=(18, 5 * ((n_drugs + 1) // 2)))
    gs = gridspec.GridSpec((n_drugs + 1) // 2, 2, hspace=0.4, wspace=0.3)

    for i, (drug, data) in enumerate(drugs_with_data):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        df = data['data'].sort_values('predicted_ic50')

        colors = ['#4CAF50' if r else '#F44336' if p else '#BDBDBD'
                  for r, p in zip(df['responded'], df['progressed'])]

        ax.bar(range(len(df)), df['predicted_ic50'], color=colors, width=1.0,
               edgecolor='none')
        ax.axhline(y=df['predicted_ic50'].median(), color='black', linestyle='--',
                   alpha=0.5, linewidth=0.8)
        ax.set_title(f'{drug} (n={data["n_treated"]}, AUC={data["auc"]:.3f}, p={data["pval"]:.3f})',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted IC₅₀', fontsize=10)
        ax.set_xlabel('Patients (ranked)', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', label=f'Response ({data["n_responded"]})'),
            Patch(facecolor='#F44336', label=f'Progression ({data["n_progressed"]})'),
            Patch(facecolor='#BDBDBD', label='Other'),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc='upper left')

    plt.savefig(os.path.join(save_dir, 'Fig6_clinical_validation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig6_clinical_validation.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig6_clinical_validation")


def plot_survival(surv_results, save_dir):
    """Figure 7: KM survival curves."""
    drugs = [k for k in surv_results if surv_results[k]['n_events'] >= 10]
    if not drugs:
        log("  Not enough survival events for figure")
        return

    n = len(drugs)
    fig, axes = plt.subplots(1, min(n, 3), figsize=(6 * min(n, 3), 5))
    if min(n, 3) == 1:
        axes = [axes]

    for i, drug in enumerate(drugs[:3]):
        ax = axes[i]
        data = surv_results[drug]
        merged = data['merged']

        kmf_s = KaplanMeierFitter()
        kmf_r = KaplanMeierFitter()

        sensitive = merged[merged['sensitive'] == 1]
        resistant = merged[merged['sensitive'] == 0]

        kmf_s.fit(sensitive['time'] / 365.25, sensitive['event'], label='Predicted Sensitive')
        kmf_r.fit(resistant['time'] / 365.25, resistant['event'], label='Predicted Resistant')

        kmf_s.plot_survival_function(ax=ax, color='#4CAF50', linewidth=2)
        kmf_r.plot_survival_function(ax=ax, color='#F44336', linewidth=2)

        ax.set_title(f'{drug}\nlog-rank p={data["logrank_p"]:.4f}, HR={data["hr"]:.2f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_ylabel('Overall Survival', fontsize=10)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig7_survival.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig7_survival.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig7_survival")


def plot_umap(coords, pids, subtype_results, pred_ic50, drug_cols, drug_names, save_dir):
    """Figure 8: UMAP embeddings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 8A: Colored by subtype
    ax = axes[0]
    sub_df = subtype_results.get('subtype_df', pd.DataFrame())
    subtype_colors = {'Luminal': '#2196F3', 'HER2+': '#FF9800', 'Triple-Negative': '#E91E63', 'Other': '#BDBDBD'}

    if not sub_df.empty:
        for i, pid in enumerate(pids):
            if pid in sub_df.index:
                st = sub_df.loc[pid, 'subtype']
                ax.scatter(coords[i, 0], coords[i, 1], c=subtype_colors.get(st, '#BDBDBD'),
                           s=8, alpha=0.7, edgecolors='none')
            else:
                ax.scatter(coords[i, 0], coords[i, 1], c='#EEEEEE', s=5, alpha=0.3, edgecolors='none')

        for st, color in subtype_colors.items():
            if st != 'Other':
                ax.scatter([], [], c=color, s=30, label=st)
        ax.legend(fontsize=9, markerscale=2)

    ax.set_title('A  Molecular Subtype', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.set_ylabel('UMAP 2', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 8B: Colored by Tamoxifen sensitivity
    ax = axes[1]
    pred_df = pd.DataFrame(pred_ic50, columns=drug_cols, index=pids)
    tam_col = 'Tamoxifen_1199'
    if tam_col in drug_cols:
        tam_vals = pred_df[tam_col].values
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=tam_vals, cmap='RdYlGn_r',
                        s=8, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7, label='Predicted IC₅₀')
    ax.set_title('B  Tamoxifen Sensitivity', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 8C: Colored by Paclitaxel sensitivity
    ax = axes[2]
    pac_col = 'Paclitaxel_1080'
    if pac_col in drug_cols:
        pac_vals = pred_df[pac_col].values
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=pac_vals, cmap='RdYlGn_r',
                        s=8, alpha=0.7, edgecolors='none')
        plt.colorbar(sc, ax=ax, shrink=0.7, label='Predicted IC₅₀')
    ax.set_title('C  Paclitaxel Sensitivity', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('UMAP 1', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig8_umap.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig8_umap.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig8_umap")


def plot_drug_clustering(corr, Z, drug_moa, save_dir):
    """Figure 9: Drug clustering heatmap + dendrogram."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.3, 1], height_ratios=[0.3, 1],
                           hspace=0.02, wspace=0.02)

    # Dendrogram (top)
    ax_dend = fig.add_subplot(gs[0, 1])
    dn = dendrogram(Z, labels=corr.columns.tolist(), ax=ax_dend, leaf_rotation=90,
                    leaf_font_size=8, color_threshold=0.5)
    ax_dend.set_title('Drug Clustering by Predicted IC₅₀ Correlation', fontsize=12, fontweight='bold')
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)
    ax_dend.spines['bottom'].set_visible(False)

    # Reorder correlation matrix by dendrogram
    order = dn['leaves']
    corr_ordered = corr.iloc[order, order]

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(corr_ordered.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_heat.set_xticks(range(len(corr_ordered)))
    ax_heat.set_xticklabels(corr_ordered.columns, rotation=45, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(corr_ordered)))
    ax_heat.set_yticklabels(corr_ordered.index, fontsize=8)
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label='Spearman ρ')

    # MOA annotation (left)
    ax_moa = fig.add_subplot(gs[1, 0])
    moa_colors = {
        'DNA crosslinker': '#F44336',
        'Microtubule stabilizer': '#4CAF50',
        'Microtubule destabilizer': '#8BC34A',
        'Antimetabolite': '#FF9800',
        'ER antagonist': '#2196F3',
        'ER degrader': '#1565C0',
        'HER2/EGFR TKI': '#9C27B0',
        'mTOR inhibitor': '#795548',
        'NAMPT inhibitor': '#607D8B',
        'BCL-2 inhibitor': '#E91E63',
        'BCL-2/BCL-XL inhibitor': '#F06292',
        'MCL-1 inhibitor': '#AD1457',
    }

    ordered_drugs = [corr_ordered.index[i] for i in range(len(corr_ordered))]
    for i, drug in enumerate(ordered_drugs):
        moa = drug_moa.get(drug, 'Other')
        color = moa_colors.get(moa, '#BDBDBD')
        ax_moa.barh(i, 1, color=color, edgecolor='none')
        ax_moa.text(0.5, i, moa, ha='center', va='center', fontsize=6, fontweight='bold')

    ax_moa.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    ax_moa.invert_yaxis()
    ax_moa.set_xlim(0, 1)
    ax_moa.axis('off')

    plt.savefig(os.path.join(save_dir, 'Fig9_drug_clustering.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig9_drug_clustering.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig9_drug_clustering")


def plot_wsi_heatmap(heatmap_data, save_dir):
    """Figure 10: WSI attention overlay."""
    if not heatmap_data:
        log("  No heatmap data")
        return

    n = len(heatmap_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, data in zip(axes, heatmap_data):
        thumb = data['thumb']
        coords = data['coords']
        attn = data['attention']
        w, h, ps = data['w'], data['h'], data['ps_l0']

        ax.imshow(thumb)

        if len(coords) > 0 and len(attn) > 0:
            # Normalize attention
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-10)

            # Scale coordinates to thumbnail
            thumb_h, thumb_w = thumb.shape[:2]
            sx, sy = thumb_w / w, thumb_h / h

            cmap = cm.get_cmap('hot')
            for (x, y), a in zip(coords, attn_norm):
                tx, ty = x * sx, y * sy
                tw, th = ps * sx, ps * sy
                color = cmap(a)
                rect = plt.Rectangle((tx, ty), tw, th, facecolor=color,
                                     edgecolor='none', alpha=0.5 * a + 0.1)
                ax.add_patch(rect)

        ax.set_title(f'{data["pid"]}\n({len(coords)} patches)', fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle('ABMIL Attention Heatmap on H&E Whole-Slide Images', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig10_wsi_attention.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig10_wsi_attention.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig10_wsi_attention")


def plot_benchmark(benchmark_results, save_dir):
    """Figure 11: Benchmark comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = list(benchmark_results.keys())
    colors = ['#BDBDBD', '#90A4AE', '#78909C', '#546E7A', '#E91E63']
    if len(colors) < len(methods):
        colors = colors + ['#BDBDBD'] * (len(methods) - len(colors))

    # 11A: PCC_global
    ax = axes[0]
    means = [benchmark_results[m]['pcc_global'][0] for m in methods]
    stds = [benchmark_results[m]['pcc_global'][1] for m in methods]
    bars = ax.barh(range(len(methods)), means, xerr=stds, color=colors[:len(methods)],
                   edgecolor='black', linewidth=0.5, capsize=5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('PCC (global)', fontsize=11)
    ax.set_title('A  Global Prediction Performance', fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9, fontweight='bold')

    # 11B: PCC_drug
    ax = axes[1]
    means = [benchmark_results[m]['pcc_drug'][0] for m in methods]
    stds = [benchmark_results[m]['pcc_drug'][1] for m in methods]
    bars = ax.barh(range(len(methods)), means, xerr=stds, color=colors[:len(methods)],
                   edgecolor='black', linewidth=0.5, capsize=5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('PCC (per-drug mean)', fontsize=11)
    ax.set_title('B  Per-Drug Prediction Performance', fontsize=12, fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig11_benchmark.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig11_benchmark.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved Fig11_benchmark")


def plot_subtype_radar(subtype_results, drug_names, save_dir):
    """Figure S4: Radar chart of drug sensitivity by subtype."""
    drug_by_sub = subtype_results.get('drug_by_subtype', {})
    subtypes = ['Luminal', 'HER2+', 'Triple-Negative']
    colors = ['#2196F3', '#FF9800', '#E91E63']

    # Get drugs that have data for all subtypes
    valid_drugs = [d for d in drug_names if all(s in drug_by_sub.get(d, {}) for s in subtypes)]
    if len(valid_drugs) < 3:
        log("  Not enough data for radar chart")
        return

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(valid_drugs), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    for subtype, color in zip(subtypes, colors):
        values = [drug_by_sub[d][subtype] for d in valid_drugs]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=subtype, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(valid_drugs, fontsize=9)
    ax.set_title('Drug Sensitivity by Molecular Subtype\n(Predicted IC₅₀)', fontsize=13,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'FigS4_subtype_radar.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'FigS4_subtype_radar.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  Saved FigS4_subtype_radar")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    log(f"Device: {DEVICE}")

    # Load model and data
    model, dataset, config, drug_cols, drug_names, gen_names, tra_names, pro_names, pids, ic50_df, pro_df = \
        load_model_and_data()
    log(f"Loaded {len(pids)} patients, {len(drug_cols)} drugs")

    # Get predictions
    pred_ic50 = get_predictions(model, dataset)
    log(f"Predictions: {pred_ic50.shape}")

    # Get embeddings
    embeddings, attn_weights = get_fused_embeddings(model, dataset)
    log(f"Embeddings: {embeddings.shape}")

    # ── 1. Clinical Validation ──
    clin_results = clinical_validation(pids, pred_ic50, drug_cols, drug_names)

    # ── 2. Survival Analysis ──
    surv_results, surv_df = survival_analysis(pids, pred_ic50, drug_cols, drug_names)

    # ── 3. Molecular Subtype ──
    subtype_results = molecular_subtype_analysis(pids, pred_ic50, drug_cols, drug_names, pro_df)

    # ── 4. UMAP ──
    umap_coords = umap_analysis(embeddings, pids, subtype_results, surv_df,
                                pred_ic50, drug_cols, drug_names)

    # ── 5. Drug Clustering ──
    corr, Z, drug_moa = drug_clustering(pred_ic50, drug_cols, drug_names)

    # ── 6. WSI Attention Heatmap ──
    heatmap_data = wsi_attention_heatmap(model, dataset, pids, n_patients=4)

    # ── 7. Benchmark ──
    benchmark_results = benchmark_comparison(dataset, drug_cols, drug_names, pids)

    # ── 8. Figures ──
    log("\n=== Generating Figures ===")
    plot_clinical_validation(clin_results, FIG_DIR)
    plot_survival(surv_results, FIG_DIR)
    plot_umap(umap_coords, pids, subtype_results, pred_ic50, drug_cols, drug_names, FIG_DIR)
    plot_drug_clustering(corr, Z, drug_moa, FIG_DIR)
    plot_wsi_heatmap(heatmap_data, FIG_DIR)
    plot_benchmark(benchmark_results, FIG_DIR)
    plot_subtype_radar(subtype_results, drug_names, FIG_DIR)

    # ── Save all results ──
    log("\n=== Saving Results ===")
    summary = {
        'clinical_validation': {k: {kk: vv for kk, vv in v.items() if kk != 'data'}
                                for k, v in clin_results.items()},
        'survival': {k: {kk: vv for kk, vv in v.items() if kk != 'merged'}
                     for k, v in surv_results.items()},
        'subtype_counts': subtype_results.get('subtype_counts', {}),
        'benchmark': benchmark_results,
    }
    with open(os.path.join(OUT_DIR, 'advanced_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log("\n=== All advanced analyses complete ===")
