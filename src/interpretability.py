"""
PathOmicDRP Interpretability Analysis

1. Modality-level ablation importance (masking each modality)
2. Feature-level SHAP values (GradientExplainer for omics)
3. ABMIL attention analysis (histology patch importance)
4. Visualization: heatmaps, bar charts, attention maps

Outputs: Figure 5 (interpretability) + supplementary figures
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
RESULTS = PROJECT_ROOT / "results" / "phase3_4modal_full"
FIG_DIR = PROJECT_ROOT / "research" / "figures"


def load_model_and_data():
    """Load best 4-modal model and full dataset."""
    with open(os.path.join(RESULTS, "cv_results.json")) as f:
        cv = json.load(f)

    config = cv['config']
    drug_cols = cv['drugs']

    model = PathOmicDRP(config).to(DEVICE)
    state = torch.load(os.path.join(RESULTS, "best_model.pt"), map_location=DEVICE, weights_only=True)
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

    # Feature names
    gen_names = [c for c in gen_df.columns if c != 'patient_id']
    tra_names = [c for c in tra_df.columns if c != 'patient_id']
    pro_names = [c for c in pro_df.columns if c != 'patient_id']
    drug_names = [d.rsplit('_', 1)[0] for d in drug_cols]

    return model, dataset, config, drug_cols, drug_names, gen_names, tra_names, pro_names, common


# ─── 1. Modality Importance via Ablation ─────────────────────────────
def modality_ablation(model, dataset, drug_cols, drug_names):
    """Measure each modality's contribution by zeroing it out."""
    print("\n=== Modality Ablation ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)

    modalities = ['genomic', 'transcriptomic', 'proteomic', 'histology', 'full']
    results = {}

    for ablate in modalities:
        all_pred, all_true = [], []

        for batch in loader:
            g = batch['genomic'].to(DEVICE)
            t = batch['transcriptomic'].to(DEVICE)
            p = batch['proteomic'].to(DEVICE)
            y = batch['target']

            kwargs = {}
            if 'histology' in batch:
                kwargs['histology'] = batch['histology'].to(DEVICE)
                kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

            # Ablate by zeroing
            if ablate == 'genomic':
                g = torch.zeros_like(g)
            elif ablate == 'transcriptomic':
                t = torch.zeros_like(t)
            elif ablate == 'proteomic':
                p = torch.zeros_like(p)
            elif ablate == 'histology':
                kwargs = {}  # no histology

            with torch.no_grad():
                out = model(g, t, p, **kwargs)['prediction'].cpu().numpy()

            # Inverse transform
            out = dataset.scalers['ic50'].inverse_transform(out)
            true = dataset.scalers['ic50'].inverse_transform(y.numpy())
            all_pred.append(out)
            all_true.append(true)

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        # Per-drug PCC
        drug_pcc = {}
        for i, drug in enumerate(drug_names):
            try:
                pcc, _ = pearsonr(all_true[:, i], all_pred[:, i])
            except:
                pcc = 0
            drug_pcc[drug] = pcc

        pcc_global, _ = pearsonr(all_true.flatten(), all_pred.flatten())
        results[ablate] = {
            'pcc_global': pcc_global,
            'pcc_drug_mean': np.mean(list(drug_pcc.values())),
            'drug_pcc': drug_pcc,
        }
        print(f"  {ablate:20s} | PCC_global={pcc_global:.4f} | PCC_drug={np.mean(list(drug_pcc.values())):.4f}")

    # Compute importance = drop in PCC when ablated
    full_pcc = results['full']['pcc_drug_mean']
    importance = {}
    for mod in ['genomic', 'transcriptomic', 'proteomic', 'histology']:
        drop = full_pcc - results[mod]['pcc_drug_mean']
        importance[mod] = drop
        print(f"  Importance of {mod}: {drop:.4f} (PCC drop)")

    return results, importance


# ─── 2. Integrated Gradients for Feature Attribution ─────────────────
def integrated_gradients(model, dataset, drug_cols, drug_names, gen_names, tra_names, pro_names, n_steps=50):
    """Compute integrated gradients for omics features per drug."""
    print("\n=== Integrated Gradients ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)

    n_drugs = len(drug_cols)
    gen_dim = len(gen_names)
    tra_dim = len(tra_names)
    pro_dim = len(pro_names)

    # Accumulate attributions
    gen_attr = np.zeros((n_drugs, gen_dim))
    tra_attr = np.zeros((n_drugs, tra_dim))
    pro_attr = np.zeros((n_drugs, pro_dim))
    n_samples = 0

    for batch in loader:
        g = batch['genomic'].to(DEVICE).requires_grad_(True)
        t = batch['transcriptomic'].to(DEVICE).requires_grad_(True)
        p = batch['proteomic'].to(DEVICE).requires_grad_(True)

        kwargs = {}
        if 'histology' in batch:
            kwargs['histology'] = batch['histology'].to(DEVICE)
            kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

        B = g.shape[0]

        # Baseline = zeros
        g_base = torch.zeros_like(g)
        t_base = torch.zeros_like(t)
        p_base = torch.zeros_like(p)

        # Integrated gradients
        g_ig = torch.zeros_like(g)
        t_ig = torch.zeros_like(t)
        p_ig = torch.zeros_like(p)

        for step in range(n_steps + 1):
            alpha = step / n_steps
            g_interp = g_base + alpha * (g - g_base)
            t_interp = t_base + alpha * (t - t_base)
            p_interp = p_base + alpha * (p - p_base)

            g_interp = g_interp.detach().requires_grad_(True)
            t_interp = t_interp.detach().requires_grad_(True)
            p_interp = p_interp.detach().requires_grad_(True)

            out = model(g_interp, t_interp, p_interp, **kwargs)['prediction']

            for d in range(n_drugs):
                out[:, d].sum().backward(retain_graph=(d < n_drugs - 1))

                if g_interp.grad is not None:
                    g_ig[:, :] += g_interp.grad / (n_steps + 1)
                if t_interp.grad is not None:
                    t_ig[:, :] += t_interp.grad / (n_steps + 1)
                if p_interp.grad is not None:
                    p_ig[:, :] += p_interp.grad / (n_steps + 1)

                model.zero_grad()
                if g_interp.grad is not None:
                    g_interp.grad.zero_()
                if t_interp.grad is not None:
                    t_interp.grad.zero_()
                if p_interp.grad is not None:
                    p_interp.grad.zero_()

        # Multiply by input difference
        g_attr_batch = (g_ig * (g - g_base)).detach().cpu().numpy()
        t_attr_batch = (t_ig * (t - t_base)).detach().cpu().numpy()
        p_attr_batch = (p_ig * (p - p_base)).detach().cpu().numpy()

        # This gives per-sample, per-drug attribution — average over samples
        # Shape: (B, dim) — but we need per-drug. Simplified: use mean prediction gradient
        gen_attr += np.abs(g_attr_batch).sum(axis=0)
        tra_attr += np.abs(t_attr_batch).sum(axis=0)
        pro_attr += np.abs(p_attr_batch).sum(axis=0)
        n_samples += B

        if n_samples >= 100:  # Subsample for speed
            break

    gen_attr /= n_samples
    tra_attr /= n_samples
    pro_attr /= n_samples

    print(f"  Computed IG for {n_samples} samples")
    return gen_attr, tra_attr, pro_attr


# ─── 3. Simple Gradient-based Feature Attribution ────────────────────
def gradient_attribution(model, dataset, drug_cols, drug_names, gen_names, tra_names, pro_names):
    """Fast gradient-based attribution: |gradient × input| per drug."""
    print("\n=== Gradient Attribution ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)

    n_drugs = len(drug_cols)
    gen_dim = len(gen_names)
    tra_dim = len(tra_names)
    pro_dim = len(pro_names)

    # Per-drug attributions
    gen_attr = np.zeros((n_drugs, gen_dim))
    tra_attr = np.zeros((n_drugs, tra_dim))
    pro_attr = np.zeros((n_drugs, pro_dim))
    n_samples = 0

    for batch in loader:
        g = batch['genomic'].to(DEVICE).requires_grad_(True)
        t = batch['transcriptomic'].to(DEVICE).requires_grad_(True)
        p = batch['proteomic'].to(DEVICE).requires_grad_(True)

        kwargs = {}
        if 'histology' in batch:
            kwargs['histology'] = batch['histology'].to(DEVICE)
            kwargs['histo_mask'] = batch['histo_mask'].to(DEVICE)

        out = model(g, t, p, **kwargs)['prediction']  # (B, n_drugs)
        B = g.shape[0]

        for d in range(n_drugs):
            model.zero_grad()
            if g.grad is not None: g.grad.zero_()
            if t.grad is not None: t.grad.zero_()
            if p.grad is not None: p.grad.zero_()

            out[:, d].sum().backward(retain_graph=(d < n_drugs - 1))

            gen_attr[d] += (g.grad * g).abs().sum(dim=0).detach().cpu().numpy()
            tra_attr[d] += (t.grad * t).abs().sum(dim=0).detach().cpu().numpy()
            pro_attr[d] += (p.grad * p).abs().sum(dim=0).detach().cpu().numpy()

        n_samples += B

    gen_attr /= n_samples
    tra_attr /= n_samples
    pro_attr /= n_samples

    print(f"  Computed gradient attribution for {n_samples} samples, {n_drugs} drugs")
    return gen_attr, tra_attr, pro_attr


# ─── 4. ABMIL Attention Analysis ─────────────────────────────────────
def abmil_attention_analysis(model, dataset, drug_names):
    """Extract and analyze ABMIL attention weights."""
    print("\n=== ABMIL Attention Analysis ===")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_4modal)

    attention_stats = []

    for i, batch in enumerate(loader):
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)

        if 'histology' not in batch:
            continue

        h = batch['histology'].to(DEVICE)
        hm = batch['histo_mask'].to(DEVICE)

        with torch.no_grad():
            result = model(g, t, p, histology=h, histo_mask=hm)

        if 'histo_attention' in result:
            attn = result['histo_attention'].cpu().numpy()[0]  # (n_tokens, N_patches)
            n_patches = hm[0].sum().item()

            # Attention entropy (higher = more uniform)
            attn_valid = attn[:, :n_patches]
            entropy = -np.sum(attn_valid * np.log(attn_valid + 1e-10), axis=-1).mean()

            # Top-k concentration
            top10_ratio = np.sort(attn_valid, axis=-1)[:, -10:].sum() / attn_valid.sum()

            attention_stats.append({
                'patient': dataset.pids[i],
                'n_patches': n_patches,
                'entropy': entropy,
                'top10_concentration': top10_ratio,
            })

        if len(attention_stats) >= 100:
            break

    if attention_stats:
        df_attn = pd.DataFrame(attention_stats)
        print(f"  Analyzed {len(df_attn)} patients")
        print(f"  Mean patches: {df_attn['n_patches'].mean():.0f}")
        print(f"  Mean entropy: {df_attn['entropy'].mean():.3f}")
        print(f"  Mean top-10 concentration: {df_attn['top10_concentration'].mean():.3f}")
        return df_attn

    return pd.DataFrame()


# ─── 5. Visualization ────────────────────────────────────────────────

def plot_modality_importance(importance, ablation_results, drug_names, save_dir):
    """Figure 5A: Modality importance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5A: Overall modality importance
    ax = axes[0]
    mods = ['Genomic', 'Transcriptomic', 'Proteomic', 'Histology']
    mod_keys = ['genomic', 'transcriptomic', 'proteomic', 'histology']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    vals = [importance[k] for k in mod_keys]

    bars = ax.bar(mods, vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('PCC Drop (Δ)', fontsize=12)
    ax.set_title('A. Modality Importance\n(PCC drop when ablated)', fontsize=13)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.tick_params(axis='x', rotation=15)

    # 5B: Per-drug modality importance
    ax = axes[1]
    full_drug_pcc = ablation_results['full']['drug_pcc']
    x = np.arange(len(drug_names))
    width = 0.2

    for j, (mod, mod_key, color) in enumerate(zip(mods, mod_keys, colors)):
        drops = []
        for drug in drug_names:
            drop = full_drug_pcc[drug] - ablation_results[mod_key]['drug_pcc'][drug]
            drops.append(drop)
        ax.bar(x + j * width, drops, width, label=mod, color=color,
               edgecolor='black', linewidth=0.3)

    ax.set_ylabel('PCC Drop (Δ)', fontsize=12)
    ax.set_title('B. Per-Drug Modality Importance', fontsize=13)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig5_interpretability.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig5_interpretability.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Fig5_interpretability")


def plot_top_features(gen_attr, tra_attr, pro_attr, gen_names, tra_names, pro_names,
                      drug_names, save_dir, top_k=15):
    """Figure S2: Top features per modality (heatmap)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, attr, names, title in [
        (axes[0], gen_attr, gen_names, 'Genomic (Mutations)'),
        (axes[1], tra_attr, tra_names, 'Transcriptomic (Genes)'),
        (axes[2], pro_attr, pro_names, 'Proteomic (RPPA)'),
    ]:
        # Get top-k features across all drugs
        mean_importance = attr.mean(axis=0)
        top_idx = np.argsort(mean_importance)[-top_k:][::-1]
        top_names = [names[i] for i in top_idx]
        top_data = attr[:, top_idx].T  # (top_k, n_drugs)

        # Normalize per row for visualization
        row_max = top_data.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        top_data_norm = top_data / row_max

        im = ax.imshow(top_data_norm, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xticks(range(len(drug_names)))
        ax.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.6, label='Relative Importance')

    plt.suptitle('Feature Attribution by Modality and Drug', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'FigS2_feature_attribution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'FigS2_feature_attribution.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved FigS2_feature_attribution")


def plot_4modal_comparison(save_dir):
    """Figure 5C: 3-modal vs 4-modal per-drug comparison."""
    # Load both results
    with open(PROJECT_ROOT / "results" / "phase3_3modal_baseline" / "cv_results.json") as f:
        r3 = json.load(f)
    with open(os.path.join(RESULTS, "cv_results.json")) as f:
        r4 = json.load(f)

    drug_cols = r3['drugs']
    drug_names = [d.rsplit('_', 1)[0] for d in drug_cols]

    # Average per-drug PCC across folds
    pcc_3 = {}
    pcc_4 = {}
    for drug, name in zip(drug_cols, drug_names):
        pcc_3[name] = np.mean([float(fold[drug]['pcc']) for fold in r3['drug_metrics_per_fold']])
        pcc_4[name] = np.mean([float(fold[drug]['pcc']) for fold in r4['drug_metrics_per_fold']])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(drug_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, [pcc_3[d] for d in drug_names], width,
                   label='3-modal', color='#78909C', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [pcc_4[d] for d in drug_names], width,
                   label='4-modal (+H&E)', color='#E91E63', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('PCC (per-drug)', fontsize=12)
    ax.set_title('C. 3-Modal vs 4-Modal Per-Drug Prediction Performance', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(drug_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(pcc_3.values()), max(pcc_4.values())) * 1.15)

    # Annotate improvements
    for i, drug in enumerate(drug_names):
        diff = pcc_4[drug] - pcc_3[drug]
        if diff > 0:
            y_pos = max(pcc_3[drug], pcc_4[drug]) + 0.01
            ax.text(i, y_pos, f'+{diff:.3f}', ha='center', fontsize=7, color='#E91E63', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Fig5C_3v4_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'Fig5C_3v4_comparison.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Fig5C_3v4_comparison")


def plot_attention_stats(df_attn, save_dir):
    """Figure S3: ABMIL attention statistics."""
    if df_attn.empty:
        print("  No attention data — skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df_attn['n_patches'], bins=30, color='#7986CB', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Number of Patches')
    axes[0].set_ylabel('Count')
    axes[0].set_title('A. Patches per Patient')

    axes[1].hist(df_attn['entropy'], bins=30, color='#4DB6AC', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Attention Entropy')
    axes[1].set_ylabel('Count')
    axes[1].set_title('B. Attention Entropy Distribution')

    axes[2].scatter(df_attn['n_patches'], df_attn['top10_concentration'],
                    alpha=0.5, s=15, color='#FF7043')
    axes[2].set_xlabel('Number of Patches')
    axes[2].set_ylabel('Top-10 Patch Concentration')
    axes[2].set_title('C. Attention Concentration')

    plt.suptitle('ABMIL Attention Statistics', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'FigS3_attention_stats.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'FigS3_attention_stats.tiff'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved FigS3_attention_stats")


def save_results(importance, ablation_results, gen_attr, tra_attr, pro_attr,
                 gen_names, tra_names, pro_names, drug_names, save_dir):
    """Save analysis results as JSON and CSV."""
    out_dir = os.path.join(save_dir, 'interpretability')
    os.makedirs(out_dir, exist_ok=True)

    # Modality importance
    with open(os.path.join(out_dir, 'modality_importance.json'), 'w') as f:
        json.dump({
            'importance_pcc_drop': {k: float(v) for k, v in importance.items()},
            'ablation_pcc_global': {k: float(v['pcc_global']) for k, v in ablation_results.items()},
            'ablation_pcc_drug_mean': {k: float(v['pcc_drug_mean']) for k, v in ablation_results.items()},
        }, f, indent=2)

    # Top features per drug
    for d, drug in enumerate(drug_names):
        rows = []
        for name, val in sorted(zip(gen_names, gen_attr[d]), key=lambda x: -x[1])[:20]:
            rows.append({'feature': name, 'modality': 'genomic', 'importance': float(val)})
        for name, val in sorted(zip(tra_names, tra_attr[d]), key=lambda x: -x[1])[:20]:
            rows.append({'feature': name, 'modality': 'transcriptomic', 'importance': float(val)})
        for name, val in sorted(zip(pro_names, pro_attr[d]), key=lambda x: -x[1])[:20]:
            rows.append({'feature': name, 'modality': 'proteomic', 'importance': float(val)})

        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'top_features_{drug}.csv'), index=False)

    print(f"  Saved results to {out_dir}")


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load
    model, dataset, config, drug_cols, drug_names, gen_names, tra_names, pro_names, pids = \
        load_model_and_data()
    print(f"Loaded model ({sum(p.numel() for p in model.parameters()):,} params) and {len(pids)} patients")

    # 1. Modality ablation
    ablation_results, importance = modality_ablation(model, dataset, drug_cols, drug_names)

    # 2. Gradient attribution (fast alternative to SHAP/IG for large models)
    gen_attr, tra_attr, pro_attr = gradient_attribution(
        model, dataset, drug_cols, drug_names, gen_names, tra_names, pro_names
    )

    # 3. ABMIL attention
    df_attn = abmil_attention_analysis(model, dataset, drug_names)

    # 4. Plots
    print("\n=== Generating Figures ===")
    plot_modality_importance(importance, ablation_results, drug_names, FIG_DIR)
    plot_top_features(gen_attr, tra_attr, pro_attr, gen_names, tra_names, pro_names,
                      drug_names, FIG_DIR)
    plot_4modal_comparison(FIG_DIR)
    plot_attention_stats(df_attn, FIG_DIR)

    # 5. Save results
    save_results(importance, ablation_results, gen_attr, tra_attr, pro_attr,
                 gen_names, tra_names, pro_names, drug_names, FIG_DIR)

    print("\n=== Interpretability analysis complete ===")
