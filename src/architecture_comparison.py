#!/usr/bin/env python3
"""
Architecture Comparison: PathOmicDRP cross-attention vs alternative fusion strategies.
All models use same data, same CV splits, same training procedure.

Baselines:
1. Late Fusion MLP (MOLI-style): modality MLPs → concat → prediction
2. Early Fusion MLP: concat all features → single MLP
3. Mean Pooling (no attention): replace ABMIL with mean pooling
4. Self-Attention Only (no cross-attention): concat all tokens → self-attn only
5. PathOmicDRP 3-modal (no histology)
6. PathOmicDRP 4-modal (full, cross-attention)
"""
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda')
DATA_DIR = DATA_ROOT
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
RESULTS_ROOT = PROJECT_ROOT / "results"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════
# Baseline 1: Late Fusion MLP (MOLI-style)
# ═══════════════════════════════════════════
class LateFusionMLP(nn.Module):
    """Each modality has its own MLP, outputs concatenated for prediction."""
    def __init__(self, gen_dim, tra_dim, pro_dim, histo_dim, hidden, n_drugs, use_histo=True):
        super().__init__()
        self.use_histo = use_histo
        self.gen_mlp = nn.Sequential(nn.Linear(gen_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        self.tra_mlp = nn.Sequential(nn.Linear(tra_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        self.pro_mlp = nn.Sequential(nn.Linear(pro_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
        concat_dim = hidden * 3
        if use_histo:
            self.histo_mlp = nn.Sequential(nn.Linear(histo_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden))
            concat_dim = hidden * 4
        self.head = nn.Sequential(nn.Linear(concat_dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_drugs))

    def forward(self, g, t, p, histology=None, histo_mask=None):
        hg = self.gen_mlp(g)
        ht = self.tra_mlp(t)
        hp = self.pro_mlp(p)
        parts = [hg, ht, hp]
        if self.use_histo and histology is not None:
            # Mean pool histology patches
            if histo_mask is not None:
                mask_f = histo_mask.unsqueeze(-1).float()
                h_mean = (histology * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                h_mean = histology.mean(dim=1)
            hh = self.histo_mlp(h_mean)
            parts.append(hh)
        fused = torch.cat(parts, dim=-1)
        return {'prediction': self.head(fused)}


# ═══════════════════════════════════════════
# Baseline 2: Early Fusion MLP
# ═══════════════════════════════════════════
class EarlyFusionMLP(nn.Module):
    """Concatenate all raw features → single deep MLP."""
    def __init__(self, gen_dim, tra_dim, pro_dim, histo_dim, hidden, n_drugs, use_histo=True):
        super().__init__()
        self.use_histo = use_histo
        input_dim = gen_dim + tra_dim + pro_dim
        if use_histo:
            input_dim += histo_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, n_drugs),
        )

    def forward(self, g, t, p, histology=None, histo_mask=None):
        parts = [g, t, p]
        if self.use_histo and histology is not None:
            if histo_mask is not None:
                mask_f = histo_mask.unsqueeze(-1).float()
                h_mean = (histology * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                h_mean = histology.mean(dim=1)
            parts.append(h_mean)
        x = torch.cat(parts, dim=-1)
        return {'prediction': self.mlp(x)}


# ═══════════════════════════════════════════
# Baseline 3: Self-Attention Only (no cross-attention)
# ═══════════════════════════════════════════
class SelfAttnOnly(nn.Module):
    """Tokenize modalities, concatenate all tokens, self-attention only (no cross-attn)."""
    def __init__(self, config):
        super().__init__()
        from model import GenomicEncoder, PathwayTokenizer, ProteomicEncoder, ABMIL, PredictionHead
        d = config['hidden_dim']
        self.gen_enc = GenomicEncoder(config['genomic_dim'], d, config.get('genomic_tokens',8))
        self.path_tok = PathwayTokenizer(config['n_pathways'], 1, d)
        self.pro_enc = ProteomicEncoder(config['proteomic_dim'], d, config.get('proteomic_tokens',16))
        self.use_histo = config.get('use_histology', False)
        if self.use_histo:
            self.histo_enc = ABMIL(config.get('histo_feature_dim',1024), d, n_tokens=config.get('histo_tokens',16))
        # Self-attention only (no cross-attention)
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=d*4, dropout=0.1,
                                       activation='gelu', batch_first=True),
            num_layers=2,
        )
        self.pred = PredictionHead(d, config.get('n_drugs',13), config.get('task','regression'))

    def forward(self, g, t, p, histology=None, histo_mask=None):
        gt = self.gen_enc(g)
        tt = self.path_tok(t)
        pt = self.pro_enc(p)
        tokens = torch.cat([gt, tt, pt], dim=1)
        if self.use_histo and histology is not None:
            ht, _ = self.histo_enc(histology, histo_mask)
            tokens = torch.cat([tokens, ht], dim=1)
        fused = self.self_attn(tokens)
        pred = self.pred(fused)
        return {'prediction': pred}


# ═══════════════════════════════════════════
# Baseline 4: Mean Pooling (no ABMIL attention)
# ═══════════════════════════════════════════
class MeanPoolHistoModel(nn.Module):
    """PathOmicDRP but replace ABMIL with simple mean pooling for H&E."""
    def __init__(self, config):
        super().__init__()
        from model import GenomicEncoder, PathwayTokenizer, ProteomicEncoder, MultiModalFusion, PredictionHead
        d = config['hidden_dim']
        self.gen_enc = GenomicEncoder(config['genomic_dim'], d, config.get('genomic_tokens',8))
        self.path_tok = PathwayTokenizer(config['n_pathways'], 1, d)
        self.pro_enc = ProteomicEncoder(config['proteomic_dim'], d, config.get('proteomic_tokens',16))
        self.use_histo = config.get('use_histology', False)
        if self.use_histo:
            n_tok = config.get('histo_tokens', 16)
            self.histo_proj = nn.Sequential(
                nn.Linear(config.get('histo_feature_dim',1024), d),
                nn.GELU(), nn.Dropout(0.1),
            )
            # Project mean-pooled to multiple tokens
            self.histo_token_proj = nn.Linear(d, n_tok * d)
            self.n_tok = n_tok; self.d = d
        self.fusion = MultiModalFusion(d, n_heads=8, n_layers=2)
        self.pred = PredictionHead(d, config.get('n_drugs',13), config.get('task','regression'))

    def forward(self, g, t, p, histology=None, histo_mask=None):
        gt = self.gen_enc(g)
        tt = self.path_tok(t)
        pt = self.pro_enc(p)
        omics = torch.cat([gt, tt, pt], dim=1)
        ht = None
        if self.use_histo and histology is not None:
            if histo_mask is not None:
                mf = histo_mask.unsqueeze(-1).float()
                h_mean = (histology * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
            else:
                h_mean = histology.mean(dim=1)
            h_proj = self.histo_proj(h_mean)  # (B, d)
            ht = self.histo_token_proj(h_proj).view(-1, self.n_tok, self.d)  # (B, n_tok, d)
        fused = self.fusion(omics, ht)
        return {'prediction': self.pred(fused)}


# ═══════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════
CONFIG_BASED_MODELS = (SelfAttnOnly, MeanPoolHistoModel, PathOmicDRP)

def train_and_eval(model_class, model_kwargs, train_ds, val_ds, n_epochs=100, lr=3e-4, bs=16):
    if model_class in CONFIG_BASED_MODELS:
        model = model_class(model_kwargs).to(DEVICE)
    else:
        model = model_class(**model_kwargs).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              collate_fn=collate_4modal, drop_last=len(train_ds)>bs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0,
                            collate_fn=collate_4modal)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)
    criterion = nn.HuberLoss(delta=1.0)

    best_loss = float('inf'); best_state = None; patience = 15; pat_count = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE)
            p=batch['proteomic'].to(DEVICE); y=batch['target'].to(DEVICE)
            kw = {}
            if 'histology' in batch:
                kw['histology']=batch['histology'].to(DEVICE)
                kw['histo_mask']=batch['histo_mask'].to(DEVICE)
            optimizer.zero_grad()
            out = model(g, t, p, **kw)['prediction']
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Quick val loss
        model.eval()
        vloss = 0; vn = 0
        with torch.no_grad():
            for batch in val_loader:
                g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE)
                p=batch['proteomic'].to(DEVICE); y=batch['target'].to(DEVICE)
                kw = {}
                if 'histology' in batch:
                    kw['histology']=batch['histology'].to(DEVICE)
                    kw['histo_mask']=batch['histo_mask'].to(DEVICE)
                out = model(g, t, p, **kw)['prediction']
                vloss += criterion(out, y).item() * len(y); vn += len(y)
        vl = vloss / vn
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_count = 0
        else:
            pat_count += 1
        if pat_count >= patience:
            break

    # Final eval
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE)
            p=batch['proteomic'].to(DEVICE); y=batch['target']
            kw = {}
            if 'histology' in batch:
                kw['histology']=batch['histology'].to(DEVICE)
                kw['histo_mask']=batch['histo_mask'].to(DEVICE)
            out = model(g, t, p, **kw)['prediction'].cpu().numpy()
            pred_o = train_ds.scalers['ic50'].inverse_transform(out)
            true_o = train_ds.scalers['ic50'].inverse_transform(y.numpy())
            all_pred.append(pred_o); all_true.append(true_o)

    all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)
    pcc_g, _ = pearsonr(all_pred.flatten(), all_true.flatten())
    drug_pccs = []
    for d in range(all_pred.shape[1]):
        try: r, _ = pearsonr(all_pred[:,d], all_true[:,d]); drug_pccs.append(r)
        except: drug_pccs.append(0)

    return float(pcc_g), float(np.mean(drug_pccs)), n_params


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
if __name__ == '__main__':
    log("Loading data...")
    with open(RESULTS_ROOT / "phase3_4modal_full" / "cv_results.json") as f:
        cv = json.load(f)
    config = cv['config']; drug_cols = cv['drugs']
    gen_df = pd.read_csv(f"{DATA_DIR}/07_integrated/X_genomic.csv")
    tra_df = pd.read_csv(f"{DATA_DIR}/07_integrated/X_transcriptomic.csv")
    pro_df = pd.read_csv(f"{DATA_DIR}/07_integrated/X_proteomic.csv")
    ic50_df = pd.read_csv(f"{DATA_DIR}/07_integrated/predicted_IC50_all_drugs.csv", index_col=0)
    hids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    pids = sorted(set(gen_df['patient_id'])&set(tra_df['patient_id'])&set(pro_df['patient_id'])&set(ic50_df.index)&hids)
    log(f"Patients: {len(pids)}")

    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])
    n_drugs = len(drug_cols)

    # Models to compare
    models = {
        'Early Fusion MLP': (EarlyFusionMLP, {'gen_dim':gen_dim,'tra_dim':tra_dim,'pro_dim':pro_dim,'histo_dim':1024,'hidden':256,'n_drugs':n_drugs,'use_histo':True}),
        'Late Fusion MLP\n(MOLI-style)': (LateFusionMLP, {'gen_dim':gen_dim,'tra_dim':tra_dim,'pro_dim':pro_dim,'histo_dim':1024,'hidden':256,'n_drugs':n_drugs,'use_histo':True}),
        'Self-Attn Only\n(no cross-attn)': (SelfAttnOnly, {**config, 'use_histology':True}),
        'Mean Pool Histo\n(no ABMIL)': (MeanPoolHistoModel, {**config, 'use_histology':True}),
        'PathOmicDRP\n3-modal': (PathOmicDRP, {**config, 'use_histology':False}),
        'PathOmicDRP\n4-modal (ours)': (PathOmicDRP, {**config, 'use_histology':True}),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Preserve previously completed results (Early Fusion, Late Fusion from prior run)
    results = {
        'Early Fusion MLP': {
            'pcc_global': (0.9643, 0.0047), 'pcc_drug': (0.7506, 0.0226), 'n_params': 2051085,
        },
        'Late Fusion MLP\n(MOLI-style)': {
            'pcc_global': (0.9629, 0.0060), 'pcc_drug': (0.7468, 0.0295), 'n_params': 1472269,
        },
    }
    skip = set(results.keys())

    for model_name, (model_class, model_kwargs) in models.items():
        if model_name in skip:
            log(f"\n  Skipping (already done): {model_name.replace(chr(10),' ')}")
            continue
        log(f"\n  Training: {model_name.replace(chr(10),' ')}")
        fold_globals, fold_drugs = [], []
        n_params = 0

        for fold, (tri, vai) in enumerate(kf.split(pids)):
            tr_pids = [pids[i] for i in tri]; va_pids = [pids[i] for i in vai]
            use_h = model_kwargs.get('use_histo', model_kwargs.get('use_histology', False))
            hdir = HISTO_DIR if use_h else None

            tr_ds = MultiDrugDataset4Modal(tr_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols, histo_dir=hdir, fit=True)
            va_ds = MultiDrugDataset4Modal(va_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols, histo_dir=hdir, scalers=tr_ds.scalers)

            # For config-based models, need to instantiate differently
            if model_class in (SelfAttnOnly, MeanPoolHistoModel, PathOmicDRP):
                pg, pd_, np_ = train_and_eval(model_class, model_kwargs, tr_ds, va_ds, n_epochs=80, bs=16)
            else:
                pg, pd_, np_ = train_and_eval(model_class, model_kwargs, tr_ds, va_ds, n_epochs=80, bs=16)

            fold_globals.append(pg); fold_drugs.append(pd_); n_params = np_
            log(f"    Fold {fold+1}: PCC_global={pg:.4f}, PCC_drug={pd_:.4f}")

        results[model_name] = {
            'pcc_global': (float(np.mean(fold_globals)), float(np.std(fold_globals))),
            'pcc_drug': (float(np.mean(fold_drugs)), float(np.std(fold_drugs))),
            'n_params': n_params,
        }
        log(f"  → {model_name.replace(chr(10),' ')}: PCC_drug={np.mean(fold_drugs):.4f}±{np.std(fold_drugs):.4f} ({n_params:,} params)")

    # Save
    out_dir = RESULTS_ROOT / "architecture_comparison"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    log("\n═══ ARCHITECTURE COMPARISON SUMMARY ═══")
    log(f"  {'Model':30s} | {'PCC_drug':15s} | {'PCC_global':15s} | {'Params':>10s}")
    log(f"  {'-'*75}")
    for name in sorted(results, key=lambda x: results[x]['pcc_drug'][0], reverse=True):
        v = results[name]
        n = name.replace('\n',' ')
        log(f"  {n:30s} | {v['pcc_drug'][0]:.4f}±{v['pcc_drug'][1]:.4f} | {v['pcc_global'][0]:.4f}±{v['pcc_global'][1]:.4f} | {v['n_params']:>10,}")

    log("\nDone!")
