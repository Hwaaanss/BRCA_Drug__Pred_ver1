#!/usr/bin/env python3
"""SOTA benchmarking for Genome Medicine submission.

Re-trains three published multi-omics / pathomics drug-response models on the
same TCGA-BRCA 4-modal 431-patient cohort, identical 5-fold CV (seed=42),
identical 13-drug target panel, and reports per-drug Pearson + mean-drug
Pearson directly comparable to PathOmicDRP.

Implemented baselines (faithful to the published architecture; adapted to
multi-drug IC50 regression — caveats documented in manuscript):

  1. PathomicFusion (Chen et al. 2020, IEEE TMI)
     - Per-modality encoders + gated bilinear (Kronecker) fusion.
     - ABMIL for histology patches, MLP for omics.
     - We use bilinear fusion between (gen⊕trans⊕prot) and histology.

  2. MOLI (Sharifi-Noghabi et al. 2019, Bioinformatics)
     - Three parallel feed-forward encoders (gen / trans / prot).
     - Concatenate bottleneck features -> classifier (here regressor).
     - Original used triplet loss on binary response; for regression we
       add an L1 consistency penalty between modality embeddings.

  3. Super.FELT-lite (Park et al. 2021, Bioinformatics)
     - Feature selection via ElasticNet per modality, followed by
       modality-specific subnets and late concat fusion.

Output:
  results/benchmark/sota_comparison.json
  results/benchmark/sota_per_drug.csv
"""
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
OUT = PROJECT_ROOT / "results" / "benchmark"
os.makedirs(OUT, exist_ok=True)

DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]

EPOCHS = 60
PATIENCE = 10
BATCH = 16
HDIM = 256

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [SOTA] {m}", flush=True)


# ---------- PathomicFusion ----------
class ABMIL(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.attn_V = nn.Linear(in_dim, hidden)
        self.attn_U = nn.Linear(in_dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)
        self.proj = nn.Linear(in_dim, hidden)

    def forward(self, x, mask=None):
        # x: (B, N, D); mask: (B, N) true=valid
        V = torch.tanh(self.attn_V(x)); U = torch.sigmoid(self.attn_U(x))
        a = self.attn_w(V * U).squeeze(-1)  # (B, N)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        w = torch.softmax(a, dim=1)  # (B, N)
        pooled = (w.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return self.proj(pooled)


class PathomicFusion(nn.Module):
    """Chen 2020: gated bilinear fusion of histology + omics embeddings."""
    def __init__(self, gen_dim, trans_dim, prot_dim, hist_dim=1024,
                 h=HDIM, n_drugs=len(DRUGS)):
        super().__init__()
        # Omics tower: concat gen+trans+prot then SNN-like MLP
        omics_in = gen_dim + trans_dim + prot_dim
        self.omics = nn.Sequential(
            nn.Linear(omics_in, h * 2), nn.ELU(), nn.AlphaDropout(0.25),
            nn.Linear(h * 2, h), nn.ELU(), nn.AlphaDropout(0.25),
        )
        # Histology tower
        self.histo = ABMIL(hist_dim, h)
        # Gated bilinear fusion (Kronecker with gating)
        self.gate_o = nn.Linear(h, h)
        self.gate_h = nn.Linear(h, h)
        self.bilinear = nn.Bilinear(h, h, h)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(h, n_drugs),
        )

    def forward(self, gen, trans, prot, histology=None, histo_mask=None):
        omics = torch.cat([gen, trans, prot], dim=1)
        zo = self.omics(omics)
        if histology is not None:
            zh = self.histo(histology, histo_mask)
        else:
            zh = torch.zeros_like(zo)
        go = torch.sigmoid(self.gate_o(zo)) * zo
        gh = torch.sigmoid(self.gate_h(zh)) * zh
        f = self.bilinear(go, gh) + go + gh
        return self.head(f)


# ---------- MOLI ----------
class MOLI(nn.Module):
    """Sharifi-Noghabi 2019: 3 parallel encoders -> concat -> classifier.
    Regression adaptation: replace triplet+classification with L1-consistency + MSE."""
    def __init__(self, gen_dim, trans_dim, prot_dim, h=64, n_drugs=len(DRUGS)):
        super().__init__()
        def enc(d):
            return nn.Sequential(
                nn.Linear(d, h * 2), nn.ReLU(), nn.BatchNorm1d(h * 2), nn.Dropout(0.3),
                nn.Linear(h * 2, h), nn.ReLU(), nn.BatchNorm1d(h),
            )
        self.e_g = enc(gen_dim); self.e_t = enc(trans_dim); self.e_p = enc(prot_dim)
        self.head = nn.Sequential(
            nn.Linear(h * 3, h * 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h * 2, n_drugs),
        )

    def forward(self, gen, trans, prot, histology=None, histo_mask=None):
        zg = self.e_g(gen); zt = self.e_t(trans); zp = self.e_p(prot)
        z = torch.cat([zg, zt, zp], dim=1)
        return self.head(z), (zg, zt, zp)


# ---------- Super.FELT-lite ----------
class SuperFELT(nn.Module):
    """Park 2021: per-modality feature selection (ElasticNet) then deep subnets + late concat."""
    def __init__(self, gen_sel, trans_sel, prot_sel, h=HDIM, n_drugs=len(DRUGS)):
        super().__init__()
        self.gen_idx = torch.tensor(gen_sel, dtype=torch.long)
        self.trans_idx = torch.tensor(trans_sel, dtype=torch.long)
        self.prot_idx = torch.tensor(prot_sel, dtype=torch.long)
        def sub(d):
            return nn.Sequential(
                nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.3),
                nn.Linear(h, h // 2), nn.ReLU(), nn.BatchNorm1d(h // 2),
            )
        self.gen_sub = sub(len(gen_sel))
        self.trans_sub = sub(len(trans_sel))
        self.prot_sub = sub(len(prot_sel))
        self.head = nn.Sequential(
            nn.Linear((h // 2) * 3, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, n_drugs),
        )

    def forward(self, gen, trans, prot, histology=None, histo_mask=None):
        g = gen.index_select(1, self.gen_idx.to(gen.device))
        t = trans.index_select(1, self.trans_idx.to(trans.device))
        p = prot.index_select(1, self.prot_idx.to(prot.device))
        z = torch.cat([self.gen_sub(g), self.trans_sub(t), self.prot_sub(p)], dim=1)
        return self.head(z)


def train_one(model, tr_loader, va_loader, scalers, model_name='base', use_histo=True, aux=False):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=3e-6)
    crit = nn.HuberLoss(delta=1.0)
    best, best_state, pat = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in tr_loader:
            g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
            p = b['proteomic'].to(DEVICE); y = b['target'].to(DEVICE)
            kw = {}
            if use_histo and 'histology' in b:
                kw['histology'] = b['histology'].to(DEVICE)
                kw['histo_mask'] = b['histo_mask'].to(DEVICE)
            opt.zero_grad()
            out = model(g, t, p, **kw)
            if aux:
                pred, (zg, zt, zp) = out
                loss = crit(pred, y) + 0.1 * (
                    torch.mean(torch.abs(zg - zt)) + torch.mean(torch.abs(zt - zp)) +
                    torch.mean(torch.abs(zg - zp))
                )
            else:
                pred = out
                loss = crit(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in va_loader:
                g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
                p = b['proteomic'].to(DEVICE); y = b['target'].to(DEVICE)
                kw = {}
                if use_histo and 'histology' in b:
                    kw['histology'] = b['histology'].to(DEVICE)
                    kw['histo_mask'] = b['histo_mask'].to(DEVICE)
                o = model(g, t, p, **kw)
                if aux: o = o[0]
                vl += crit(o, y).item() * len(y); n += len(y)
            vl /= n
        sch.step()
        if vl < best:
            best = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            break
    model.load_state_dict(best_state); model.to(DEVICE)
    # eval PCC per drug
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for b in va_loader:
            g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
            p = b['proteomic'].to(DEVICE); y = b['target'].to(DEVICE)
            kw = {}
            if use_histo and 'histology' in b:
                kw['histology'] = b['histology'].to(DEVICE)
                kw['histo_mask'] = b['histo_mask'].to(DEVICE)
            o = model(g, t, p, **kw)
            if aux: o = o[0]
            preds.append(o.cpu().numpy()); trues.append(y.cpu().numpy())
    P = np.concatenate(preds); T = np.concatenate(trues)
    P = scalers['ic50'].inverse_transform(P)
    T = scalers['ic50'].inverse_transform(T)
    per_drug = []
    for j in range(P.shape[1]):
        try: r, _ = pearsonr(T[:, j], P[:, j])
        except Exception: r = 0.0
        per_drug.append(float(r))
    global_pcc, _ = pearsonr(T.flatten(), P.flatten())
    return {'pcc_drug_mean': float(np.mean(per_drug)),
            'pcc_drug_per': per_drug,
            'pcc_global': float(global_pcc),
            'predictions': P.tolist(),
            'truths': T.tolist()}


def select_features_elasticnet(X, Y, k):
    """Return indices of top-k features by |coef| from a multi-output ElasticNet (single target = mean Y)."""
    y = Y.mean(axis=1)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
    en.fit(X, y)
    coefs = np.abs(en.coef_)
    if len(coefs) <= k: return list(range(len(coefs)))
    return np.argsort(coefs)[-k:].tolist()


def main():
    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    common = sorted(set(gen_df['patient_id']) & set(tra_df['patient_id']) &
                    set(pro_df['patient_id']) & set(ic50_df.index))
    histo_ids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(common) & histo_ids)
    log(f"4-modal patients: {len(common)}")

    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {'PathomicFusion': [], 'MOLI': [], 'SuperFELT': []}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(common)):
        log(f"=== Fold {fold+1}/5 ===")
        tr_ids = [common[i] for i in tr_idx]; va_ids = [common[i] for i in va_idx]

        tr_ds = MultiDrugDataset4Modal(tr_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, scalers=tr_ds.scalers)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                               collate_fn=collate_4modal, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                               collate_fn=collate_4modal)

        # --- PathomicFusion (4-modal) ---
        log("  [PathomicFusion] training...")
        pf = PathomicFusion(gen_dim, tra_dim, pro_dim, hist_dim=1024).to(DEVICE)
        r = train_one(pf, tr_loader, va_loader, tr_ds.scalers, 'PathomicFusion', use_histo=True)
        r.pop('predictions', None); r.pop('truths', None)
        log(f"    PCC_drug={r['pcc_drug_mean']:.4f}  PCC_global={r['pcc_global']:.4f}")
        results['PathomicFusion'].append(r)

        # --- MOLI (3-modal) ---
        log("  [MOLI] training...")
        moli = MOLI(gen_dim, tra_dim, pro_dim, h=64).to(DEVICE)
        r = train_one(moli, tr_loader, va_loader, tr_ds.scalers, 'MOLI', use_histo=False, aux=True)
        r.pop('predictions', None); r.pop('truths', None)
        log(f"    PCC_drug={r['pcc_drug_mean']:.4f}  PCC_global={r['pcc_global']:.4f}")
        results['MOLI'].append(r)

        # --- SuperFELT (3-modal with EN feature selection) ---
        log("  [SuperFELT] feature selection + training...")
        Yt = tr_ds.ic50_data
        k_g = min(50, gen_dim); k_t = 500; k_p = min(100, pro_dim)
        gen_sel = select_features_elasticnet(tr_ds.gen_data, Yt, k_g)
        trans_sel = select_features_elasticnet(tr_ds.tra_data, Yt, k_t)
        prot_sel = select_features_elasticnet(tr_ds.pro_data, Yt, k_p)
        sf = SuperFELT(gen_sel, trans_sel, prot_sel).to(DEVICE)
        r = train_one(sf, tr_loader, va_loader, tr_ds.scalers, 'SuperFELT', use_histo=False)
        r.pop('predictions', None); r.pop('truths', None)
        log(f"    PCC_drug={r['pcc_drug_mean']:.4f}  PCC_global={r['pcc_global']:.4f}")
        results['SuperFELT'].append(r)

        # checkpoint
        with open(os.path.join(OUT, 'sota_comparison_partial.json'), 'w') as f:
            json.dump({'folds_done': fold + 1, 'results': results, 'drugs': DRUGS}, f, indent=2)

    # aggregate
    agg = {}
    for m, folds in results.items():
        drugs_mean = np.mean([f['pcc_drug_mean'] for f in folds])
        drugs_std = np.std([f['pcc_drug_mean'] for f in folds], ddof=1)
        global_mean = np.mean([f['pcc_global'] for f in folds])
        per_drug_arr = np.array([f['pcc_drug_per'] for f in folds])  # (5, 13)
        agg[m] = {
            'pcc_drug_mean_mean': float(drugs_mean),
            'pcc_drug_mean_std': float(drugs_std),
            'pcc_global_mean': float(global_mean),
            'per_drug_mean': per_drug_arr.mean(axis=0).tolist(),
            'per_drug_std': per_drug_arr.std(axis=0, ddof=1).tolist(),
        }

    # Add PathOmicDRP reference (from reinforce cv_ablation.json full condition)
    try:
        ref = json.load(open(PROJECT_ROOT / "results" / "reinforce" / "cv_ablation.json"))
        full = ref['aggregate']['full']
        # per_drug from per_fold full
        pdrug = np.array([f['full']['per_drug'] for f in ref['per_fold']])
        agg['PathOmicDRP'] = {
            'pcc_drug_mean_mean': full['pcc_drug_mean'],
            'pcc_drug_mean_std': full['pcc_drug_std'],
            'per_drug_mean': pdrug.mean(axis=0).tolist(),
            'per_drug_std': pdrug.std(axis=0, ddof=1).tolist(),
        }
    except Exception as e:
        log(f"ref load fail: {e}")

    out = {'drugs': DRUGS, 'results': results, 'aggregate': agg}
    with open(os.path.join(OUT, 'sota_comparison.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # per-drug wide table
    rows = []
    for m, a in agg.items():
        row = {'method': m, 'pcc_drug_mean_mean': a.get('pcc_drug_mean_mean')}
        for j, d in enumerate(DRUGS):
            row[d] = a['per_drug_mean'][j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'sota_per_drug.csv'), index=False)
    log("Saved sota_comparison.json + sota_per_drug.csv")
    for m, a in agg.items():
        log(f"  {m:16s}: PCC_drug = {a.get('pcc_drug_mean_mean', float('nan')):.4f} "
            f"± {a.get('pcc_drug_mean_std', 0):.4f}")

if __name__ == '__main__':
    main()
