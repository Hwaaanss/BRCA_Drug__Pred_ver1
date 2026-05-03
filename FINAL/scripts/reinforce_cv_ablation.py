#!/usr/bin/env python3
"""Per-modality CV-averaged inference-time ablation.

Replaces the single-fold "H&E = 89.5%" claim with 5-fold CV mean ± SD drops,
enabling proper uncertainty quantification on modality importance.

Output: results/reinforce/cv_ablation.json
"""
import os, sys, json, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[2])).resolve()
FINAL_ROOT = PROJECT_ROOT / "FINAL"
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal, train_epoch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
OUT_DIR = FINAL_ROOT / "results" / "reinforce"
os.makedirs(OUT_DIR, exist_ok=True)

DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def evaluate_with_ablation(model, loader, scalers, drop=None):
    """Evaluate with optional modality zeroing at inference time.
    drop in {None, 'genomic','transcriptomic','proteomic','histology'}
    """
    model.eval()
    preds, trues = [], []
    for batch in loader:
        g = batch['genomic'].to(DEVICE)
        t = batch['transcriptomic'].to(DEVICE)
        p = batch['proteomic'].to(DEVICE)
        y = batch['target'].to(DEVICE)
        if drop == 'genomic': g = torch.zeros_like(g)
        if drop == 'transcriptomic': t = torch.zeros_like(t)
        if drop == 'proteomic': p = torch.zeros_like(p)
        kw = {}
        if 'histology' in batch and drop != 'histology':
            kw['histology'] = batch['histology'].to(DEVICE)
            kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
        out = model(g, t, p, **kw)['prediction']
        preds.append(out.cpu().numpy()); trues.append(y.cpu().numpy())
    P = np.concatenate(preds); T = np.concatenate(trues)
    if scalers and 'ic50' in scalers:
        P = scalers['ic50'].inverse_transform(P)
        T = scalers['ic50'].inverse_transform(T)
    per_drug = []
    for i in range(P.shape[1]):
        try: r, _ = pearsonr(T[:, i], P[:, i])
        except Exception: r = 0.0
        per_drug.append(float(r))
    return float(np.mean(per_drug)), per_drug


def main():
    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    gen_ids = set(gen_df['patient_id']); tra_ids = set(tra_df['patient_id'])
    pro_ids = set(pro_df['patient_id']); ic_ids = set(ic50_df.index)
    common = sorted(gen_ids & tra_ids & pro_ids & ic_ids)
    histo_ids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(common) & histo_ids)
    log(f"4-modal patients: {len(common)}")

    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])

    config = get_default_config(
        genomic_dim=gen_dim, n_pathways=tra_dim,
        proteomic_dim=pro_dim, n_drugs=len(DRUGS), use_histology=True,
    )
    config['task'] = 'regression'
    config['modality_dropout'] = 0.1
    config['hidden_dim'] = 256

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    conditions = ['full', 'drop_genomic', 'drop_transcriptomic', 'drop_proteomic', 'drop_histology']
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(common)):
        log(f"=== Fold {fold+1}/5 ===")
        tr_ids = [common[i] for i in tr_idx]
        va_ids = [common[i] for i in va_idx]

        tr_ds = MultiDrugDataset4Modal(tr_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, scalers=tr_ds.scalers)
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=4,
                               collate_fn=collate_4modal, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=4,
                               collate_fn=collate_4modal)

        model = PathOmicDRP(config).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=3e-6)
        crit = nn.HuberLoss(delta=1.0)

        best_loss, best_state, patience_ct = float('inf'), None, 0
        for ep in range(100):
            tl = train_epoch(model, tr_loader, opt, crit, use_histo=True)
            model.eval()
            with torch.no_grad():
                vl = 0.0; n = 0
                for b in va_loader:
                    g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
                    p = b['proteomic'].to(DEVICE); y = b['target'].to(DEVICE)
                    kw = {'histology': b['histology'].to(DEVICE), 'histo_mask': b['histo_mask'].to(DEVICE)}
                    o = model(g, t, p, **kw)['prediction']
                    vl += crit(o, y).item() * len(y); n += len(y)
                vl /= n
            sch.step()
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ct = 0
            else:
                patience_ct += 1
            if patience_ct >= 15:
                log(f"  early stop ep {ep+1}")
                break
        model.load_state_dict(best_state); model.to(DEVICE)

        # save fold model
        torch.save(best_state, os.path.join(OUT_DIR, f"fold{fold+1}_model.pt"))

        fold_res = {}
        for cond in conditions:
            drop_name = None if cond == 'full' else cond.replace('drop_', '')
            mean_pcc, per_drug = evaluate_with_ablation(model, va_loader, tr_ds.scalers, drop=drop_name)
            fold_res[cond] = {'pcc_drug_mean': mean_pcc, 'per_drug': per_drug}
            log(f"  {cond:25s}: PCC_drug={mean_pcc:.4f}")
        fold_results.append(fold_res)

        with open(os.path.join(OUT_DIR, 'cv_ablation_partial.json'), 'w') as f:
            json.dump({'folds_done': fold+1, 'fold_results': fold_results}, f, indent=2, default=float)

    # Aggregate
    summary = {'drugs': DRUGS, 'conditions': conditions, 'per_fold': fold_results}
    agg = {}
    full_means = np.array([fr['full']['pcc_drug_mean'] for fr in fold_results])
    for cond in conditions:
        vals = np.array([fr[cond]['pcc_drug_mean'] for fr in fold_results])
        drops = full_means - vals
        agg[cond] = {
            'pcc_drug_mean_per_fold': vals.tolist(),
            'pcc_drug_mean': float(vals.mean()),
            'pcc_drug_std':  float(vals.std(ddof=1)),
            'drop_per_fold': drops.tolist(),
            'drop_mean': float(drops.mean()),
            'drop_std':  float(drops.std(ddof=1)),
            'drop_ci95': [float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))],
        }
    # Relative importance (share of total drop across modalities)
    drop_means = {c: agg[c]['drop_mean'] for c in conditions if c != 'full'}
    total = sum(max(0, v) for v in drop_means.values()) or 1e-9
    for c in drop_means:
        agg[c]['relative_importance_pct_of_total_drop'] = 100.0 * max(0, drop_means[c]) / total
    summary['aggregate'] = agg

    with open(os.path.join(OUT_DIR, 'cv_ablation.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log("Saved cv_ablation.json")
    log("Aggregate drops (mean ± std):")
    for c in conditions:
        if c == 'full': continue
        a = agg[c]
        log(f"  {c:25s}: drop={a['drop_mean']:+.4f} ± {a['drop_std']:.4f} "
            f"[{a['drop_ci95'][0]:+.4f}, {a['drop_ci95'][1]:+.4f}] "
            f"rel={a.get('relative_importance_pct_of_total_drop', 0):.1f}%")


if __name__ == '__main__':
    main()
