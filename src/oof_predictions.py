#!/usr/bin/env python3
"""Generate out-of-fold (OOF) predictions from the 5 saved PathOmicDRP fold models.

Loads results/reinforce/fold{1..5}_model.pt, runs inference on each fold's validation
set using the same KFold(shuffle=True, random_state=42) split as reinforce_cv_ablation,
and writes a 431 x 13 OOF prediction matrix in real IC50 units plus a long-format CSV
suitable for per-drug downstream analyses (decision curves, biological validation).

Output:
  results/oof/oof_predictions.csv       (patient_id, fold, 13 drug columns, true_* columns)
  results/oof/oof_predictions_long.csv  (patient_id, fold, drug, pred_IC50, true_IC50)
  results/oof/oof_summary.json          per-drug Pearson / Spearman on OOF
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = DATA_ROOT / "07_integrated"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"
FOLDS_DIR = PROJECT_ROOT / "results" / "reinforce"
OUT = PROJECT_ROOT / "results" / "oof"
os.makedirs(OUT, exist_ok=True)

DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

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

    config = get_default_config(genomic_dim=gen_dim, n_pathways=tra_dim,
                                proteomic_dim=pro_dim, n_drugs=len(DRUGS), use_histology=True)
    config['task'] = 'regression'
    config['hidden_dim'] = 256

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros((len(common), len(DRUGS)), dtype=np.float32)
    oof_true = np.zeros((len(common), len(DRUGS)), dtype=np.float32)
    fold_assign = np.zeros(len(common), dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(common)):
        log(f"Fold {fold+1}")
        tr_ids = [common[i] for i in tr_idx]; va_ids = [common[i] for i in va_idx]
        tr_ds = MultiDrugDataset4Modal(tr_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, scalers=tr_ds.scalers)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=2,
                               collate_fn=collate_4modal)

        model = PathOmicDRP(config).to(DEVICE)
        state = torch.load(os.path.join(FOLDS_DIR, f"fold{fold+1}_model.pt"),
                           map_location=DEVICE, weights_only=True)
        model.load_state_dict(state); model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for b in va_loader:
                g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
                p = b['proteomic'].to(DEVICE); y = b['target'].to(DEVICE)
                kw = {}
                if 'histology' in b:
                    kw['histology'] = b['histology'].to(DEVICE)
                    kw['histo_mask'] = b['histo_mask'].to(DEVICE)
                o = model(g, t, p, **kw)['prediction']
                preds.append(o.cpu().numpy()); trues.append(y.cpu().numpy())
        P = np.concatenate(preds); T = np.concatenate(trues)
        P = tr_ds.scalers['ic50'].inverse_transform(P)
        T = tr_ds.scalers['ic50'].inverse_transform(T)
        for j, i in enumerate(va_idx):
            oof_pred[i] = P[j]; oof_true[i] = T[j]; fold_assign[i] = fold + 1

    # wide
    wide = pd.DataFrame(oof_pred, columns=[f"pred_{d}" for d in DRUGS])
    wide.insert(0, 'patient_id', common); wide.insert(1, 'fold', fold_assign)
    for j, d in enumerate(DRUGS): wide[f"true_{d}"] = oof_true[:, j]
    wide.to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)

    # long
    rows = []
    for i, pid in enumerate(common):
        for j, d in enumerate(DRUGS):
            rows.append({'patient_id': pid, 'fold': int(fold_assign[i]),
                         'drug': d, 'pred_IC50': float(oof_pred[i, j]),
                         'true_IC50': float(oof_true[i, j])})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "oof_predictions_long.csv"), index=False)

    # per-drug summary
    summary = {}
    for j, d in enumerate(DRUGS):
        r, pr = pearsonr(oof_true[:, j], oof_pred[:, j])
        rs, ps = spearmanr(oof_true[:, j], oof_pred[:, j])
        summary[d] = {'pearson_r': float(r), 'pearson_p': float(pr),
                      'spearman_r': float(rs), 'spearman_p': float(ps)}
    r_all, _ = pearsonr(oof_true.flatten(), oof_pred.flatten())
    summary['__global__'] = {'pearson_r': float(r_all), 'n_patients': len(common)}
    with open(os.path.join(OUT, "oof_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote OOF predictions for {len(common)} patients x {len(DRUGS)} drugs")
    log(f"Global PCC (OOF): {r_all:.4f}")

if __name__ == '__main__':
    main()
