#!/usr/bin/env python3
"""Extract per-patient 256-d fused embeddings and token-level attention weights
from the 5 saved PathOmicDRP fold models (OOF).

Outputs
  results/oof/oof_embeddings.npy          (431 x 256) float32
  results/oof/oof_embedding_pids.json     order of patient_ids
  results/oof/attention_pool_weights.npz  pooling attention per patient (padded)
  results/oof/attention_histology.npz     histology ABMIL attention (padded)
  results/oof/token_layout.json           describes which token index = which modality
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torch.nn.functional as F

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

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [EMB] {m}", flush=True)


def forward_with_hooks(model, g, t, p, histology=None, histo_mask=None):
    """Re-implement forward exposing intermediate fused tokens and pooling attn."""
    gen_tokens = model.genomic_encoder(g)
    path_tokens = model.pathway_tokenizer(t)
    prot_tokens = model.proteomic_encoder(p)
    omics_tokens = torch.cat([gen_tokens, path_tokens, prot_tokens], dim=1)

    histo_tokens = None; histo_attn = None
    if model.use_histology and histology is not None:
        histo_tokens, histo_attn = model.histology_encoder(histology, histo_mask)

    if histo_tokens is not None:
        for layer in model.fusion.layers:
            omics_tokens, histo_tokens = layer(omics_tokens, histo_tokens)
        all_tokens = torch.cat([omics_tokens, histo_tokens], dim=1)
    else:
        all_tokens = omics_tokens
    fused = model.fusion.self_attn(all_tokens)

    attn_scores = model.prediction_head.pool_attn(fused).squeeze(-1)
    attn_w = F.softmax(attn_scores, dim=-1)
    pooled = torch.bmm(attn_w.unsqueeze(1), fused).squeeze(1)  # (B, D)
    pred = model.prediction_head.head(pooled)
    return pooled, attn_w, fused, histo_attn, pred


def main():
    gen_df = pd.read_csv(os.path.join(BASE, "X_genomic.csv"))
    tra_df = pd.read_csv(os.path.join(BASE, "X_transcriptomic.csv"))
    pro_df = pd.read_csv(os.path.join(BASE, "X_proteomic.csv"))
    ic50_df = pd.read_csv(os.path.join(BASE, "predicted_IC50_all_drugs.csv"), index_col=0)

    common = sorted(set(gen_df['patient_id']) & set(tra_df['patient_id']) &
                    set(pro_df['patient_id']) & set(ic50_df.index))
    histo_ids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(common) & histo_ids)
    log(f"N={len(common)}")

    gen_dim = len([c for c in gen_df.columns if c != 'patient_id'])
    tra_dim = len([c for c in tra_df.columns if c != 'patient_id'])
    pro_dim = len([c for c in pro_df.columns if c != 'patient_id'])

    config = get_default_config(genomic_dim=gen_dim, n_pathways=tra_dim,
                                proteomic_dim=pro_dim, n_drugs=len(DRUGS), use_histology=True)
    config['task'] = 'regression'; config['hidden_dim'] = 256

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    N = len(common); HD = 256
    emb = np.zeros((N, HD), dtype=np.float32)
    # n_omics tokens = 8 gen + 2001 path + 16 prot; + 16 histo = tons of pathway tokens.
    # pooling attn over all tokens — store only aggregated per-modality sums per patient
    # Token slicing: [0:8]=gen, [8:8+n_path]=path, [8+n_path:8+n_path+16]=prot, [8+n_path+16:+16]=histo
    GT = 8; PT = tra_dim; RT = 16; HT = 16
    TOT = GT + PT + RT + HT
    pool_modality_sum = np.zeros((N, 4), dtype=np.float32)  # [gen, path, prot, histo]
    top_pathway_attn = np.zeros((N, PT), dtype=np.float32)  # keep pathway attn only
    histo_abmil = {}  # pid -> (n_patches,) attention weights

    fold_assign = np.zeros(N, dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(common)):
        log(f"Fold {fold+1}")
        tr_ids = [common[i] for i in tr_idx]; va_ids = [common[i] for i in va_idx]
        tr_ds = MultiDrugDataset4Modal(tr_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_ids, gen_df, tra_df, pro_df, ic50_df, DRUGS,
                                       histo_dir=HISTO_DIR, scalers=tr_ds.scalers)
        va_loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=2,
                               collate_fn=collate_4modal)
        model = PathOmicDRP(config).to(DEVICE)
        state = torch.load(os.path.join(FOLDS_DIR, f"fold{fold+1}_model.pt"),
                           map_location=DEVICE, weights_only=True)
        model.load_state_dict(state); model.eval()

        batch_start = 0
        with torch.no_grad():
            for b in va_loader:
                g = b['genomic'].to(DEVICE); t = b['transcriptomic'].to(DEVICE)
                p = b['proteomic'].to(DEVICE)
                kw = {}
                if 'histology' in b:
                    kw['histology'] = b['histology'].to(DEVICE)
                    kw['histo_mask'] = b['histo_mask'].to(DEVICE)
                pooled, aw, fused, histo_attn, pred = forward_with_hooks(model, g, t, p, **kw)
                B = pooled.shape[0]
                for bi in range(B):
                    gi = va_idx[batch_start + bi]
                    emb[gi] = pooled[bi].cpu().numpy()
                    aw_np = aw[bi].cpu().numpy()  # (TOT,)
                    # slice per modality
                    s = 0
                    pool_modality_sum[gi, 0] = aw_np[s:s+GT].sum(); s += GT
                    top_pathway_attn[gi] = aw_np[s:s+PT]
                    pool_modality_sum[gi, 1] = aw_np[s:s+PT].sum(); s += PT
                    pool_modality_sum[gi, 2] = aw_np[s:s+RT].sum(); s += RT
                    if aw_np.shape[0] >= s + HT:
                        pool_modality_sum[gi, 3] = aw_np[s:s+HT].sum()
                    fold_assign[gi] = fold + 1
                    if histo_attn is not None:
                        ha = histo_attn[bi].cpu().numpy()  # may be (H_tokens, N_patches) or (N_patches,)
                        if ha.ndim == 2:
                            ha = ha.mean(axis=0)
                        pid = common[gi]
                        # trim to actual patch count via histo_mask
                        m = kw['histo_mask'][bi].cpu().numpy()
                        histo_abmil[pid] = ha[:int(m.sum())]
                batch_start += B

    np.save(os.path.join(OUT, "oof_embeddings.npy"), emb)
    with open(os.path.join(OUT, "oof_embedding_pids.json"), 'w') as f:
        json.dump({'patient_ids': common, 'fold': fold_assign.tolist()}, f)
    np.savez_compressed(os.path.join(OUT, "attention_pool_modality.npz"),
                        pool_modality=pool_modality_sum, pids=np.array(common),
                        columns=np.array(['genomic', 'pathway', 'proteomic', 'histology']))
    np.savez_compressed(os.path.join(OUT, "attention_pathway_pool.npz"),
                        pathway_attn=top_pathway_attn, pids=np.array(common))
    # Histology attention — variable length, store as object
    np.savez_compressed(os.path.join(OUT, "attention_histology.npz"),
                        **{pid: histo_abmil[pid] for pid in histo_abmil})
    with open(os.path.join(OUT, "token_layout.json"), 'w') as f:
        json.dump({'n_gen_tokens': GT, 'n_pathway_tokens': PT,
                   'n_prot_tokens': RT, 'n_histo_tokens': HT,
                   'hidden_dim': HD,
                   'path_columns': [c for c in tra_df.columns if c != 'patient_id']}, f)
    log("Done")
    log(f"Mean pool attn share: gen={pool_modality_sum[:,0].mean():.3f} "
        f"path={pool_modality_sum[:,1].mean():.3f} prot={pool_modality_sum[:,2].mean():.3f} "
        f"histo={pool_modality_sum[:,3].mean():.3f}")


if __name__ == '__main__':
    main()
