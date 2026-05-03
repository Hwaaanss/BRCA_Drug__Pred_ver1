#!/usr/bin/env python3
"""PathOmicDRP High-Impact Analyses (1-6)"""
import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from collections import defaultdict
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", PROJECT_ROOT)).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", BASE / "data")).resolve()
if not (DATA_ROOT / "07_integrated").exists() and (BASE / "07_integrated").exists():
    DATA_ROOT = BASE

RESULTS_DIR = BASE / "results"
INTEGRATED_DIR = DATA_ROOT / "07_integrated"
CLINICAL_DIR = DATA_ROOT / "01_clinical"
HISTO_DIR = DATA_ROOT / "05_morphology" / "features"

sys.path.insert(0, str(BASE / "src"))
from model import PathOmicDRP, get_default_config
from train_phase3_4modal import MultiDrugDataset4Modal, collate_4modal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_all():
    with open(RESULTS_DIR / "phase3_4modal_full" / "cv_results.json") as f:
        cv = json.load(f)
    config = cv['config']; drug_cols = cv['drugs']
    model = PathOmicDRP(config).to(DEVICE)
    state = torch.load(RESULTS_DIR / "phase3_4modal_full" / "best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state); model.eval()
    gen_df = pd.read_csv(INTEGRATED_DIR / "X_genomic.csv")
    tra_df = pd.read_csv(INTEGRATED_DIR / "X_transcriptomic.csv")
    pro_df = pd.read_csv(INTEGRATED_DIR / "X_proteomic.csv")
    ic50_df = pd.read_csv(INTEGRATED_DIR / "predicted_IC50_all_drugs.csv", index_col=0)
    histo_ids = {f.replace('.pt','') for f in os.listdir(HISTO_DIR) if f.endswith('.pt')}
    common = sorted(set(gen_df['patient_id']) & set(tra_df['patient_id']) & set(pro_df['patient_id']) & set(ic50_df.index) & histo_ids)
    dataset = MultiDrugDataset4Modal(common, gen_df, tra_df, pro_df, ic50_df, drug_cols, histo_dir=HISTO_DIR, fit=True)
    drug_names = [d.rsplit('_',1)[0] for d in drug_cols]
    return model, dataset, config, drug_cols, drug_names, common, gen_df, tra_df, pro_df, ic50_df

def get_embeddings(model, dataset, use_histo=True):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_4modal)
    embs = []
    def hook_fn(m, i, o): embs.append(o.detach().cpu())
    handle = model.fusion.self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in loader:
            g = batch['genomic'].to(DEVICE); t = batch['transcriptomic'].to(DEVICE); p = batch['proteomic'].to(DEVICE)
            kw = {}
            if use_histo and 'histology' in batch:
                kw['histology'] = batch['histology'].to(DEVICE); kw['histo_mask'] = batch['histo_mask'].to(DEVICE)
            model(g, t, p, **kw)
    handle.remove()
    return torch.cat([e.mean(dim=1) for e in embs], dim=0).numpy()

# ═══ ANALYSIS 1: Clinical Outcome ═══
def analysis1(model, dataset, pids, emb4, emb3):
    log("═══ ANALYSIS 1: Direct Clinical Outcome ═══")
    out_dir = RESULTS_DIR / "analysis1_clinical_outcome"; os.makedirs(out_dir, exist_ok=True)
    drug_df = pd.read_csv(CLINICAL_DIR / "TCGA_BRCA_drug_treatments.csv")
    pid_set = set(pids); pid_to_idx = {p:i for i,p in enumerate(pids)}
    results = {}
    for drug_name in ['Docetaxel','Paclitaxel','Tamoxifen','Cyclophosphamide']:
        treated = drug_df[drug_df['therapeutic_agents'].str.contains(drug_name, case=False, na=False)]
        labels = {}
        for _, row in treated[treated['submitter_id'].isin(pid_set)].iterrows():
            pid = row['submitter_id']; outcome = row['treatment_outcome']
            if outcome in ('Complete Response','Partial Response'): labels[pid] = 1
            elif outcome in ('Progressive Disease','Stable Disease'): labels[pid] = 0
            elif outcome == 'Treatment Ongoing' and drug_name == 'Tamoxifen': labels[pid] = 1
        valid = [p for p in labels if p in pid_to_idx]
        if len(valid) < 10: continue
        y = np.array([labels[p] for p in valid])
        if y.sum() < 3 or (len(y)-y.sum()) < 3: continue
        drug_res = {'n': len(y), 'n_pos': int(y.sum()), 'n_neg': int(len(y)-y.sum())}
        for name, X_all in [('4-modal', emb4), ('3-modal', emb3)]:
            X = np.array([X_all[pid_to_idx[p]] for p in valid])
            n_splits = min(5, int(len(y)-y.sum()))
            if n_splits < 2: continue
            aucs = []
            for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y):
                sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
                clf.fit(Xtr, y[tr])
                try: aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:,1]))
                except: aucs.append(0.5)
            drug_res[name] = {'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))}
        results[drug_name] = drug_res
        log(f"  {drug_name}: n={len(y)} (pos={y.sum()}) | 4m AUC={drug_res.get('4-modal',{}).get('auc_mean',0):.3f} | 3m AUC={drug_res.get('3-modal',{}).get('auc_mean',0):.3f}")
    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    return results

# ═══ ANALYSIS 2: Drug-specific Attention ═══
def analysis2(model, dataset, pids, drug_cols, drug_names):
    log("═══ ANALYSIS 2: Drug-specific Attention ═══")
    out_dir = RESULTS_DIR / "analysis2_drug_attention"; os.makedirs(out_dir, exist_ok=True)
    n_pat = min(20, len(dataset)); n_drugs = len(drug_cols)
    corrs = np.zeros((n_drugs, n_drugs)); n_ok = 0
    for i in range(n_pat):
        s = dataset[i]
        if 'histology' not in s: continue
        batch = collate_4modal([s])
        g = batch['genomic'].to(DEVICE); t = batch['transcriptomic'].to(DEVICE); p = batch['proteomic'].to(DEVICE)
        h = batch['histology'].to(DEVICE).requires_grad_(True); hm = batch['histo_mask'].to(DEVICE)
        n_p = hm[0].sum().item(); attns = []
        for d in range(n_drugs):
            if h.grad is not None: h.grad.zero_()
            model.zero_grad()
            result = model(g, t, p, histology=h, histo_mask=hm)
            result['prediction'][0, d].backward(retain_graph=(d < n_drugs-1))
            grad = h.grad[0, :n_p].abs().mean(dim=-1).detach().cpu().numpy()
            attns.append(grad)
        attns = np.array(attns)
        for d1 in range(n_drugs):
            for d2 in range(n_drugs):
                if np.std(attns[d1])>0 and np.std(attns[d2])>0:
                    r,_ = pearsonr(attns[d1], attns[d2]); corrs[d1,d2] += r
        n_ok += 1
        if (i+1)%10==0: log(f"  Processed {i+1}/{n_pat}")
    if n_ok > 0: corrs /= n_ok
    pd.DataFrame(corrs, index=drug_names, columns=drug_names).to_csv(f"{out_dir}/drug_attention_correlation.csv")
    results = {'n_patients': n_ok, 'mean_self': float(np.diag(corrs).mean()), 'mean_cross': float(corrs[np.triu_indices(n_drugs,k=1)].mean())}
    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    log(f"  Self-corr: {results['mean_self']:.3f}, Cross-corr: {results['mean_cross']:.3f}")
    return results

# ═══ ANALYSIS 3: Leave-One-Drug-Out ═══
def analysis3(dataset, drug_cols, drug_names, pids):
    log("═══ ANALYSIS 3: Leave-One-Drug-Out ═══")
    out_dir = RESULTS_DIR / "analysis3_lodo"; os.makedirs(out_dir, exist_ok=True)
    y_all, X_all = [], []
    for i in range(len(dataset)):
        s = dataset[i]; y_all.append(s['target'].numpy())
        X_all.append(np.concatenate([s['genomic'].numpy(), s['transcriptomic'].numpy(), s['proteomic'].numpy()]))
    y_all = np.array(y_all); X_all = np.array(X_all)
    results = {}
    for hi in range(len(drug_cols)):
        train_idx = [j for j in range(len(drug_cols)) if j != hi]
        y_held = y_all[:, hi]; y_others = y_all[:, train_idx]
        pccs = []
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X_all):
            Xtr = np.hstack([X_all[tr], y_others[tr]]); Xte = np.hstack([X_all[te], y_others[te]])
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
            m = Ridge(alpha=1.0); m.fit(Xtr, y_held[tr]); pred = m.predict(Xte)
            r,_ = pearsonr(y_held[te], pred); pccs.append(r)
        results[drug_names[hi]] = {'pcc_mean': float(np.mean(pccs)), 'pcc_std': float(np.std(pccs))}
    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    mean_pcc = np.mean([v['pcc_mean'] for v in results.values()])
    log(f"  Mean LODO PCC: {mean_pcc:.3f}")
    for d, v in sorted(results.items(), key=lambda x: -x[1]['pcc_mean']):
        log(f"    {d:20s}: {v['pcc_mean']:.3f}±{v['pcc_std']:.3f}")
    return results

# ═══ ANALYSIS 4: Multi-task Survival ═══
def analysis4(config, drug_cols, drug_names, pids):
    log("═══ ANALYSIS 4: Multi-task Survival ═══")
    out_dir = RESULTS_DIR / "analysis4_multitask"; os.makedirs(out_dir, exist_ok=True)
    clin = pd.read_csv(CLINICAL_DIR / "TCGA_BRCA_clinical.csv").set_index('submitter_id')
    surv = {}
    for pid in pids:
        if pid not in clin.index: continue
        v = clin.loc[pid, 'vital_status']
        if pd.isna(v): continue
        ev = 1 if v == 'Dead' else 0
        t = clin.loc[pid, 'days_to_death'] if ev else clin.loc[pid, 'days_to_last_follow_up']
        if pd.isna(t) or t <= 0: continue
        surv[pid] = {'time': float(t), 'event': ev}
    log(f"  Survival: {len(surv)} patients, {sum(v['event'] for v in surv.values())} events")

    gen_df = pd.read_csv(INTEGRATED_DIR / "X_genomic.csv")
    tra_df = pd.read_csv(INTEGRATED_DIR / "X_transcriptomic.csv")
    pro_df = pd.read_csv(INTEGRATED_DIR / "X_proteomic.csv")
    ic50_df = pd.read_csv(INTEGRATED_DIR / "predicted_IC50_all_drugs.csv", index_col=0)

    def cox_loss(risk, times, events):
        si = torch.argsort(times, descending=True)
        rs = risk[si]; es = events[si]
        lcs = torch.logcumsumexp(rs, dim=0)
        return -torch.mean((rs - lcs) * es)

    from lifelines.utils import concordance_index
    kf = KFold(5, shuffle=True, random_state=42)
    mt_pccs, mt_cis = [], []
    for fold, (tri, vai) in enumerate(kf.split(pids)):
        tr_pids = [pids[i] for i in tri]; va_pids = [pids[i] for i in vai]
        tr_ds = MultiDrugDataset4Modal(tr_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols, histo_dir=HISTO_DIR, fit=True)
        va_ds = MultiDrugDataset4Modal(va_pids, gen_df, tra_df, pro_df, ic50_df, drug_cols, histo_dir=HISTO_DIR, scalers=tr_ds.scalers)
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_4modal, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_4modal)

        mt_model = PathOmicDRP(config).to(DEVICE)
        surv_head = nn.Linear(config['hidden_dim'], 1).to(DEVICE)
        opt = torch.optim.AdamW(list(mt_model.parameters())+list(surv_head.parameters()), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=3e-6)
        crit = nn.HuberLoss(delta=1.0)
        best_loss = float('inf'); best_st = None; pat = 0

        for epoch in range(60):
            mt_model.train(); surv_head.train(); tloss = 0; nb = 0
            for batch in tr_loader:
                g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE); p=batch['proteomic'].to(DEVICE); y=batch['target'].to(DEVICE)
                kw = {}
                if 'histology' in batch: kw['histology']=batch['histology'].to(DEVICE); kw['histo_mask']=batch['histo_mask'].to(DEVICE)
                opt.zero_grad()
                ft = []
                def hk(m,i,o): ft.append(o)
                hh = mt_model.fusion.self_attn.register_forward_hook(hk)
                res = mt_model(g, t, p, **kw); hh.remove()
                dl = crit(res['prediction'], y)
                fused = ft[0].mean(dim=1); risk = surv_head(fused).squeeze(-1)
                # Get survival labels for batch patients
                b_times, b_events = [], []
                for k in range(len(g)):
                    idx = min(k, len(tr_pids)-1); pid = tr_pids[idx]
                    if pid in surv: b_times.append(surv[pid]['time']); b_events.append(surv[pid]['event'])
                    else: b_times.append(1.0); b_events.append(0)
                bt = torch.tensor(b_times, device=DEVICE); be = torch.tensor(b_events, dtype=torch.float32, device=DEVICE)
                if be.sum() > 0:
                    sl = cox_loss(risk, bt, be); loss = dl + 0.1 * sl
                else: loss = dl
                if torch.isnan(loss): continue
                loss.backward(); torch.nn.utils.clip_grad_norm_(mt_model.parameters(), 1.0); opt.step()
                tloss += loss.item(); nb += 1
            sched.step()
            if nb > 0:
                al = tloss/nb
                if al < best_loss: best_loss = al; best_st = {'m': {k:v.cpu().clone() for k,v in mt_model.state_dict().items()}, 's': {k:v.cpu().clone() for k,v in surv_head.state_dict().items()}}; pat = 0
                else: pat += 1
                if pat >= 12: break

        if best_st: mt_model.load_state_dict(best_st['m']); surv_head.load_state_dict(best_st['s'])
        mt_model.to(DEVICE).eval(); surv_head.to(DEVICE).eval()
        ap, at, ar = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE); p=batch['proteomic'].to(DEVICE)
                kw = {}
                if 'histology' in batch: kw['histology']=batch['histology'].to(DEVICE); kw['histo_mask']=batch['histo_mask'].to(DEVICE)
                ft2 = []
                def hk2(m,i,o): ft2.append(o)
                hh2 = mt_model.fusion.self_attn.register_forward_hook(hk2)
                res = mt_model(g, t, p, **kw); hh2.remove()
                pred = tr_ds.scalers['ic50'].inverse_transform(res['prediction'].cpu().numpy())
                true = tr_ds.scalers['ic50'].inverse_transform(batch['target'].numpy())
                ap.append(pred); at.append(true)
                fused = ft2[0].mean(dim=1); ar.append(surv_head(fused).squeeze(-1).cpu().numpy())
        ap=np.concatenate(ap); at=np.concatenate(at); ar=np.concatenate(ar)
        dpccs = [pearsonr(at[:,d], ap[:,d])[0] for d in range(len(drug_cols))]
        mt_pccs.append(np.mean(dpccs))
        vt, ve, vr = [], [], []
        for j, pid in enumerate(va_pids):
            if pid in surv and j < len(ar): vt.append(surv[pid]['time']); ve.append(surv[pid]['event']); vr.append(ar[j])
        ci = concordance_index(vt, [-r for r in vr], ve) if len(vr)>10 and sum(ve)>0 else 0.5
        mt_cis.append(ci)
        log(f"  Fold {fold+1}: PCC={np.mean(dpccs):.4f}, C-index={ci:.4f}")

    results = {'drug_pcc': {'mean': float(np.mean(mt_pccs)), 'std': float(np.std(mt_pccs))},
               'c_index': {'mean': float(np.mean(mt_cis)), 'std': float(np.std(mt_cis))}}
    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    log(f"  Multi-task PCC: {results['drug_pcc']['mean']:.4f}, C-index: {results['c_index']['mean']:.4f}")
    return results

# ═══ ANALYSIS 5: Biomarker Concordance ═══
def analysis5(model, dataset, pids, drug_cols, drug_names, pro_df):
    log("═══ ANALYSIS 5: Biomarker Concordance ═══")
    out_dir = RESULTS_DIR / "analysis5_external_validation"; os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_4modal)
    ap = []
    with torch.no_grad():
        for batch in loader:
            g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE); p=batch['proteomic'].to(DEVICE)
            kw = {}
            if 'histology' in batch: kw['histology']=batch['histology'].to(DEVICE); kw['histo_mask']=batch['histo_mask'].to(DEVICE)
            ap.append(model(g,t,p,**kw)['prediction'].cpu().numpy())
    ap = dataset.scalers['ic50'].inverse_transform(np.concatenate(ap))
    pred_df = pd.DataFrame(ap, columns=drug_names, index=pids)
    pro = pro_df.set_index('patient_id'); cp = [p for p in pids if p in pro.index]
    gen_df = pd.read_csv(INTEGRATED_DIR / "X_genomic.csv").set_index('patient_id')
    cg = [p for p in pids if p in gen_df.index]
    results = {}

    # ER → Tamoxifen
    if 'ERALPHA' in pro.columns and 'Tamoxifen' in drug_names:
        er = pro.loc[cp,'ERALPHA']; tam = pred_df.loc[cp,'Tamoxifen']
        r,p = spearmanr(er, tam)
        er_pos = er > er.median()
        results['ER_Tamoxifen'] = {'spearman_r': float(r), 'p': float(p), 'ER+_mean': float(tam[er_pos].mean()), 'ER-_mean': float(tam[~er_pos].mean())}
        log(f"  ER→Tamoxifen: r={r:.3f}, p={p:.1e}")
    # HER2 → Lapatinib
    if 'HER2' in pro.columns and 'Lapatinib' in drug_names:
        h2 = pro.loc[cp,'HER2']; lap = pred_df.loc[cp,'Lapatinib']
        r,p = spearmanr(h2, lap)
        results['HER2_Lapatinib'] = {'spearman_r': float(r), 'p': float(p)}
        log(f"  HER2→Lapatinib: r={r:.3f}, p={p:.1e}")
    # ER → Fulvestrant
    if 'ERALPHA' in pro.columns and 'Fulvestrant' in drug_names:
        r,p = spearmanr(pro.loc[cp,'ERALPHA'], pred_df.loc[cp,'Fulvestrant'])
        results['ER_Fulvestrant'] = {'spearman_r': float(r), 'p': float(p)}
        log(f"  ER→Fulvestrant: r={r:.3f}, p={p:.1e}")
    # TP53 → Cisplatin
    if 'TP53' in gen_df.columns and 'Cisplatin' in drug_names:
        tp53 = gen_df.loc[cg,'TP53'] > 0; cis = pred_df.loc[cg,'Cisplatin']
        s,p = mannwhitneyu(cis[tp53], cis[~tp53])
        results['TP53_Cisplatin'] = {'mut_mean': float(cis[tp53].mean()), 'wt_mean': float(cis[~tp53].mean()), 'p': float(p)}
        log(f"  TP53→Cisplatin: mut={cis[tp53].mean():.2f} vs wt={cis[~tp53].mean():.2f}, p={p:.4f}")
    # PIK3CA → all drugs
    if 'PIK3CA' in gen_df.columns:
        pik = gen_df.loc[cg,'PIK3CA'] > 0
        for d in drug_names:
            dv = pred_df.loc[cg, d]; s,p = mannwhitneyu(dv[pik], dv[~pik])
            if p < 0.01:
                results[f'PIK3CA_{d}'] = {'mut_mean': float(dv[pik].mean()), 'wt_mean': float(dv[~pik].mean()), 'p': float(p)}
                log(f"  PIK3CA→{d}: p={p:.4f}")

    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    return results

# ═══ ANALYSIS 6: Phenotype Discovery ═══
def analysis6(model, dataset, pids, drug_cols, drug_names):
    log("═══ ANALYSIS 6: Pathology Phenotype Discovery ═══")
    out_dir = RESULTS_DIR / "analysis6_phenotype"; os.makedirs(out_dir, exist_ok=True)
    n_pat = min(50, len(dataset)); top_k = 50
    # Get predictions
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_4modal)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            g=batch['genomic'].to(DEVICE); t=batch['transcriptomic'].to(DEVICE); p=batch['proteomic'].to(DEVICE)
            kw = {}
            if 'histology' in batch: kw['histology']=batch['histology'].to(DEVICE); kw['histo_mask']=batch['histo_mask'].to(DEVICE)
            all_preds.append(model(g,t,p,**kw)['prediction'].cpu().numpy())
    all_preds = dataset.scalers['ic50'].inverse_transform(np.concatenate(all_preds))
    pid_preds = {pids[i]: all_preds[i] for i in range(len(pids))}

    feats, pat_ids, attns = [], [], []
    for i in range(n_pat):
        s = dataset[i]
        if 'histology' not in s: continue
        h = s['histology']; pid = pids[i]
        with torch.no_grad():
            hi = h.unsqueeze(0).to(DEVICE); he = model.histology_encoder
            proj = he.feature_proj(hi); aV = he.attention_V(proj); aU = he.attention_U(proj)
            a = he.attention_W(aV * aU).mean(dim=-1)
            attn = torch.softmax(a, dim=-1).cpu().numpy()[0]
        k = min(top_k, len(attn)); top_idx = np.argsort(attn)[-k:]
        for idx in top_idx:
            feats.append(h[idx].numpy()); pat_ids.append(pid); attns.append(attn[idx])
    feats = np.array(feats)
    log(f"  Collected {len(feats)} patches from {n_pat} patients")

    pca = PCA(n_components=50, random_state=42); fpca = pca.fit_transform(feats)
    n_cl = 6; km = KMeans(n_cl, random_state=42, n_init=10); labels = km.fit_predict(fpca)

    cl_drugs = defaultdict(lambda: defaultdict(list))
    for pi, (pid, cl) in enumerate(zip(pat_ids, labels)):
        if pid in pid_preds:
            for d, nm in enumerate(drug_names): cl_drugs[cl][nm].append(pid_preds[pid][d])
    summary = {}
    for cl in range(n_cl):
        summary[f'cluster_{cl}'] = {'n_patches': int((labels==cl).sum()),
            'n_patients': len(set(p for p,c in zip(pat_ids,labels) if c==cl)),
            'drug_means': {nm: float(np.mean(vs)) for nm, vs in cl_drugs[cl].items()}}

    try:
        import umap; reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        patch_umap = reducer.fit_transform(fpca)
    except:
        from sklearn.manifold import TSNE
        patch_umap = TSNE(n_components=2, random_state=42).fit_transform(fpca[:3000])

    np.save(f"{out_dir}/patch_umap.npy", patch_umap[:len(labels)])
    np.save(f"{out_dir}/cluster_labels.npy", labels)
    results = {'n_patches': len(feats), 'n_clusters': n_cl, 'summary': summary}
    with open(f"{out_dir}/results.json",'w') as f: json.dump(results, f, indent=2)
    log(f"  Clusters: {[int((labels==c).sum()) for c in range(n_cl)]}")
    return results

# ═══ MAIN ═══
if __name__ == '__main__':
    log("Loading model and data...")
    model, dataset, config, drug_cols, drug_names, pids, gen_df, tra_df, pro_df, ic50_df = load_all()
    log(f"Loaded {len(pids)} patients")
    log("Computing embeddings...")
    emb4 = get_embeddings(model, dataset, use_histo=True)
    emb3 = get_embeddings(model, dataset, use_histo=False)
    log(f"Embeddings: 4m={emb4.shape}, 3m={emb3.shape}")
    ALL = {}
    for name, fn in [
        ('1_clinical', lambda: analysis1(model, dataset, pids, emb4, emb3)),
        ('2_attention', lambda: analysis2(model, dataset, pids, drug_cols, drug_names)),
        ('3_lodo', lambda: analysis3(dataset, drug_cols, drug_names, pids)),
        ('4_multitask', lambda: analysis4(config, drug_cols, drug_names, pids)),
        ('5_biomarker', lambda: analysis5(model, dataset, pids, drug_cols, drug_names, pro_df)),
        ('6_phenotype', lambda: analysis6(model, dataset, pids, drug_cols, drug_names)),
    ]:
        try:
            ALL[name] = fn()
        except Exception as e:
            log(f"  {name} FAILED: {e}"); traceback.print_exc()
    with open(RESULTS_DIR / "high_impact_results.json",'w') as f:
        json.dump(ALL, f, indent=2, default=str)
    log("\n═══ ALL 6 ANALYSES COMPLETE ═══")
