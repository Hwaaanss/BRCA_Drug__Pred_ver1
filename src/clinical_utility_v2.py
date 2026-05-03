#!/usr/bin/env python3
"""Clinical utility v2 — uses PathOmicDRP 256-d embeddings + LogReg clinical head
(as in the original w1_clinical_validation pipeline) to produce per-patient
response probabilities, then runs decision-curve analysis + the ER+/HER2-
treatment recommendation scenario.

Outputs
  results/clinical_utility/decision_curves_v2.csv
  results/clinical_utility/decision_curves_v2.json
  results/clinical_utility/scenario_er_her2.csv         (from predicted IC50 ranking)
  results/clinical_utility/clinical_utility_summary_v2.json
"""
import os, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

OOF_DIR = PROJECT_ROOT / "results" / "oof"
BASE = DATA_ROOT / "07_integrated"
CLIN_DIR = DATA_ROOT / "01_clinical"
OUT = PROJECT_ROOT / "results" / "clinical_utility"
os.makedirs(OUT, exist_ok=True)

PANEL_DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]
CLINICAL_DRUGS = ['Doxorubicin', 'Cyclophosphamide', 'Paclitaxel', 'Docetaxel', 'Tamoxifen']


def log(m): print(f"[{time.strftime('%H:%M:%S')}] [CU2] {m}", flush=True)


def get_labels(tx, drug, pid_set):
    """Binary non-response label (target of prediction for decision support):
    y=1 means NON-responder (PD/SD); y=0 means responder (CR/PR).
    This matches the clinical framing: decision rule triggers when predicted
    non-response probability crosses a threshold (patient gets alternative tx)."""
    lbls = {}
    treated = tx[tx['therapeutic_agents'].str.contains(drug, case=False, na=False)]
    for _, r in treated.iterrows():
        pid = r['submitter_id']
        if pid not in pid_set: continue
        o = r['treatment_outcome']
        if o in ('Complete Response', 'Partial Response'):
            lbls[pid] = 0
        elif o in ('Progressive Disease', 'Stable Disease'):
            lbls[pid] = 1
    return lbls


def dca(y, proba, thresholds):
    n = len(y); prev = y.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        pred = (proba >= pt)
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb = tp / n - fp / n * (pt / (1 - pt + 1e-9))
        nb_model.append(nb)
        nb_all.append(prev - (1 - prev) * (pt / (1 - pt + 1e-9)))
    return np.array(nb_model), np.array(nb_all)


def main():
    # Load embeddings
    emb = np.load(os.path.join(OOF_DIR, 'oof_embeddings.npy'))
    meta = json.load(open(os.path.join(OOF_DIR, 'oof_embedding_pids.json')))
    pids = meta['patient_ids']
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    log(f"Embeddings: {emb.shape}")

    tx = pd.read_csv(os.path.join(CLIN_DIR, 'TCGA_BRCA_drug_treatments.csv'))
    oof = pd.read_csv(os.path.join(OOF_DIR, 'oof_predictions.csv')).set_index('patient_id')
    trans = pd.read_csv(os.path.join(BASE, 'X_transcriptomic.csv')).set_index('patient_id')

    thresholds = np.linspace(0.05, 0.6, 56)
    dca_rows = []; dca_summary = {}

    for drug in CLINICAL_DRUGS:
        lbls = get_labels(tx, drug, set(pids))
        if len(lbls) < 15 or sum(lbls.values()) < 3 or (len(lbls) - sum(lbls.values())) < 3:
            log(f"  skip {drug}: n={len(lbls)} pos={sum(lbls.values())}")
            continue
        lbl_pids = sorted(lbls.keys())
        X = np.stack([emb[pid_to_idx[p]] for p in lbl_pids])
        y = np.array([lbls[p] for p in lbl_pids], dtype=int)
        # Predict RESPONSE (CR/PR) = 1; non-response (PD/SD) = 0.
        y_nr = y
        n_pos = int(y_nr.sum()); n_neg = len(y_nr) - n_pos
        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            log(f"  skip {drug}: n_splits<2 ({n_pos}/{n_neg})")
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        proba_oof = np.zeros(len(y_nr))
        for tr, te in skf.split(X, y_nr):
            sc = StandardScaler()
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1)
            clf.fit(sc.fit_transform(X[tr]), y_nr[tr])
            proba_oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

        auc = roc_auc_score(y_nr, proba_oof)
        # bootstrap CI
        rng = np.random.RandomState(42)
        aucs_bs = []
        for _ in range(1000):
            idx = rng.randint(0, len(y_nr), len(y_nr))
            try:
                if len(np.unique(y_nr[idx])) == 2:
                    aucs_bs.append(roc_auc_score(y_nr[idx], proba_oof[idx]))
            except Exception: pass
        ci = (float(np.percentile(aucs_bs, 2.5)), float(np.percentile(aucs_bs, 97.5))) if aucs_bs else (None, None)

        nb_m, nb_all = dca(y_nr, proba_oof, thresholds)
        for pt, nm, na in zip(thresholds, nb_m, nb_all):
            dca_rows.append({'drug': drug, 'threshold': float(pt),
                             'nb_model': float(nm), 'nb_treat_all': float(na)})
        dca_summary[drug] = {
            'n_total': int(len(y_nr)),
            'n_nonresponder': int(n_pos),
            'n_responder': int(n_neg),
            'auc_nonresponse': float(auc),
            'auc_bootstrap_ci95': list(ci),
            'auc_dca_integrated_gain': float(np.trapezoid(np.maximum(nb_m - nb_all, 0), thresholds)),
            'dominant_range': [float(thresholds[(nb_m > nb_all) & (nb_m > 0)].min())
                               if np.any((nb_m > nb_all) & (nb_m > 0)) else None,
                               float(thresholds[(nb_m > nb_all) & (nb_m > 0)].max())
                               if np.any((nb_m > nb_all) & (nb_m > 0)) else None],
        }
        log(f"  {drug:18s}: n={len(y_nr)} NR={n_pos} AUC={auc:.3f} CI=[{ci[0]:.2f},{ci[1]:.2f}] "
            f"ΔAUC_DCA={dca_summary[drug]['auc_dca_integrated_gain']:.4f}")

    pd.DataFrame(dca_rows).to_csv(os.path.join(OUT, 'decision_curves_v2.csv'), index=False)
    with open(os.path.join(OUT, 'decision_curves_v2.json'), 'w') as f:
        json.dump({'thresholds': thresholds.tolist(), 'summary': dca_summary}, f, indent=2)

    # ER+/HER2- scenario on IC50 ranking (per-drug z-scored so recommendation reflects
    # within-cohort RELATIVE sensitivity rather than raw IC50 scale differences).
    common = oof.index.intersection(trans.index)
    esr1 = np.log1p(trans.loc[common, 'ESR1'].astype(float).values)
    erbb2 = np.log1p(trans.loc[common, 'ERBB2'].astype(float).values)
    subset = common[(esr1 >= np.median(esr1)) & (erbb2 < np.percentile(erbb2, 75))]
    pred_mat = np.stack([oof.loc[subset, f"pred_{d}"].values for d in PANEL_DRUGS], axis=1)  # (N, 13)
    pred_z = (pred_mat - pred_mat.mean(axis=0)) / (pred_mat.std(axis=0) + 1e-9)
    fulv_idx = PANEL_DRUGS.index('Fulvestrant_1816')
    fulv_z = pred_z[:, fulv_idx]
    nr_mask = fulv_z >= np.percentile(fulv_z, 75)
    nr_pids = subset[nr_mask].tolist()
    log(f"ER+/HER2- subset={len(subset)}, predicted-NR={len(nr_pids)}")

    rows = []
    for i, pid in enumerate(nr_pids):
        preds = {d: float(oof.loc[pid, f"pred_{d}"]) for d in PANEL_DRUGS}
        # z-scored within cohort: use subset indexing
        si = list(subset).index(pid)
        alt_z = {d: float(pred_z[si, j]) for j, d in enumerate(PANEL_DRUGS) if d != 'Fulvestrant_1816'}
        alt = min(alt_z, key=alt_z.get)
        actual = tx[tx['submitter_id'] == pid]
        rows.append({
            'patient_id': pid,
            'ESR1_log1p': float(np.log1p(trans.loc[pid, 'ESR1'])),
            'ERBB2_log1p': float(np.log1p(trans.loc[pid, 'ERBB2'])),
            'pred_Fulvestrant_IC50': preds['Fulvestrant_1816'],
            'fulvestrant_z_within_cohort': float(pred_z[list(subset).index(pid), fulv_idx]),
            'recommended_alternative': alt,
            'alt_pred_IC50': preds[alt],
            'alt_z_within_cohort': alt_z[alt],
            'delta_fulv_minus_alt_IC50': preds['Fulvestrant_1816'] - preds[alt],
            'actual_treatments': ';'.join(actual['therapeutic_agents'].dropna().astype(str).tolist()) or 'NONE',
            'actual_outcomes': ';'.join(actual['treatment_outcome'].dropna().astype(str).tolist()) or 'NONE',
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'scenario_er_her2.csv'), index=False)
    alt_counts = pd.Series([r['recommended_alternative'] for r in rows]).value_counts().to_dict()

    # Concordance: among predicted-NR patients who actually received fulvestrant/tamoxifen (ER-targeted),
    # how many had bad outcomes? compared to responders.
    targ_outcomes = {'good': 0, 'bad': 0, 'any': 0}
    for r in rows:
        t = r['actual_treatments'].lower(); o = r['actual_outcomes'].lower()
        if 'tamoxifen' in t or 'fulvestrant' in t or 'letrozole' in t or 'anastrozole' in t:
            targ_outcomes['any'] += 1
            if 'progressive' in o or 'stable' in o:
                targ_outcomes['bad'] += 1
            elif 'complete' in o or 'partial' in o:
                targ_outcomes['good'] += 1

    summary = {
        'decision_curves': dca_summary,
        'scenario': {
            'n_subset_er_her2': int(len(subset)),
            'n_predicted_nonresponders': int(len(nr_pids)),
            'recommendation_distribution': alt_counts,
            'among_predicted_NR_on_hormone_therapy': targ_outcomes,
        },
    }
    with open(os.path.join(OUT, 'clinical_utility_summary_v2.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Scenario: {summary['scenario']}")


if __name__ == '__main__':
    main()
