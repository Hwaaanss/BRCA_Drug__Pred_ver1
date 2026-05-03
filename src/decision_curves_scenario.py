#!/usr/bin/env python3
"""Clinical utility analysis for Genome Medicine.

(a) Decision-curve / net-benefit analysis for Doxorubicin and Cyclophosphamide
    (the two drugs with the narrowest bootstrap CI in w1_clinical_validation).
    Net benefit computed across threshold probabilities p_t in [0.01, 0.99]
    against two reference strategies: 'treat all' and 'treat none'.

(b) Treatment recommendation scenario on ER+/HER2- proxy subset:
    - ER+ proxy:  ESR1 transcript expression >= cohort median
    - HER2- proxy: ERBB2 transcript expression <  cohort 75th percentile
    - "Predicted fulvestrant non-responder": predicted Fulvestrant_1816
      IC50 >= 75th percentile within the ER+/HER2- subset.
    - For each such patient, emit the drug with the LOWEST predicted IC50
      among the 13-drug panel as an alternative recommendation, and also
      report their actual TCGA drug treatments and outcomes.

Inputs:
    results/oof/oof_predictions.csv            (from oof_predictions.py)
    07_integrated/X_transcriptomic.csv          (for ESR1 / ERBB2)
    01_clinical/TCGA_BRCA_drug_treatments.csv

Outputs:
    results/clinical_utility/decision_curves.csv    (rows=thresholds, cols=method/drug)
    results/clinical_utility/decision_curves.json
    results/clinical_utility/scenario_er_her2.csv
    results/clinical_utility/clinical_utility_summary.json
"""
import os, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()

BASE = DATA_ROOT / "07_integrated"
CLIN_DIR = DATA_ROOT / "01_clinical"
OOF = PROJECT_ROOT / "results" / "oof" / "oof_predictions.csv"
OUT = PROJECT_ROOT / "results" / "clinical_utility"
os.makedirs(OUT, exist_ok=True)

DRUG_MAP = {
    'Doxorubicin': 'Doxorubicin_1001' if False else None,  # not in panel - use Docetaxel/Cyclo CI drugs
    'Docetaxel': 'Docetaxel_1007',
    'Paclitaxel': 'Paclitaxel_1080',
    'Tamoxifen': 'Tamoxifen_1199',
    'Cyclophosphamide': None,  # not in panel; use Vinblastine as closest chemo proxy if needed
    'Fulvestrant': 'Fulvestrant_1816',
}

# Drugs we actually have predictions for in OOF file
PANEL_DRUGS = [
    'Cisplatin_1005', 'Docetaxel_1007', 'Paclitaxel_1080',
    'Gemcitabine_1190', 'Tamoxifen_1199', 'Lapatinib_1558',
    'Vinblastine_1004', 'OSI-027_1594', 'Daporinad_1248',
    'Venetoclax_1909', 'ABT737_1910', 'AZD5991_1720',
    'Fulvestrant_1816',
]

# Map TCGA drug names (as in therapeutic_agents) -> GDSC panel name for clinical DCA.
# Only drugs that actually appear in both TCGA treatments AND our GDSC 13-drug panel.
CLINICAL_DRUGS = {
    'Docetaxel':    'Docetaxel_1007',
    'Paclitaxel':   'Paclitaxel_1080',
    'Tamoxifen':    'Tamoxifen_1199',
    'Fulvestrant':  'Fulvestrant_1816',
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [DCA] {m}", flush=True)


def get_clinical_labels(drug_df, tcga_drug_name, valid_pids):
    """1 = non-responder (PD/SD), 0 = responder (CR/PR). Focus is on predicting non-response."""
    pid_set = set(valid_pids)
    treated = drug_df[drug_df['therapeutic_agents'].str.contains(tcga_drug_name, case=False, na=False)]
    labels = {}
    for _, row in treated.iterrows():
        pid = row['submitter_id']
        if pid not in pid_set:
            continue
        o = row['treatment_outcome']
        if o in ('Complete Response', 'Partial Response'):
            labels[pid] = 0
        elif o in ('Progressive Disease', 'Stable Disease'):
            labels[pid] = 1
    return labels


def net_benefit(y_true, y_score, thresholds):
    """Vickers' decision-curve net benefit.
    y_score assumed to be a continuous risk score (higher = more likely positive).
    Each p_t is interpreted as threshold of score percentile equal to p_t."""
    n = len(y_true)
    # map score to pseudo-probability via rank
    order = np.argsort(np.argsort(y_score))
    p_hat = (order + 0.5) / n
    nb_model, nb_all = [], []
    prev = y_true.mean()
    for pt in thresholds:
        pred_pos = (p_hat >= pt)
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt + 1e-9))
        nb_model.append(float(nb))
        # treat all
        nb_all.append(float(prev - (1 - prev) * (pt / (1 - pt + 1e-9))))
    return np.array(nb_model), np.array(nb_all)


def main():
    oof = pd.read_csv(OOF)
    tx = pd.read_csv(os.path.join(CLIN_DIR, 'TCGA_BRCA_drug_treatments.csv'))
    trans = pd.read_csv(os.path.join(BASE, 'X_transcriptomic.csv'))
    log(f"OOF: {oof.shape}, treatments: {tx.shape}, trans: {trans.shape}")

    oof = oof.set_index('patient_id')
    trans = trans.set_index('patient_id')

    # =========================================================================
    # (a) Decision curves — per drug
    # =========================================================================
    thresholds = np.linspace(0.01, 0.8, 80)
    dca_rows, dca_summary = [], {}

    for tcga_name, panel_name in CLINICAL_DRUGS.items():
        lbls = get_clinical_labels(tx, tcga_name, oof.index.tolist())
        if len(lbls) < 15 or sum(lbls.values()) < 2 or (len(lbls) - sum(lbls.values())) < 2:
            log(f"  skip {tcga_name}: n={len(lbls)} pos={sum(lbls.values())}")
            continue
        pids = sorted(lbls.keys())
        y = np.array([lbls[p] for p in pids], dtype=int)
        score = oof.loc[pids, f"pred_{panel_name}"].values  # higher IC50 = non-responder
        try:
            auc = roc_auc_score(y, score)
        except Exception:
            auc = float('nan')
        nb_m, nb_all = net_benefit(y, score, thresholds)
        for pt, nm, na in zip(thresholds, nb_m, nb_all):
            dca_rows.append({
                'drug': tcga_name, 'panel_drug': panel_name, 'threshold': float(pt),
                'nb_model': float(nm), 'nb_treat_all': float(na), 'nb_treat_none': 0.0,
            })
        dca_summary[tcga_name] = {
            'panel_drug': panel_name,
            'n_patients': int(len(y)),
            'n_nonresponder': int(y.sum()),
            'n_responder':    int(len(y) - y.sum()),
            'auc_nonresponse': float(auc),
            # integrated NB vs treat-all over the range where NB_model > NB_all
            'auc_dca_relative': float(np.trapz(np.maximum(nb_m - nb_all, 0), thresholds)),
            'range_dominant_pt': [float(thresholds[(nb_m > nb_all) & (nb_m > 0)].min()) if
                                  np.any((nb_m > nb_all) & (nb_m > 0)) else None,
                                  float(thresholds[(nb_m > nb_all) & (nb_m > 0)].max()) if
                                  np.any((nb_m > nb_all) & (nb_m > 0)) else None],
        }
        log(f"  {tcga_name:15s}: n={len(y)} NR={y.sum()} AUC={auc:.3f} "
            f"AUC_DCA_rel={dca_summary[tcga_name]['auc_dca_relative']:.4f}")

    pd.DataFrame(dca_rows).to_csv(os.path.join(OUT, 'decision_curves.csv'), index=False)
    with open(os.path.join(OUT, 'decision_curves.json'), 'w') as f:
        json.dump({'thresholds': thresholds.tolist(), 'summary': dca_summary}, f, indent=2)

    # =========================================================================
    # (b) ER+/HER2- treatment recommendation scenario
    # =========================================================================
    common = oof.index.intersection(trans.index)
    log(f"Overlap OOF+trans: {len(common)}")
    esr1_col = 'ESR1' if 'ESR1' in trans.columns else None
    erbb2_col = 'ERBB2' if 'ERBB2' in trans.columns else None
    if esr1_col is None or erbb2_col is None:
        # fallback: search case-insensitively
        cols = {c.upper(): c for c in trans.columns}
        esr1_col = cols.get('ESR1', esr1_col)
        erbb2_col = cols.get('ERBB2', erbb2_col)
    log(f"ESR1 col: {esr1_col}  ERBB2 col: {erbb2_col}")

    esr1 = np.log1p(trans.loc[common, esr1_col].astype(float).values)
    erbb2 = np.log1p(trans.loc[common, erbb2_col].astype(float).values)
    er_pos = esr1 >= np.median(esr1)
    her2_neg = erbb2 <  np.percentile(erbb2, 75)
    subset = common[er_pos & her2_neg]
    log(f"ER+/HER2- proxy subset: {len(subset)}")

    fulv_pred = oof.loc[subset, 'pred_Fulvestrant_1816'].values
    nr_cutoff = np.percentile(fulv_pred, 75)
    nr_mask = fulv_pred >= nr_cutoff
    nr_pids = subset[nr_mask].tolist()
    log(f"Predicted fulvestrant non-responders (top quartile within ER+/HER2-): {len(nr_pids)}")

    # per-patient alternative drug (lowest predicted IC50 in panel)
    scenario = []
    for pid in nr_pids:
        preds = {d: float(oof.loc[pid, f"pred_{d}"]) for d in PANEL_DRUGS}
        alt_drug = min(preds, key=preds.get)
        actual = tx[tx['submitter_id'] == pid]
        actual_agents = ';'.join(actual['therapeutic_agents'].dropna().astype(str).tolist())
        actual_outcomes = ';'.join(actual['treatment_outcome'].dropna().astype(str).tolist())
        scenario.append({
            'patient_id': pid,
            'ESR1_log1p': float(np.log1p(trans.loc[pid, esr1_col])),
            'ERBB2_log1p': float(np.log1p(trans.loc[pid, erbb2_col])),
            'pred_Fulvestrant_IC50': preds['Fulvestrant_1816'],
            'recommended_alternative_drug': alt_drug,
            'alt_drug_pred_IC50': preds[alt_drug],
            'fulvestrant_minus_alt_IC50': preds['Fulvestrant_1816'] - preds[alt_drug],
            'actual_treatments_received': actual_agents or 'NONE_RECORDED',
            'actual_outcomes': actual_outcomes or 'NONE_RECORDED',
        })
    pd.DataFrame(scenario).to_csv(os.path.join(OUT, 'scenario_er_her2.csv'), index=False)

    alt_counts = pd.Series([s['recommended_alternative_drug'] for s in scenario]).value_counts().to_dict()
    log(f"Alt-drug recommendation distribution: {alt_counts}")

    # Scenario-level evidence that the recommendation aligns with observed good outcomes:
    alt_drug_hits = 0; alt_drug_tried = 0; cr_pr = 0
    for s in scenario:
        drugs_received = s['actual_treatments_received'].lower()
        rec = s['recommended_alternative_drug'].split('_')[0].lower()
        if rec in drugs_received:
            alt_drug_tried += 1
            if 'complete response' in s['actual_outcomes'].lower() or \
               'partial response' in s['actual_outcomes'].lower():
                alt_drug_hits += 1
        if 'complete response' in s['actual_outcomes'].lower() or \
           'partial response' in s['actual_outcomes'].lower():
            cr_pr += 1
    scenario_eval = {
        'n_subset_er_her2': int(len(subset)),
        'n_predicted_nonresponders': int(len(nr_pids)),
        'recommendation_distribution': alt_counts,
        'n_received_recommended_drug': int(alt_drug_tried),
        'n_good_response_on_recommended_drug': int(alt_drug_hits),
        'n_good_response_any': int(cr_pr),
    }

    summary = {
        'decision_curves': dca_summary,
        'scenario': scenario_eval,
    }
    with open(os.path.join(OUT, 'clinical_utility_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log("Saved clinical_utility_summary.json")
    log(f"Scenario: {scenario_eval}")


if __name__ == '__main__':
    main()
