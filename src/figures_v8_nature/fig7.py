"""Fig 7 — Imputed-IC50 vs clinical-AUC dissociation (dumbbell motif).

Dumbbell plot: methods as rows; two dots per row (imputed-PCC vs clinical-AUC mean);
connector line coloured by improvement direction. PathOmicDRP rows highlighted.
"""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gs
from . import style as S
from .loaders import J


def _mean_clinical_auc(d, key):
    """mean clinical AUC across 3 drugs for a given method key."""
    return float(np.mean([d[drug][key]['auc_mean'] for drug in ['Docetaxel', 'Paclitaxel', 'Cyclophosphamide']]))


def panel_a_dumbbell(ax):
    cmp_ = J('results/strengthening/analysis6_clinical_auc_comparison.json')
    # Imputed PCC per method
    from src.figures_v8_nature.loaders import load_phase_cv
    cv3 = load_phase_cv('phase3_3modal_baseline')['avg']['pcc_per_drug_mean']['mean']
    cv4 = load_phase_cv('phase3_4modal_full')['avg']['pcc_per_drug_mean']['mean']
    adv = J('results/advanced_analysis/advanced_analysis_results.json')['benchmark']

    methods = [
        ('PathOmicDRP 4-modal', cv4, _mean_clinical_auc(cmp_, 'PathOmicDRP_4modal'), True),
        ('PathOmicDRP 3-modal', cv3, _mean_clinical_auc(cmp_, 'PathOmicDRP_3modal'), True),
        ('ElasticNet',          float(adv['ElasticNet']['pcc_drug'][0]),
                                _mean_clinical_auc(cmp_, 'ElasticNet_IC50_13d'),    False),
        ('Raw omics MLP',       0.45,
                                _mean_clinical_auc(cmp_, 'Raw_omics_2657d'),         False),
        ('Raw omics + H&E',     0.46,
                                _mean_clinical_auc(cmp_, 'Raw_omics+histo_3681d'),   False),
    ]
    ys = np.arange(len(methods))
    for y, (name, pcc, auc, hero) in zip(ys, methods):
        col_left = '#3C7CAE'
        col_right = S.PAL['hero'] if hero else '#8B99AE'
        line_col = S.PAL['pos'] if auc > pcc else S.PAL['neg']
        ax.plot([pcc, auc], [y, y], color=line_col, lw=1.8, alpha=0.55, solid_capstyle='round', zorder=2)
        ax.scatter(pcc, y, s=55, color=col_left, edgecolor='#222', lw=0.5,
                   zorder=5, label=r'Imputed IC$_{50}$ PCC' if y == 0 else None)
        ax.scatter(auc, y, s=70, color=col_right, edgecolor='#222', lw=0.5,
                   marker='s', zorder=5, label='Clinical response AUC' if y == 0 else None)
        ax.text(pcc, y + 0.30, f'{pcc:.2f}', ha='center', va='bottom', fontsize=6.2, color=col_left)
        ax.text(auc, y - 0.28, f'{auc:.2f}', ha='center', va='top', fontsize=6.2, color=col_right,
                fontweight='bold' if hero else 'normal')
    ax.axvline(0.5, color='#888', ls='--', lw=0.5)
    ax.text(0.5, len(methods) - 0.4, 'chance', fontsize=6, color='#888', ha='center')
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in methods], fontsize=7,
                       fontweight='bold')  # all bold; hero additionally colored
    for i, (_, _, _, hero) in enumerate(methods):
        if hero:
            ax.get_yticklabels()[i].set_color(S.PAL['hero'])
    ax.invert_yaxis()
    ax.set_xlabel('Score')
    ax.set_xlim(0.0, 1.05)
    ax.legend(loc='lower right', fontsize=6.5, handlelength=1.0, handletextpad=0.3, ncol=1)
    S.despine(ax)


def panel_b_grouped(ax):
    d = J('results/strengthening/analysis6_clinical_auc_comparison.json')
    drugs = ['Docetaxel', 'Paclitaxel', 'Cyclophosphamide']
    methods = [('PathOmicDRP 4m',   'PathOmicDRP_4modal', S.PAL['hero']),
               ('PathOmicDRP 3m',   'PathOmicDRP_3modal', S.PAL['hero_light']),
               ('ElasticNet',       'ElasticNet_IC50_13d', '#3C7CAE'),
               ('Raw omics',        'Raw_omics_2657d', '#8B99AE')]
    xs_base = np.arange(len(drugs))
    w = 0.19
    for i, (name, key, col) in enumerate(methods):
        offs = (i - 1.5) * w
        vals = [d[drug][key]['auc_mean'] for drug in drugs]
        errs = [d[drug][key]['auc_std']  for drug in drugs]
        ax.bar(xs_base + offs, vals, width=w, yerr=errs,
               color=col, edgecolor='#222', linewidth=0.45, label=name,
               error_kw={'elinewidth':0.5, 'ecolor':'#444', 'capsize':1.3})
        for x, v in zip(xs_base + offs, vals):
            ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=5.6,
                    color='#222', fontweight='bold' if 'PathOmicDRP 4m' == name else 'normal')
    ax.axhline(0.5, color='#888', ls='--', lw=0.5)
    ax.set_xticks(xs_base); ax.set_xticklabels(
        [f'{d}\n(n={J("results/strengthening/analysis6_clinical_auc_comparison.json")[d]["n"]})'
         for d in drugs], fontsize=6.8)
    ax.set_ylabel('Clinical response AUC')
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=6.3, loc='upper left', ncol=2, handlelength=1.0, handletextpad=0.4)
    S.despine(ax)


def make(out_dir):
    S.apply_rc()
    fig = plt.figure(figsize=(S.W_DOUBLE, 3.8))
    grid = gs.GridSpec(1, 2, width_ratios=[1.08, 1], wspace=0.32,
                       top=0.92, bottom=0.18, left=0.13, right=0.98)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    panel_a_dumbbell(ax_a)
    panel_b_grouped(ax_b)
    S.panel(ax_a, 'a', x=-0.22); S.panel(ax_b, 'b')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_dir}/Fig7_clinical_vs_imputed.{ext}")
    plt.close(fig)
    print('[Fig7] saved')


if __name__ == '__main__':
    make(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'figures', 'figures_v8'))
