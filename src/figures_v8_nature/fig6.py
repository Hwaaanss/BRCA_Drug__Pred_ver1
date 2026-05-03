"""Fig 6 — Generalisation (LODO, multi-task, phenotype).

 a  Leave-one-drug-out PCC forest with 95% CI, ordered, median reference.
 b  Drug-drug attention (prediction) similarity heatmap (mako).
 c  Single vs multi-task dumbbell per drug.
 d  H&E phenotype UMAP with discrete clusters (2500 patches).
 e  Phenotype × drug heatmap with cluster ribbon on rows.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.gridspec as gs
from matplotlib.patches import Rectangle
from . import style as S
from .loaders import J, DATA_ROOT, load_lodo, load_multitask, load_phenotype, DRUG_ORDER_13, DRUG_MOA


def panel_a_lodo(ax):
    d = load_lodo()
    drugs = list(d.keys())
    drugs.sort(key=lambda k: -d[k]['pcc_mean'])
    means = [d[k]['pcc_mean'] for k in drugs]
    stds  = [d[k]['pcc_std']  for k in drugs]
    ys = np.arange(len(drugs))
    grand = np.mean(means)
    # colour by distance from median
    colors = [S.PAL['hero'] if m > grand else '#8B99AE' for m in means]
    for y, m, s, c in zip(ys, means, stds, colors):
        ax.errorbar(m, y, xerr=s, fmt='o', markersize=4.5, color=c,
                    markeredgecolor='#222', markeredgewidth=0.4,
                    elinewidth=0.7, capsize=2, zorder=5)
    ax.axvline(grand, color='#333', ls='--', lw=0.7)
    ax.text(grand, len(drugs) - 0.2, f'mean = {grand:.2f}',
            fontsize=6.2, color='#333', ha='left', va='top')
    ax.set_yticks(ys); ax.set_yticklabels(drugs, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel('PCC (leave-one-drug-out)')
    ax.set_xlim(0.70, 1.0)
    S.despine(ax)


def panel_b_attn_corr(ax):
    """Drug-drug predicted-IC50 correlation as proxy for attention similarity."""
    ic = pd.read_csv(DATA_ROOT / "07_integrated" / "predicted_IC50_all_drugs.csv", index_col=0)
    ren = {c: c.split('_')[0] for c in ic.columns}
    ic = ic.rename(columns=ren)
    ic = ic.loc[:, ~ic.columns.duplicated()]
    drugs = [d for d in DRUG_ORDER_13 if d in ic.columns]
    C = ic[drugs].corr(method='spearman').values
    im = ax.imshow(C, cmap='YlOrRd', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(drugs))); ax.set_xticklabels(drugs, rotation=40, ha='right', fontsize=6)
    ax.set_yticks(range(len(drugs))); ax.set_yticklabels(drugs, fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=16)
    cb.set_label('Correlation', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    S.despine(ax, left=False, bottom=False)


def panel_c_multitask(ax):
    mt = load_multitask()
    cv3 = J('results/phase3_3modal_baseline/cv_results.json')['avg']
    single_pcc = cv3['pcc_per_drug_mean']
    multi_pcc = mt['drug_pcc']
    items = [
        (r'PCC$_{drug}$', single_pcc['mean'], multi_pcc['mean'],
         single_pcc['std'], multi_pcc['std']),
        ('C-index', 0.5, mt['c_index']['mean'],
         0.0, mt['c_index']['std']),
    ]
    xs = np.arange(len(items))
    w = 0.32
    m1 = [i[1] for i in items]; e1 = [i[3] for i in items]
    m2 = [i[2] for i in items]; e2 = [i[4] for i in items]
    ax.bar(xs - w/2, m1, w, yerr=e1, color='#8B99AE', edgecolor='#222', linewidth=0.5,
           label='Single-task',
           error_kw={'elinewidth':0.7, 'ecolor':'#333', 'capsize':2})
    ax.bar(xs + w/2, m2, w, yerr=e2, color=S.PAL['hero'], edgecolor='#222', linewidth=0.5,
           label='Multi-task',
           error_kw={'elinewidth':0.7, 'ecolor':'#333', 'capsize':2})
    for x, v in zip(xs - w/2, m1):
        ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=6.3)
    for x, v in zip(xs + w/2, m2):
        ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=6.3,
                color=S.PAL['hero'], fontweight='bold')
    ax.axhline(0.5, color='#888', linestyle='--', lw=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([i[0] for i in items], fontsize=7)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6.5, loc='upper left', handlelength=1.2, ncol=2)
    S.despine(ax)


def panel_d_phenotype_umap(ax):
    r, cl, um = load_phenotype()
    ncl = len(np.unique(cl))
    cmap_cols = ['#E64B35', '#4DBBD5', '#00A087', '#F39B7F', '#8491B4', '#91D1C2']
    for k in range(ncl):
        m = cl == k
        info = r['summary'].get(f'cluster_{k}', {})
        lab = f'C{k} (n={info.get("n_patients","?")}pts)'
        ax.scatter(um[m, 0], um[m, 1], s=4, c=cmap_cols[k % 6],
                   edgecolor='none', alpha=0.75, label=lab)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=5.5, ncol=2, markerscale=1.8,
              frameon=True, framealpha=0.85, edgecolor='#ddd',
              handlelength=0.6, handletextpad=0.3, columnspacing=0.6, borderpad=0.3)
    S.despine(ax)


def panel_e_phenotype_drug(ax):
    r, _, _ = load_phenotype()
    summary = r['summary']
    drugs = list(next(iter(summary.values()))['drug_means'].keys())
    drugs = [d for d in DRUG_ORDER_13 if d in drugs]
    clusters = sorted(summary.keys())
    M = np.array([[summary[c]['drug_means'][d] for d in drugs] for c in clusters])
    # z-score per column
    z = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    im = ax.imshow(z, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(len(drugs))); ax.set_xticklabels(drugs, rotation=40, ha='right', fontsize=6)
    ax.set_yticks(range(len(clusters)))
    labels = [f'C{k.replace("cluster_","")} (n={summary[k]["n_patients"]}pt)' for k in clusters]
    ax.set_yticklabels(labels, fontsize=6.5)
    # cluster ribbon on left
    cmap_cols = ['#E64B35', '#4DBBD5', '#00A087', '#F39B7F', '#8491B4', '#91D1C2']
    for i, _ in enumerate(clusters):
        ax.add_patch(Rectangle((-1.4, i - 0.5), 0.55, 1, facecolor=cmap_cols[i],
                               edgecolor='white', lw=0.3, clip_on=False))
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, aspect=15)
    cb.set_label(r'z-score IC$_{50}$', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    S.despine(ax, left=False, bottom=False)


def make(out_dir):
    S.apply_rc()
    fig = plt.figure(figsize=(S.W_DOUBLE, 6.6))
    outer = gs.GridSpec(2, 1, height_ratios=[1, 1.05], hspace=0.45,
                        top=0.96, bottom=0.08, left=0.07, right=0.98)
    top = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                     wspace=0.5, width_ratios=[1, 1.1, 0.9])
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    bot = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                     wspace=0.28, width_ratios=[1, 1.15])
    ax_d = fig.add_subplot(bot[0, 0])
    ax_e = fig.add_subplot(bot[0, 1])

    panel_a_lodo(ax_a); panel_b_attn_corr(ax_b); panel_c_multitask(ax_c)
    panel_d_phenotype_umap(ax_d); panel_e_phenotype_drug(ax_e)

    S.panel(ax_a, 'a', x=-0.22); S.panel(ax_b, 'b'); S.panel(ax_c, 'c')
    S.panel(ax_d, 'd'); S.panel(ax_e, 'e', x=-0.13)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_dir}/Fig6_generalization.{ext}")
    plt.close(fig)
    print('[Fig6] saved')


if __name__ == '__main__':
    make(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'figures', 'figures_v8'))
