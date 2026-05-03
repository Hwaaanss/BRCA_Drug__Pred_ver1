"""Fig 4 — Biological interpretation.

Motif: Kinker/Nat Genet 2020 UMAPs + Gonçalves/Cancer Cell clustermap + L1000 correlation matrix.
 a  UMAP by subtype with gaussian density contours.
 b  UMAP by Tamoxifen IC50 (viridis continuous).
 c  Subtype × drug z-scored IC50 clustermap (RdBu_r) with row dendrogram & MOA colour bar.
 d  Drug×drug Spearman correlation matrix, mask upper triangle, MOA sidebar.
 e  Polar small-multiples: per-subtype radar of z-scored drug sensitivity.
"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.gridspec as gs
from scipy.stats import spearmanr, gaussian_kde, zscore
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.patches import Rectangle
from . import style as S
from .loaders import load_umap_patient, DATA_ROOT, DRUG_MOA, DRUG_ORDER_13


def _load_ic50_with_subtype():
    u = load_umap_patient()
    ic = pd.read_csv(DATA_ROOT / "07_integrated" / "predicted_IC50_all_drugs.csv", index_col=0)
    # rename IC50 columns to drug short names if needed; keep 13
    ren = {c: c.split('_')[0] for c in ic.columns}
    ic = ic.rename(columns=ren)
    ic = ic.loc[:, ~ic.columns.duplicated()]  # drop duplicated short-names
    drugs = [d for d in DRUG_ORDER_13 if d in ic.columns]
    ic = ic[drugs]
    df = u.merge(ic, left_on='patient_id', right_index=True, how='inner')
    return df, drugs


def panel_a_umap_subtype(ax):
    df, _ = _load_ic50_with_subtype()
    subs = ['Luminal', 'HER2+', 'TNBC']
    # If 'TNBC' not in data, match 'Triple-Negative'
    if 'TNBC' not in df['subtype'].unique():
        df['subtype'] = df['subtype'].replace({'Triple-Negative':'TNBC'})
    for s in subs:
        sub = df[df['subtype'] == s]
        if len(sub) < 3: continue
        ax.scatter(sub['x'], sub['y'], s=8, c=S.SUB[s],
                   edgecolor='none', alpha=0.80, label=f'{s} (n={len(sub)})')
        # density contour
        try:
            k = gaussian_kde(np.vstack([sub['x'], sub['y']]))
            X, Y = np.mgrid[df['x'].min():df['x'].max():60j,
                            df['y'].min():df['y'].max():60j]
            Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=3, colors=S.SUB[s],
                       linewidths=0.6, alpha=0.6)
        except Exception:
            pass
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=6.2, frameon=True,
              framealpha=0.9, edgecolor='#ddd', handlelength=0.8,
              handletextpad=0.3, borderpad=0.3)
    S.despine(ax)


def panel_b_umap_drug(ax):
    df, _ = _load_ic50_with_subtype()
    sc = ax.scatter(df['x'], df['y'], c=df['Tamoxifen'], cmap='viridis',
                    s=8, edgecolor='none', alpha=0.88, vmin=df['Tamoxifen'].quantile(0.02),
                    vmax=df['Tamoxifen'].quantile(0.98))
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, aspect=18)
    cb.ax.tick_params(labelsize=6); cb.set_label(r'Tamoxifen IC$_{50}$', fontsize=6.5)
    S.despine(ax)


def panel_c_subtype_drug(ax):
    df, drugs = _load_ic50_with_subtype()
    if 'TNBC' not in df['subtype'].unique():
        df['subtype'] = df['subtype'].replace({'Triple-Negative':'TNBC'})
    subs = ['Luminal', 'HER2+', 'TNBC']
    df = df[df['subtype'].isin(subs)]
    # z-score per drug across all patients
    zmat = df.groupby('subtype')[drugs].mean().loc[subs]
    zmat = (zmat - df[drugs].mean()) / df[drugs].std()
    # cluster columns (drugs)
    link = linkage(zmat.T.values, method='average', metric='euclidean')
    col_order = leaves_list(link)
    cols = list(zmat.columns)
    zmat = zmat.iloc[:, col_order]
    drugs_ord = [cols[i] for i in col_order]

    im = ax.imshow(zmat.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    # MOA colour bar above columns
    for j, d in enumerate(drugs_ord):
        c = S.MOA.get(DRUG_MOA.get(d, ''), '#ccc')
        ax.add_patch(Rectangle((j - 0.5, -0.9), 1, 0.4, facecolor=c, edgecolor='white', lw=0.4,
                               clip_on=False))
    ax.set_xticks(np.arange(len(drugs_ord))); ax.set_xticklabels(drugs_ord, rotation=40, ha='right', fontsize=6.2)
    ax.set_yticks(np.arange(len(subs))); ax.set_yticklabels(subs, fontsize=7)
    # add significance stars (Kruskal-Wallis proxy from variance)
    from scipy.stats import kruskal
    for j, d in enumerate(drugs_ord):
        vals_by_sub = [df[df['subtype']==s][d].values for s in subs]
        try:
            _, p = kruskal(*vals_by_sub)
            p = float(p)
        except Exception:
            p = 1.0
        star = S.sig_stars(p)
        for i in range(len(subs)):
            ax.text(j, i, star if abs(zmat.iloc[i, j]) > 0.5 else '',
                    ha='center', va='center', fontsize=5.5, color='#222')
    cb = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.02, aspect=16, orientation='vertical')
    cb.set_label(r'z-score IC$_{50}$', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    S.despine(ax, left=False, bottom=False)
    ax.set_ylim(len(subs) - 0.5, -1.1)


def panel_d_corrmat(ax):
    df, drugs = _load_ic50_with_subtype()
    # reorder by MOA cluster
    moa_order = ['DNA damage','Tubulin','Hormone','Kinase','mTOR/NAD','Apoptosis']
    drugs_ord = sorted(drugs, key=lambda d: (moa_order.index(DRUG_MOA.get(d, 'Kinase'))
                                              if DRUG_MOA.get(d) in moa_order else 99))
    C = df[drugs_ord].corr(method='spearman').values
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)
    Cm = np.ma.masked_array(C, mask)
    im = ax.imshow(Cm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(drugs_ord))); ax.set_xticklabels(drugs_ord, rotation=40, ha='right', fontsize=6.2)
    ax.set_yticks(range(len(drugs_ord))); ax.set_yticklabels(drugs_ord, fontsize=6.2)
    # MOA stripe on left
    for i, d in enumerate(drugs_ord):
        c = S.MOA.get(DRUG_MOA.get(d, ''), '#ccc')
        ax.add_patch(Rectangle((-1.3, i - 0.5), 0.6, 1, facecolor=c, edgecolor='white', lw=0.3,
                               clip_on=False))
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, aspect=16)
    cb.set_label('Spearman ρ', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    S.despine(ax, left=False, bottom=False)


def panel_e_radar(ax_list):
    """Polar radar of z-score drug sensitivity per subtype."""
    df, drugs = _load_ic50_with_subtype()
    if 'TNBC' not in df['subtype'].unique():
        df['subtype'] = df['subtype'].replace({'Triple-Negative':'TNBC'})
    subs = ['Luminal', 'HER2+', 'TNBC']
    # dedupe & lock column order
    drugs = list(dict.fromkeys(drugs))
    means = df.groupby('subtype')[drugs].mean().loc[subs]
    zmat = (means - df[drugs].mean()) / df[drugs].std()
    drugs = list(zmat.columns)
    angles = np.linspace(0, 2*np.pi, len(drugs), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    vmax = 1.3
    for ax, s in zip(ax_list, subs):
        vals = zmat.loc[s].values
        vals_c = np.concatenate([vals, vals[:1]])
        ax.plot(closed, vals_c, color=S.SUB[s], lw=1.2)
        ax.fill(closed, vals_c, color=S.SUB[s], alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels([d[:4] for d in drugs], fontsize=5.5)
        ax.set_ylim(-vmax, vmax); ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['-1','0','1'], fontsize=5.5, color='#888')
        ax.grid(color='#ddd', linewidth=0.4)
        ax.set_title(s, fontsize=7.5, color=S.SUB[s], fontweight='bold', pad=4)
        ax.spines['polar'].set_linewidth(0.5)
        ax.spines['polar'].set_color('#bbb')


def make(out_dir):
    S.apply_rc()
    fig = plt.figure(figsize=(S.W_DOUBLE, 6.9))
    # Row 1: UMAP a,b + clustermap c;  Row 2: correlation matrix d + radar e (3 subplots)
    outer = gs.GridSpec(2, 1, height_ratios=[1, 1.1], hspace=0.42,
                        top=0.96, bottom=0.07, left=0.06, right=0.98)
    # Top row: 3 panels
    top = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 0],
                                     wspace=0.38, width_ratios=[1, 1, 1.35])
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])

    bot = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0],
                                     wspace=0.20, width_ratios=[1.2, 1.8])
    ax_d = fig.add_subplot(bot[0, 0])
    # Radar subgrid (3 polar)
    radar_grid = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=bot[0, 1], wspace=0.10)
    ax_e = [fig.add_subplot(radar_grid[0, i], projection='polar') for i in range(3)]

    panel_a_umap_subtype(ax_a)
    panel_b_umap_drug(ax_b)
    panel_c_subtype_drug(ax_c)
    panel_d_corrmat(ax_d)
    panel_e_radar(ax_e)

    S.panel(ax_a, 'a')
    S.panel(ax_b, 'b')
    S.panel(ax_c, 'c', x=-0.12)
    S.panel(ax_d, 'd', x=-0.18)
    # panel e letter above first polar
    ax_e[0].text(-0.25, 1.18, 'e', transform=ax_e[0].transAxes,
                 fontsize=10, fontweight='bold')

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_dir}/Fig4_biological.{ext}")
    plt.close(fig)
    print('[Fig4] saved')


if __name__ == '__main__':
    make(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'figures', 'figures_v8'))
