"""Fig 8 — Modality-dropout robustness (Boehm/Nat Cancer motif).

Line + 95 % CI ribbon. Three architectures over 0/1/2/3 dropped modalities.
Annotate endpoint retention; shaded reference band for full model.
"""
import os, numpy as np, matplotlib.pyplot as plt
from scipy.stats import sem, t
from . import style as S
from .loaders import load_robustness


def make(out_dir):
    S.apply_rc()
    fig, ax = plt.subplots(figsize=(S.W_ONEHALF, 3.6))
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.13, right=0.97)

    d = load_robustness()['retention_summary']
    colors = {
        'PathOmicDRP (Cross-Attention)': S.PAL['hero'],
        'EarlyFusionMLP':                '#0072B2',
        'SelfAttnOnly':                  '#E69F00',
    }
    labels_display = {
        'PathOmicDRP (Cross-Attention)': 'PathOmicDRP (cross-attention)',
        'EarlyFusionMLP':                'Early-fusion MLP',
        'SelfAttnOnly':                  'Self-attention only',
    }
    xs = np.array([0, 1, 2, 3])

    for method, entry in d.items():
        means = np.array([entry[str(k)]['mean'] for k in xs])
        stds  = np.array([entry[str(k)]['std']  for k in xs])
        ns    = np.array([max(len(entry[str(k)]['values']), 1) for k in xs])
        # 95% CI
        ci = 1.96 * stds / np.sqrt(ns)
        col = colors.get(method, '#555')
        ax.plot(xs, means, marker='o', markersize=5, linewidth=1.4,
                color=col, label=labels_display.get(method, method),
                markerfacecolor=col, markeredgecolor='#222', markeredgewidth=0.4)
        ax.fill_between(xs, means - ci, means + ci, color=col, alpha=0.16, linewidth=0)
        # endpoint annotation
        ax.text(3.08, means[-1], f'{means[-1]:.0f}%',
                va='center', fontsize=6.5, color=col, fontweight='bold')

    ax.axhline(100, color='#888', linestyle='--', lw=0.5)
    ax.axhspan(100, 101, color='#ccc', alpha=0.0)  # reference placeholder

    # Gap highlight between PathOmicDRP and SelfAttn at x=3
    y1 = d['PathOmicDRP (Cross-Attention)']['3']['mean']
    y2 = d['SelfAttnOnly']['3']['mean']
    ax.annotate('', xy=(3.35, y1), xytext=(3.35, y2),
                arrowprops=dict(arrowstyle='<->', lw=0.8, color='#555'))
    ax.text(3.45, (y1 + y2) / 2, f'+{y1 - y2:.1f} pp\n(cross-attn\nrobustness)',
            fontsize=6, color='#333', va='center')

    ax.set_xticks(xs)
    ax.set_xticklabels(['0\n(full)', '1', '2', '3\n(single modality)'], fontsize=7)
    ax.set_xlabel('Number of dropped modalities')
    ax.set_ylabel(r'PCC$_{drug}$ retention (%)')
    ax.set_xlim(-0.2, 4.1)
    ax.set_ylim(25, 108)
    ax.legend(loc='lower left', fontsize=6.8, handlelength=1.4, frameon=False)
    S.despine(ax)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_dir}/Fig8_modality_robustness.{ext}")
    plt.close(fig)
    print('[Fig8] saved')


if __name__ == '__main__':
    make(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'figures', 'figures_v8'))
