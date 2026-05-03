"""Fig 1 — Study design & cohort (schematic + availability + CONSORT).
Motif: Chen/PORPOISE (Cancer Cell 2022) Fig 1 + Luecken scIB style cohort bars.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from . import style as S
from .loaders import load_cohort_sizes


def _box(ax, x, y, w, h, text, fc, ec, textsize=7, bold=True, alpha=1.0, text_color='#111', lw=0.8):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center', fontsize=textsize,
            fontweight='bold' if bold else 'normal', color=text_color)


def _arrow(ax, x1, y1, x2, y2, lw=0.9, color='#666'):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=6,
                        color=color, linewidth=lw, shrinkA=1.2, shrinkB=1.2)
    ax.add_patch(a)


def draw_schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    lane = {'input':0.82, 'enc':0.60, 'tok':0.44, 'fus':0.24, 'pred':0.07}
    mods = [
        ('Genomic',        'Mutations + TMB\n193 features',      S.PAL['genomic']),
        ('Transcriptomic', 'Gene expression\n19,938 features',   S.PAL['transcriptomic']),
        ('Proteomic',      'RPPA proteins\n464 features',        S.PAL['proteomic']),
        ('Histology',      'H&E WSI UNI\n1024-d patches',        S.PAL['histology']),
    ]
    enc_labels = [
        'MLP encoder\n→ 8 tokens',
        'Pathway tokeniser\n→ 2,000 tokens',
        'MLP encoder\n→ 16 tokens',
        'ABMIL aggregator\n→ 16 tokens',
    ]
    xs = np.linspace(0.09, 0.91, 4)
    w_in, h_in, w_en, h_en = 0.17, 0.13, 0.17, 0.10
    # Input cards — coloured side strip + white card
    for x, (name, sub, c) in zip(xs, mods):
        ax.add_patch(Rectangle((x - w_in/2, lane['input'] - h_in/2), 0.013, h_in,
                               facecolor=c, edgecolor='none'))
        _box(ax, x - w_in/2 + 0.013, lane['input'] - h_in/2, w_in - 0.013, h_in,
             '', fc='white', ec=c, textsize=6, bold=False, lw=0.9)
        ax.text(x + 0.006, lane['input'] + h_in/2 - 0.022, name,
                ha='center', va='top', fontsize=7.6, fontweight='bold', color=c)
        ax.text(x + 0.006, lane['input'] - h_in/2 + 0.035, sub,
                ha='center', va='center', fontsize=6.3, color='#333')
    # Encoders
    for x, label, (_, _, c) in zip(xs, enc_labels, mods):
        _box(ax, x - w_en/2, lane['enc'] - h_en/2, w_en, h_en,
             label, fc='#F6F6F6', ec='#999', textsize=6.2, bold=False)
        _arrow(ax, x, lane['input'] - h_in/2, x, lane['enc'] + h_en/2)
    # Token strips
    for x, (_, _, c) in zip(xs, mods):
        ax.add_patch(Rectangle((x - 0.065, lane['tok'] - 0.015), 0.13, 0.028,
                               facecolor=c, edgecolor='#333', linewidth=0.4, alpha=0.85))
        _arrow(ax, x, lane['enc'] - h_en/2, x, lane['tok'] + 0.013)
    # Fusion bar
    fx, fy, fw, fh = 0.12, lane['fus'] - 0.055, 0.76, 0.11
    _box(ax, fx, fy, fw, fh,
         'Bidirectional cross-attention fusion\n2 layers × 8 heads    (omics ↔ histology)',
         fc='#EAF2FB', ec=S.PAL['fusion'], textsize=7.5, bold=True,
         text_color=S.PAL['fusion'], lw=1.1)
    for x in xs:
        _arrow(ax, x, lane['tok'] - 0.015, x, fy + fh, color='#777')
    # Prediction
    _box(ax, 0.22, lane['pred'] - 0.035, 0.56, 0.075,
         'Attention pooling  →  256-d embedding  →  IC$_{50}$  (13 drugs)',
         fc=S.PAL['hero'], ec=S.PAL['hero'], textsize=7.5, bold=True,
         text_color='white', lw=0)
    _arrow(ax, 0.50, fy, 0.50, lane['pred'] + 0.04, color='#222', lw=1.2)
    # Left-side row annotations (tiny italic)
    for y, t in [(lane['input'], 'Inputs'), (lane['enc'], 'Encoders'),
                 (lane['tok'], 'Tokens'), (lane['fus'], 'Fusion'),
                 (lane['pred'], 'Prediction')]:
        ax.text(-0.005, y, t, ha='right', va='center', fontsize=6.3,
                style='italic', color='#888')


def draw_cohort_bars(ax):
    sizes = load_cohort_sizes()
    order = ['Clinical','Transcriptomic','Genomic','Drug treatment','Proteomic','Histology']
    colors = {
        'Clinical':'#6c757d', 'Transcriptomic':S.PAL['transcriptomic'], 'Genomic':S.PAL['genomic'],
        'Histology':S.PAL['histology'], 'Proteomic':S.PAL['proteomic'],
        'Drug treatment':S.PAL['hero'],
    }
    ys = np.arange(len(order))[::-1]
    vals = [sizes[k] for k in order]
    ax.barh(ys, vals, color=[colors[k] for k in order],
            edgecolor='white', linewidth=0.5, height=0.70)
    for y, v in zip(ys, vals):
        ax.text(v + 12, y, f'{v:,}', va='center', ha='left', fontsize=7, color='#222')
    ax.set_yticks(ys); ax.set_yticklabels(order)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel('Patients (n)')
    ax.axvline(431, color=S.PAL['hero'], linestyle='--', linewidth=0.7, alpha=0.8)
    ax.text(431 + 5, len(order) - 0.2, '4-modal\nintersection',
            color=S.PAL['hero'], fontsize=6.2, ha='left', va='top',
            fontweight='bold')
    S.despine(ax)


def draw_consort(ax):
    stages = [('Transcriptomic only',  1095, S.PAL['transcriptomic']),
              ('+ Genomic',              964, '#8E7CC3'),
              ('+ Proteomic',            431, S.PAL['proteomic']),
              ('+ Histology (4-modal)',  431, S.PAL['hero'])]
    ys = np.arange(len(stages))[::-1]
    maxv = max(s[1] for s in stages)
    for y, (name, v, c) in zip(ys, stages):
        ax.barh(y, v, height=0.62, color=c, edgecolor='white', linewidth=0.5, alpha=0.95)
        ax.text(v + 12, y, f'{v:,}', va='center', fontsize=7, color='#222')
        ax.text(-18, y, name, va='center', ha='right', fontsize=7,
                fontweight='bold', color='#333')
    for i in range(len(stages)-1):
        v1, v2 = stages[i][1], stages[i+1][1]
        drop = v1 - v2
        if drop > 0:
            y_mid = (ys[i] + ys[i+1]) / 2
            ax.text(v1 * 0.5, y_mid, f'attrition −{drop}',
                    fontsize=6.2, color='#999', ha='center', va='center', style='italic')
    ax.set_xlim(-maxv*0.42, maxv * 1.14)
    ax.set_ylim(-0.6, len(stages) - 0.35)
    ax.set_yticks([])
    ax.set_xticks([0, 300, 600, 900, 1200])
    ax.set_xlabel('Patients (n)')
    ax.spines['left'].set_visible(False)
    S.despine(ax, left=False)
    ax.spines['bottom'].set_bounds(0, maxv)


def make(out_dir):
    S.apply_rc()
    fig = plt.figure(figsize=(S.W_DOUBLE, 5.1))
    grid = gs.GridSpec(2, 2, height_ratios=[1.55, 1.0],
                       width_ratios=[1.05, 1.0], hspace=0.48, wspace=0.40,
                       top=0.97, bottom=0.08, left=0.055, right=0.985)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    draw_schematic(ax_a)
    draw_cohort_bars(ax_b)
    draw_consort(ax_c)

    S.panel(ax_a, 'a', x=0.005, y=1.00)
    S.panel(ax_b, 'b', x=-0.22, y=1.06)
    S.panel(ax_c, 'c', x=-0.32, y=1.06)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_dir}/Fig1_study_design.{ext}")
    plt.close(fig)
    print('[Fig1] saved')


if __name__ == '__main__':
    make(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'research', 'figures', 'figures_v8'))
