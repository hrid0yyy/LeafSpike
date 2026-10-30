"""Data figures for the manuscript (cross-mode accuracy, per-layer firing). Reads only committed results.

Run from the repo root:  python figures/make_paper_figures.py
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'figures')

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'legend.fontsize': 6.8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'savefig.dpi': 600,
})
C_TRAINED, C_OTHER = '#0072B2', '#D55E00'


def eval_rows():
    rows = defaultdict(list)
    with open(os.path.join(ROOT, 'results', 'eval_metrics.csv'), newline='') as f:
        for r in csv.DictReader(f):
            rows[r['name']].append(r)
    return rows


def cross_mode_figure():
    rows = eval_rows()
    # (name, label, trained mode)
    spec = [
        ('leafspike_integer_attn', 'Integer-trained\n$D{=}4$', 'integer'),
        ('leafspike_no_maxshort', 'Aligned-trained\n$D{=}4$ (final)', 'aligned'),
        ('leafspike_final_sparse_w5', 'Aligned, reg.\n$w{=}5$', 'aligned'),
        ('leafspike_final_sparse_w10', 'Aligned, reg.\n$w{=}10$ (sparse)', 'aligned'),
        ('leafspike_sparse_w30', 'Aligned, reg.\n$w{=}30$', 'aligned'),
        ('leafspike_D2', 'Aligned\n$D{=}2$', 'aligned'),
        ('leafspike_D1', 'Aligned\n$D{=}1$', 'aligned'),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    x = np.arange(len(spec))
    w = 0.36
    for i, (name, label, mode) in enumerate(spec):
        rs = rows[name]
        a = np.array([float(r['acc_fp32']) for r in rs])
        o = np.array([float(r['acc_other_fp32']) for r in rs])
        for vals, dx, col in ((a, -w / 2, C_TRAINED), (o, w / 2, C_OTHER)):
            m = vals.mean()
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.bar(x[i] + dx, m, w * 0.92, color=col, alpha=0.85, edgecolor='none', zorder=2)
            if len(vals) > 1:
                ax.errorbar(x[i] + dx, m, yerr=sd, fmt='none', ecolor='black', elinewidth=0.6, capsize=1.6, zorder=3)
                ax.scatter(np.full(len(vals), x[i] + dx), vals, s=4, color='black', zorder=4, linewidths=0)
            ax.text(x[i] + dx, m + (sd if len(vals) > 1 else 0) + 1.2, f'{m:.1f}', ha='center', va='bottom',
                    fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] + f'\n(n={len(rows[s[0]])})' for s in spec])
    ax.set_ylabel('Lab test accuracy (%)')
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis='y', linewidth=0.4, alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_TRAINED, label='evaluated in the trained attention mode'),
                       Patch(color=C_OTHER, label='evaluated in the other attention mode')],
              loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'cross_mode.{ext}'), dpi=600 if ext == 'png' else None)
    plt.close(fig)


def layer_stats(name, seeds=(42, 43, 44)):
    per = []
    for s in seeds:
        p = os.path.join(ROOT, 'results', 'runs', f'{name}_s{s}', 'result.json')
        if os.path.exists(p):
            with open(p) as f:
                per.append(json.load(f)['profile']['layers'])
    tags = [l['layer'] for l in per[0]]
    spk = np.array([[l['mean_spikes_per_neuron'] for l in L] for L in per])
    return tags, spk


def firing_figure():
    tags, a = layer_stats('leafspike_no_maxshort')
    tags2, b = layer_stats('leafspike_final_sparse_w10')
    assert tags == tags2
    idx = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    ax.plot(idx, a.mean(0), color=C_TRAINED, lw=0.9, label='LeafSpike-S (final)')
    ax.plot(idx, b.mean(0), color=C_OTHER, lw=0.9, label='LeafSpike-S-sparse ($w{=}10$)')
    ax.fill_between(idx, a.min(0), a.max(0), color=C_TRAINED, alpha=0.2, lw=0)
    ax.fill_between(idx, b.min(0), b.max(0), color=C_OTHER, alpha=0.2, lw=0)
    # stage boundaries
    def first(prefix):
        return next(i for i, t in enumerate(tags) if t.startswith(prefix))
    bounds = [('Stage 1', 0), ('Stage 2', first('downsample2')), ('Stage 3', first('downsample3')),
              ('Stage 4', first('downsample4'))]
    for k, (lab, i0) in enumerate(bounds):
        if i0 > 0:
            ax.axvline(i0 - 0.5, color='black', lw=0.5, ls=':', alpha=0.7)
        i1 = bounds[k + 1][1] if k + 1 < len(bounds) else len(tags)
        ax.text((i0 + i1) / 2, 2.12, lab, ha='center', va='bottom', fontsize=6.5)
    qkv = [i for i, t in enumerate(tags) if t.endswith(('q_spike', 'k_spike', 'v_spike'))]
    ax.scatter(qkv, b.mean(0)[qkv], s=5, color=C_OTHER, zorder=3, linewidths=0)
    ax.scatter(qkv, a.mean(0)[qkv], s=5, color=C_TRAINED, zorder=3, linewidths=0,
               label='$q$/$k$/$v$ spike layers (dots)')
    ax.set_xlim(-1, len(tags))
    ax.set_ylim(0, 2.4)
    ax.set_xlabel('Spiking layer (network order)')
    ax.set_ylabel('Mean spikes / neuron')
    ax.grid(axis='y', linewidth=0.4, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower center', frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'firing_profile.{ext}'), dpi=600 if ext == 'png' else None)
    plt.close(fig)
    print('layers', len(tags), 'mean spikes final %.3f sparse %.3f' % (a.mean(), b.mean()))


if __name__ == '__main__':
    cross_mode_figure()
    firing_figure()
