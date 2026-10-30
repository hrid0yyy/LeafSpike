"""Collect finished jobs into results tables and the accuracy-vs-energy figure (runs/report/ or results/)."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .energy import compute_mem_lb, compute_mem_onchip

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


COLS = ['acc_fp32', 'f1_fp32', 'acc_ref', 'f1_ref', 'params_M', 'dense_GMACs', 'GSOPs',
        'avg_spikes_per_neuron', 'compute_fp32_mJ', 'compute_int8_mJ', 'mem_int8_mJ', 'mem_int8_lb_mJ',
        'mem_int8_onchip_mJ', 'mem_int8_onchip_lb_mJ', 'latency_ms_bs1', 'train_hours', 'acc_other_attn',
        'acc_other_fp32', 'zs_acc', 'zs_f1', 'flips_fp16', 'flips_run1_run2', 'flips_bs1']


def _job_dir(runs_dir, jid):
    p1 = os.path.join(runs_dir, jid)
    p2 = os.path.join(runs_dir, 'runs', jid)
    if os.path.exists(p1):
        return p1
    return p2 if os.path.exists(p2) else None


def _prediction_metrics(path):
    """Accuracy and macro-F1 from a saved per-image prediction CSV (no torch needed)."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    labels, preds = df['label'].to_numpy(), df['pred'].to_numpy()
    acc = float((labels == preds).mean() * 100)
    try:
        from sklearn.metrics import f1_score
        f1 = float(f1_score(labels, preds, average='macro', zero_division=0) * 100)
    except Exception:
        f1 = None
    return acc, f1


def _index_csv(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if 'job' not in df.columns:
        return {}
    return {str(r['job']): r for _, r in df.iterrows()}


def collect(cfg, jobs):
    rows = []
    runs_dir = cfg['runs_dir']
    metrics = _index_csv(os.path.join(runs_dir, 'eval_metrics.csv')) or \
        _index_csv(os.path.join(runs_dir, 'report', 'eval_metrics.csv'))
    zeroshot = _index_csv(os.path.join(runs_dir, 'zeroshot.csv')) or \
        _index_csv(os.path.join(runs_dir, 'report', 'zeroshot.csv'))
    for j in jobs:
        jdir = _job_dir(runs_dir, j['id'])
        p = os.path.join(jdir, 'result.json') if jdir else None
        r = read_json(p) if p else None
        if not r:
            continue
        prof = r.get('profile', {})
        other = next((v for k, v in r.items() if k.startswith('lab_test_as_')), None)
        mem_ub = prof.get('mem_int8_mJ')
        mem_lb = prof.get('mem_int8_lb_mJ')
        if mem_lb is None and prof:
            mem_lb = compute_mem_lb(prof)
        onchip_ub = prof.get('mem_int8_onchip_mJ')
        onchip_lb = prof.get('mem_int8_onchip_lb_mJ')
        if prof:
            if onchip_ub is None:
                onchip_ub = compute_mem_onchip(prof, lower_bound=False)
            if onchip_lb is None:
                onchip_lb = compute_mem_onchip(prof, lower_bound=True)

        acc_fp32, f1_fp32 = _prediction_metrics(os.path.join(jdir, 'test_predictions_fp32.csv'))
        m = metrics.get(j['id'])
        z = zeroshot.get(j['id'])

        rows.append(dict(model=j['name'], seed=j['seed'], phase=j['phase'], kind=j['kind'],
                         pretrained=r.get('pretrained'),
                         acc_fp32=acc_fp32, f1_fp32=f1_fp32,
                         acc_ref=r['lab_test']['acc'] * 100, f1_ref=r['lab_test']['macro_f1'] * 100,
                         params_M=prof.get('params_M'), dense_GMACs=prof.get('dense_GMACs'), GSOPs=prof.get('GSOPs'),
                         avg_spikes_per_neuron=prof.get('avg_spikes_per_neuron'),
                         compute_fp32_mJ=prof.get('compute_fp32_mJ'), compute_int8_mJ=prof.get('compute_int8_mJ'),
                         mem_int8_mJ=mem_ub, mem_int8_lb_mJ=mem_lb,
                         mem_int8_onchip_mJ=onchip_ub, mem_int8_onchip_lb_mJ=onchip_lb,
                         latency_ms_bs1=r.get('latency_ms_bs1'),
                         train_hours=r.get('train_hours'),
                         acc_other_attn=(other['acc'] * 100 if other else None),
                         acc_other_fp32=(float(m['acc_other_fp32']) if m is not None and not pd.isna(m.get('acc_other_fp32')) else None),
                         zs_acc=(float(z['field_acc']) if z is not None else
                                 (r['zeroshot_field']['acc'] * 100 if 'zeroshot_field' in r else None)),
                         zs_f1=(float(z['field_f1']) if z is not None else
                                (r['zeroshot_field']['macro_f1'] * 100 if 'zeroshot_field' in r else None)),
                         flips_fp16=(float(m['flips_fp16']) if m is not None and not pd.isna(m.get('flips_fp16')) else None),
                         flips_run1_run2=(float(m['flips_run1_run2']) if m is not None and not pd.isna(m.get('flips_run1_run2')) else None),
                         flips_bs1=(float(m['flips_bs1']) if m is not None and not pd.isna(m.get('flips_bs1')) else None)))
    return pd.DataFrame(rows)


def fmt(mean, std, n, digits):
    if pd.isna(mean):
        return '-'
    if n > 1 and not pd.isna(std):
        return f'{mean:.{digits}f} ± {std:.{digits}f}'
    return f'{mean:.{digits}f}'


def write_report(cfg, jobs, out_dir=None):
    if out_dir is None:
        if os.path.exists(os.path.join(cfg['runs_dir'], 'summary.md')):
            out = cfg['runs_dir']
        else:
            out = os.path.join(cfg['runs_dir'], 'report')
    else:
        out = out_dir
    os.makedirs(out, exist_ok=True)
    df = collect(cfg, jobs)
    if df.empty:
        return None
    df.to_csv(os.path.join(out, 'all_runs.csv'), index=False)
    num = df[['model'] + COLS].apply(pd.to_numeric, errors='coerce').assign(model=df.model)
    g = num.groupby('model', sort=False)
    mean, std, n = g.mean(), g.std(), g.size()
    mean.assign(n_seeds=n).to_csv(os.path.join(out, 'summary_mean.csv'))
    std.to_csv(os.path.join(out, 'summary_std.csv'))

    digits = dict(acc_fp32=2, f1_fp32=2, acc_ref=2, f1_ref=2, params_M=2, dense_GMACs=3, GSOPs=3,
                   avg_spikes_per_neuron=3, compute_fp32_mJ=3, compute_int8_mJ=4, mem_int8_mJ=3,
                   mem_int8_lb_mJ=3, mem_int8_onchip_mJ=3, mem_int8_onchip_lb_mJ=3, latency_ms_bs1=1,
                   train_hours=2, acc_other_attn=2, acc_other_fp32=2, zs_acc=2, zs_f1=2,
                   flips_fp16=1, flips_run1_run2=1, flips_bs1=1)
    head = ['model', 'seeds'] + COLS
    lines = ['# Results (lab test = PlantVillage-tomato, duplicate-aware stratified split)', '',
             'Accuracy/F1 come from the deterministic FP32 evaluation (`test_predictions_fp32.csv`); '
             '`acc_ref`/`f1_ref` keep the training-time (fp16) evaluation as a reference. '
             'Energy columns are estimates (see experiments/lsx/energy.py); latency is PyTorch batch-1 on this GPU '
             '(spiking model runs as dense tensors).', '',
             '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for m in mean.index:
        cells = [m, str(int(n[m]))] + [fmt(mean.loc[m, c], std.loc[m, c], n[m], digits[c]) for c in COLS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    SHORT_NAMES = {
        'deit_s': 'DeiT-S',
        'swin_t': 'Swin-T',
        'mobilevit_xs': 'M-ViT-XS',
        'mobilevit_xxs': 'M-ViT-XXS',
        'leafspike': 'LeafSpike (with shortcut)',
        'leafspike_no_maxshort': 'LeafSpike-S',
        'leafspike_final_sparse_w10': 'LeafSpike-S-sparse',
        'leafspike_final_sparse_w5': 'LS-S-sparse (w=5)',
        'leafspike_integer_attn': 'LS (IntAttn)',
        'leafspike_D2': 'LS (D=2)',
        'leafspike_D1': 'LS (D=1)',
        'leafspike_firereg': 'LS (Reg)',
        'leafspike_D2_firereg': 'LS (D=2+Reg)',
        'leafspike_sparse_w10': 'LS (w=10)',
        'leafspike_sparse_w30': 'LS (w=30)',
    }

    T1_KEYS = ['deit_s', 'swin_t', 'mobilevit_xs', 'mobilevit_xxs',
               'leafspike_no_maxshort', 'leafspike_final_sparse_w10']
    T1_LABELS = {
        'deit_s': 'DeiT-S',
        'swin_t': 'Swin-T',
        'mobilevit_xs': 'MobileViT-XS',
        'mobilevit_xxs': 'MobileViT-XXS',
        'leafspike_no_maxshort': 'LeafSpike-S',
        'leafspike_final_sparse_w10': 'LeafSpike-S-sparse (w=10)',
    }

    excluded = []
    for m in mean.index:
        if mean.loc[m, 'acc_fp32'] < 90.0:
            excluded.append(f"{SHORT_NAMES.get(m, m)} ({mean.loc[m, 'acc_fp32']:.1f}%)")

    styles = {
        'deit_s': dict(color='#1f77b4', marker='s'),
        'swin_t': dict(color='#17becf', marker='^'),
        'mobilevit_xs': dict(color='#9467bd', marker='D'),
        'mobilevit_xxs': dict(color='#2ca02c', marker='v'),
        'leafspike_no_maxshort': dict(color='#ff7f0e', marker='o'),
        'leafspike_final_sparse_w10': dict(color='#d62728', marker='*'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    axes = axes.ravel()
    panels = [
        (axes[0], 'compute_fp32_mJ', None, '(a) FP32 arithmetic',
         'Arithmetic energy, FP32 45 nm (mJ, log)'),
        (axes[1], 'compute_int8_mJ', None, '(b) INT8 arithmetic',
         'Arithmetic energy, INT8 45 nm (mJ, log)'),
        (axes[2], 'mem_int8_mJ', 'mem_int8_lb_mJ', '(c) Memory, DRAM weights',
         'INT8 compute + memory, DRAM weights (mJ, log)'),
        (axes[3], 'mem_int8_onchip_mJ', 'mem_int8_onchip_lb_mJ', '(d) Memory, on-chip weights',
         'INT8 compute + memory, on-chip weights (mJ, log)'),
    ]

    handles, labels = [], []
    for ax, col, lb_col, title, xlabel in panels:
        for m in mean.index:
            y = mean.loc[m, 'acc_fp32']
            x = mean.loc[m, col]
            if pd.isna(x) or pd.isna(y) or x <= 0 or y < 90.0:
                continue
            snn = m.startswith('leafspike')
            key_model = m in T1_KEYS
            if key_model:
                color = styles[m]['color']
                marker = styles[m]['marker']
                size = 7 if marker == '*' else 5.5
            else:
                color = '0.55'
                marker = 'o'
                size = 3.0

            if lb_col and snn:
                x_lb = mean.loc[m, lb_col]
                if not pd.isna(x_lb) and x_lb < x:
                    ax.plot([x_lb, x], [y, y], color=color, linewidth=2.2, alpha=0.85, zorder=2,
                            solid_capstyle='butt')
                    ax.plot([x_lb, x_lb], [y - 0.05, y + 0.05], color=color, linewidth=1.4, zorder=2)
                    ax.plot([x, x], [y - 0.05, y + 0.05], color=color, linewidth=1.4, zorder=2)

            ax.errorbar(x, y, yerr=(std.loc[m, 'acc_fp32'] if n[m] > 1 else None), fmt=marker,
                        color=color, capsize=2.5, markersize=size,
                        alpha=1.0 if key_model else 0.75, zorder=3 if key_model else 2)
            if ax is axes[0] and key_model:
                handles.append(plt.Line2D([], [], marker=marker, linestyle='none',
                                          color=color, markersize=size))
                labels.append(T1_LABELS[m])
            if key_model and m in ('deit_s', 'swin_t'):
                # DeiT-S and Swin-T nearly coincide; separate the labels with short leader lines.
                off = (-62, 14) if m == 'deit_s' else (-62, -16)
                ax.annotate(T1_LABELS[m], (x, y), fontsize=6.5, color=color,
                            xytext=off, textcoords='offset points', ha='right', va='center',
                            arrowprops=dict(arrowstyle='-', lw=0.6, color=color,
                                            shrinkA=0, shrinkB=3))
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('lab test accuracy (%)', fontsize=8)
        ax.set_ylim(97.0, 100.2)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)

    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.07, 1, 1.0])

    os.makedirs(os.path.join(ROOT, 'paper', 'figures'), exist_ok=True)
    plt.savefig(os.path.join(out, 'accuracy_vs_energy.png'), dpi=150)
    plt.savefig(os.path.join(ROOT, 'paper', 'figures', 'accuracy_vs_energy.png'), dpi=300)
    plt.savefig(os.path.join(ROOT, 'paper', 'figures', 'accuracy_vs_energy.pdf'))
    plt.close(fig)
    return out
