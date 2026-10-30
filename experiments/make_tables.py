"""Generate publication-ready LaTeX tables for paper/tables/*.tex.

Reads:
- results/summary_mean.csv
- results/summary_std.csv
- results/all_runs.csv
- results/significance.md

Generates:
- paper/tables/t1_main_comparison.tex
- paper/tables/t2_ablations.tex
- paper/tables/t3_significance.tex
- paper/tables/t4_zeroshot.tex
"""
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, 'results')
TABLES_DIR = os.path.join(ROOT, 'paper', 'tables')


def fmt_stat(mean_val, std_val, digits=2, is_snn=False):
    if pd.isna(mean_val):
        return '-'
    if pd.isna(std_val) or std_val == 0.0:
        return f'{mean_val:.{digits}f}'
    return f'{mean_val:.{digits}f} $\\pm$ {std_val:.{digits}f}'


def generate_t1_main_comparison(mean_df, std_df):
    """Table 1: Main benchmark comparison between ViT baselines and final LeafSpike models."""
    rows = [
        ('DeiT-S', 'deit_s', False),
        ('Swin-T', 'swin_t', False),
        ('MobileViT-XS', 'mobilevit_xs', False),
        ('MobileViT-XXS', 'mobilevit_xxs', False),
        ('LeafSpike-S (Final)', 'leafspike_no_maxshort', True),
        ('LeafSpike-S-sparse ($w=10$)', 'leafspike_final_sparse_w10', True),
    ]

    tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{\textbf{Main comparison on PlantVillage-Tomato (lab test set, duplicate-aware split)}. '
        r'Mean $\pm$ standard deviation across three random seeds. '
        r'Arithmetic and memory energies are 45\,nm model estimates; $E_{\text{mem}}$ (DRAM) fetches the INT8 '
        r'weights from DRAM, while $E_{\text{mem}}$ (on-chip) reads them from SRAM and is a sensitivity bound, not '
        r'a claim that the weights fit on chip. LeafSpike synaptic operations (GSOPs) are marked with an asterisk ($^*$).}',
        r'\label{tab:main_comparison}',
        r'\small',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lccccccccc}',
        r'\toprule',
        r'\textbf{Model} & \textbf{Params} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{MACs/SOPs} & \textbf{$E_{\text{FP32}}$} & \textbf{$E_{\text{INT8}}$} & \textbf{$E_{\text{mem}}$ DRAM (lb--ub)} & \textbf{$E_{\text{mem}}$ on-chip (lb--ub)} & \textbf{Latency} \\',
        r' & \textbf{(M)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(G)} & \textbf{(mJ)} & \textbf{(mJ)} & \textbf{(mJ)} & \textbf{(mJ)} & \textbf{(ms)} \\',
        r'\midrule',
    ]

    for label, m_key, is_snn in rows:
        if m_key not in mean_df.index:
            continue
        m = mean_df.loc[m_key]
        s = std_df.loc[m_key]

        params = f"{m['params_M']:.2f}"
        acc = fmt_stat(m['acc_fp32'], s['acc_fp32'], 2)
        f1 = fmt_stat(m['f1_fp32'], s['f1_fp32'], 2)

        if is_snn:
            ops = f"{m['GSOPs']:.3f}$^*$"
        else:
            ops = f"{m['dense_GMACs']:.3f}"

        e_fp32 = fmt_stat(m['compute_fp32_mJ'], s['compute_fp32_mJ'], 3)
        e_int8 = fmt_stat(m['compute_int8_mJ'], s['compute_int8_mJ'], 4)

        mem_ub = m['mem_int8_mJ']
        mem_lb = m.get('mem_int8_lb_mJ', mem_ub)
        if is_snn and abs(mem_ub - mem_lb) > 1e-4:
            e_mem = f"{mem_lb:.3f}--{mem_ub:.3f}"
        else:
            e_mem = f"{mem_ub:.3f}"

        oc_ub = m['mem_int8_onchip_mJ']
        oc_lb = m.get('mem_int8_onchip_lb_mJ', oc_ub)
        if is_snn and abs(oc_ub - oc_lb) > 1e-4:
            e_mem_oc = f"{oc_lb:.3f}--{oc_ub:.3f}"
        else:
            e_mem_oc = f"{oc_ub:.3f}"

        lat = fmt_stat(m['latency_ms_bs1'], s['latency_ms_bs1'], 1)

        tex.append(f"{label} & {params} & {acc} & {f1} & {ops} & {e_fp32} & {e_int8} & "
                   f"{e_mem} & {e_mem_oc} & {lat} \\\\")

    tex.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}'
    ])
    return '\n'.join(tex) + '\n'


def generate_t2_ablations(mean_df, std_df):
    """Table 2: Ablations with per-row regulariser settings and base model."""
    groups = [
        ('Shortcut', [
            ('With max-pool shortcut', 'leafspike', 'with shortcut'),
            ('Without shortcut (final)', 'leafspike_no_maxshort', 'no shortcut'),
        ]),
        ('Attention mode ($D=4$)', [
            ('Time-aligned (trained aligned)', 'leafspike_no_maxshort', 'no shortcut'),
            ('Integer (trained integer)', 'leafspike_integer_attn', 'with shortcut'),
        ]),
        ('Spike count per neuron ($D$)', [
            ('$D=4$ (counts 0--4, final)', 'leafspike_no_maxshort', 'no shortcut'),
            ('$D=2$ (counts 0--2)', 'leafspike_D2', 'with shortcut'),
            ('$D=1$ (binary)', 'leafspike_D1', 'with shortcut'),
        ]),
        ('Firing regulariser', [
            ('Baseline ($w=0$)', 'leafspike_no_maxshort', 'no shortcut'),
            ('$w=1$, target 0.05, last 30\\%', 'leafspike_firereg', 'with shortcut'),
            ('$w=5$, target 0.02, last 100\\%', 'leafspike_final_sparse_w5', 'no shortcut'),
            ('$w=10$, target 0.02, last 100\\% (final sparse)', 'leafspike_final_sparse_w10', 'no shortcut'),
            ('$w=10$, target 0.02, last 100\\% (1 seed, ablation)', 'leafspike_sparse_w10', 'with shortcut'),
            ('$w=30$, target 0.02, last 100\\%', 'leafspike_sparse_w30', 'with shortcut'),
        ]),
        ('$D=2$ + regulariser', [
            ('$w=1$, target 0.05, last 30\\%', 'leafspike_D2_firereg', 'with shortcut'),
        ]),
    ]

    tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{\textbf{Ablation studies on LeafSpike components and sparsity regularisation}. '
        r'Reported on PlantVillage-Tomato test set (deterministic FP32). The final sparse model ($w=10$, 3 seeds) '
        r'cuts synaptic operations by 39\% and FP32 arithmetic energy by 36\% at 99.50 $\pm$ 0.07\% accuracy. '
        r'The most aggressive run ($w=30$) reaches a 61\% reduction at 97.99\%. Each row states the number of '
        r'seeds, the max-pool shortcut setting and, for regularised runs, the weight, target firing rate and the '
        r'fraction of training over which the penalty is applied.}',
        r'\label{tab:ablations}',
        r'\small',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{llcccccccc}',
        r'\toprule',
        r'\textbf{Configuration} & \textbf{Base} & \textbf{$n$} & \textbf{Accuracy (\%)} & \textbf{Macro-F1 (\%)} & '
        r'\textbf{GSOPs} & \textbf{Spikes/Neuron} & \textbf{$E_{\text{FP32}}$ (mJ)} & '
        r'\textbf{$E_{\text{INT8}}$ (mJ)} & \textbf{Acc. other mode (\%)} \\',
        r'\midrule',
    ]

    for group_name, items in groups:
        group_header = r'\multicolumn{10}{l}{\textit{' + group_name + r'}} \\'
        tex.append(group_header)
        for label, m_key, base in items:
            if m_key not in mean_df.index:
                continue
            m = mean_df.loc[m_key]
            s = std_df.loc[m_key]

            acc = fmt_stat(m['acc_fp32'], s['acc_fp32'], 2)
            f1 = fmt_stat(m['f1_fp32'], s['f1_fp32'], 2)
            gsops = fmt_stat(m['GSOPs'], s['GSOPs'], 3)
            spk = fmt_stat(m['avg_spikes_per_neuron'], s['avg_spikes_per_neuron'], 3)
            e_fp32 = fmt_stat(m['compute_fp32_mJ'], s['compute_fp32_mJ'], 3)
            e_int8 = fmt_stat(m['compute_int8_mJ'], s['compute_int8_mJ'], 4)
            acc_other = fmt_stat(m.get('acc_other_fp32', float('nan')),
                                 s.get('acc_other_fp32', float('nan')), 2)
            n_seeds = int(m['n_seeds']) if not pd.isna(m.get('n_seeds')) else 1

            tex.append(f"\\quad {label} & {base} & {n_seeds} & {acc} & {f1} & {gsops} & {spk} & {e_fp32} & "
                       f"{e_int8} & {acc_other} \\\\")
        tex.append(r'\midrule')

    # Remove last midrule and replace with bottomrule
    if tex[-1] == r'\midrule':
        tex[-1] = r'\bottomrule'
    else:
        tex.append(r'\bottomrule')

    tex.extend([
        r'\end{tabular}}',
        r'\par\smallskip',
        r"{\footnotesize \textbf{Notes.} ``Acc.\ other mode'' is the accuracy when the trained weights are run "
        r"in the other attention mode (deterministic FP32, batch 64), i.e.\ integer weights evaluated as aligned "
        r"spikes, or aligned weights evaluated as integer surrogates. $n$ is the number of training seeds. "
        r"The regulariser default is target 0.05 over the last 30\% of epochs; all other regularised rows use "
        r"target 0.02 over the whole training run. The integer-attention row uses the with-shortcut base; the "
        r"max-pool shortcut has no measurable effect (99.74 $\pm$ 0.15 with it vs 99.73 $\pm$ 0.10 without; "
        r"McNemar pooled 11/12, $p = 1.00$).}",
        r'\end{table}'
    ])
    return '\n'.join(tex) + '\n'


def _significance_rows():
    """Parse significance.md into rows; pooled rows keep raw p only; 1-seed pools are dropped."""
    sig_path = os.path.join(RESULTS_DIR, 'significance.md')
    if not os.path.exists(sig_path):
        return []
    lines = open(sig_path, encoding='utf-8').read().splitlines()
    sig_lines = lines[:next((i for i, l in enumerate(lines) if l.startswith('## PlantDoc')), len(lines))]
    table_lines = [l for l in sig_lines if l.startswith('|') and not l.startswith('|---')]
    rows = []
    for line in table_lines[1:]:
        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) != 8:
            continue
        model, baseline, seed, errs, disc, raw_p, adj_p, sig = parts
        pooled = 'Pooled' in seed
        if pooled and '1 seed' in seed:
            continue
        rows.append(dict(model=model, baseline=baseline, seed=seed.replace('**', ''),
                         comp=f'{model} vs {baseline}', errs=errs, disc=disc,
                         raw_p=raw_p, adj_p=('--' if pooled else adj_p),
                         sig=('--' if pooled else ('Yes' if 'Yes' in sig else 'No')),
                         pooled=pooled))
    return rows


def _write_t3(rows, path, caption, label):
    tex = [
        r'\begin{table}[t]',
        r'\centering',
        caption,
        f'\\label{{{label}}}',
        r'\small',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{llccccc}',
        r'\toprule',
        r'\textbf{Comparison} & \textbf{Seed} & \textbf{Errors} & \textbf{Discordant ($b/c$)} & '
        r'\textbf{Raw $p$} & \textbf{Adj. $p$} & \textbf{Sig. ($\alpha=0.05$)} \\',
        r'\midrule',
    ]
    for r in rows:
        seed_tex = (r'\textbf{' + r['seed'] + '}') if r['pooled'] else r['seed']
        comp_tex = (r'\textbf{' + r['comp'] + '}') if r['pooled'] else r['comp']
        tex.append(f"{comp_tex} & {seed_tex} & {r['errs']} & {r['disc']} & {r['raw_p']} & "
                   f"{r['adj_p']} & {r['sig']} \\\\")
    tex.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\par\smallskip',
        r'{\footnotesize \textbf{Protocol.} Primary inference: exact McNemar test per seed on paired predictions '
        r'(the same 2{,}593 test images), Holm-Bonferroni across the three seeds of each model-vs-baseline family. '
        r'Pooled rows sum discordant counts over seeds and are descriptive only (not independent: the test images '
        r'are reused); for them only the raw $p$ is shown.}',
        r'\end{table}'
    ])
    return '\n'.join(tex) + '\n'


def generate_t3_significance():
    """Table 3 (main): final model and sparse w=10 vs the four ViT baselines."""
    rows = _significance_rows()
    if not rows:
        return '% significance.md not found\n'
    main_models = ('LeafSpike-S (Final, No Shortcut)', 'LeafSpike-S-sparse (w=10)')
    keep = [r for r in rows if r['model'] in main_models]
    caption = (
        r'\caption{\textbf{Exact McNemar tests for the final model and the sparse model} (2,593 test images). '
        r'Per-seed tests are the primary inference: Holm-Bonferroni within each model-vs-baseline family across '
        r'the three seeds. Pooled rows are descriptive only; they show the raw $p$.}'
    )
    return _write_t3(keep, os.path.join(TABLES_DIR, 't3_significance.tex'), caption, 'tab:significance')


def generate_t3b_significance():
    """Table 3b (appendix): shortcut block, shortcut ablation and w=5 comparisons."""
    rows = _significance_rows()
    if not rows:
        return '% significance.md not found\n'
    main_models = ('LeafSpike-S (Final, No Shortcut)', 'LeafSpike-S-sparse (w=10)')
    keep = [r for r in rows if r['model'] not in main_models]
    caption = (
        r'\caption{\textbf{Appendix: exact McNemar tests for the secondary comparisons} (2,593 test images). '
        r'Same protocol as Table~\ref{tab:significance}: per-seed tests are primary, pooled rows are descriptive '
        r'and show the raw $p$.}'
    )
    return _write_t3(keep, os.path.join(TABLES_DIR, 't3b_significance.tex'), caption, 'tab:significance_appendix')


ZERO_LABELS = {
    'deit_s': 'DeiT-S',
    'swin_t': 'Swin-T',
    'mobilevit_xs': 'MobileViT-XS',
    'mobilevit_xxs': 'MobileViT-XXS',
    'leafspike': 'LeafSpike (with shortcut)',
    'leafspike_no_maxshort': 'LeafSpike-S (final)',
    'leafspike_final_sparse_w10': 'LeafSpike-S-sparse ($w=10$)',
    'leafspike_final_sparse_w5': 'LeafSpike-S-sparse ($w=5$)',
    'leafspike_integer_attn': 'LS (integer attention)',
    'leafspike_D2': 'LS ($D=2$)',
    'leafspike_D1': 'LS ($D=1$)',
    'leafspike_firereg': 'LS (reg $w=1$)',
    'leafspike_D2_firereg': 'LS ($D=2$ + reg $w=1$)',
    'leafspike_sparse_w10': 'LS (with shortcut, $w=10$)',
    'leafspike_sparse_w30': 'LS (with shortcut, $w=30$)',
}


def generate_t4_zeroshot():
    """Table 4: PlantDoc zero-shot robustness evaluation (read from zeroshot.csv)."""
    zs_path = os.path.join(RESULTS_DIR, 'zeroshot.csv')
    if not os.path.exists(zs_path):
        return "% zeroshot.csv not found\n"

    df = pd.read_csv(zs_path)
    g = df.groupby('model', sort=False)
    mean, std = g[['field_acc', 'field_f1']].mean(), g[['field_acc', 'field_f1']].std()

    tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{\textbf{Zero-shot cross-domain evaluation on the PlantDoc field dataset} (903 field images across 9 shared tomato disease classes, without fine-tuning). '
        r'Predictions are restricted to the shared tomato classes.}',
        r'\label{tab:zeroshot}',
        r'\small',
        r'\begin{tabular}{lcc}',
        r'\toprule',
        r'\textbf{Model} & \textbf{Field Accuracy (\%)} & \textbf{Field Macro-F1 (\%)} \\',
        r'\midrule',
    ]

    for model in mean.index:
        label = ZERO_LABELS.get(model, model)
        acc = fmt_stat(mean.loc[model, 'field_acc'], std.loc[model, 'field_acc'], 2)
        f1 = fmt_stat(mean.loc[model, 'field_f1'], std.loc[model, 'field_f1'], 2)
        tex.append(f"{label} & {acc} & {f1} \\\\")

    tex.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])
    return '\n'.join(tex) + '\n'


def _int_or_dash(v):
    return '-' if pd.isna(v) else str(int(v))


def generate_t5_stability():
    """Table 5 (appendix): per-job prediction stability (fp16 vs FP32, determinism, batch size)."""
    stab_path = os.path.join(RESULTS_DIR, 'stability.csv')
    if not os.path.exists(stab_path):
        return '% stability.csv not found\n'
    df = pd.read_csv(stab_path)

    tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{\textbf{Prediction stability per job.} \emph{FP16$\to$FP32} counts per-image predictions '
        r'that change between the original fp16 training-time evaluation and the deterministic FP32 evaluation; '
        r'\emph{run 1 vs 2} repeats the deterministic evaluation; \emph{bs 64 vs 1} compares batch sizes. '
        r'The last column is the batch-1 accuracy (deterministic FP32).}',
        r'\label{tab:stability}',
        r'\footnotesize',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'\textbf{Job} & \textbf{FP16$\to$FP32 flips} & \textbf{FP32 run 1 vs 2} & \textbf{bs 64 vs 1} & \textbf{FP32 errors} & \textbf{Acc.\ bs1 (\%)} \\',
        r'\midrule',
    ]
    for _, r in df.iterrows():
        acc_bs1 = '-' if pd.isna(r.get('acc_bs1')) else f"{r['acc_bs1']:.2f}"
        job_tex = str(r['job']).replace('_', r'\_')
        tex.append(f"\\texttt{{{job_tex}}} & {_int_or_dash(r.get('flips_fp16'))} & "
                   f"{_int_or_dash(r.get('flips_run1'))} & {_int_or_dash(r.get('flips_bs1'))} & "
                   f"{_int_or_dash(r.get('errors_fp32'))} & {acc_bs1} \\\\")
    tex.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}'
    ])
    return '\n'.join(tex) + '\n'


def write_preview():
    """Minimal document that inputs T1-T5, used for a compile check."""
    tex = '\n'.join([
        r'\documentclass[10pt]{article}',
        r'\usepackage[margin=1in]{geometry}',
        r'\usepackage{booktabs}',
        r'\usepackage{graphicx}',
        r'\usepackage{amsmath}',
        r'\begin{document}',
        r'\section*{Table preview}',
    ] + [f'\\input{{{name}}}' + '\n\\clearpage' for name in
         ('t1_main_comparison', 't2_ablations', 't3_significance', 't3b_significance',
          't4_zeroshot', 't5_stability')] + [
        r'\section*{Figure}',
        r'\begin{figure}[h]',
        r'\centering',
        r'\includegraphics[width=\linewidth]{../figures/accuracy_vs_energy.pdf}',
        r'\caption{Accuracy vs energy. Coloured markers are the T1 models; grey markers are the ablation runs '
        r'(see Table~\ref{tab:ablations}). (a) FP32 arithmetic energy; (b) INT8 arithmetic energy; '
        r'(c) INT8 compute + memory with DRAM weights; (d) with on-chip weights. In (c) and (d) the bar spans '
        r'the event-driven lower bound to the pessimistic upper bound. LS ($D=1$, 28.7\%) is excluded from '
        r'the axes.}',
        r'\end{figure}',
        r'\end{document}',
    ]) + '\n'
    path = os.path.join(TABLES_DIR, 'preview.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(tex)
    return path


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    mean_path = os.path.join(RESULTS_DIR, 'summary_mean.csv')
    std_path = os.path.join(RESULTS_DIR, 'summary_std.csv')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing summary CSVs in {RESULTS_DIR}")

    mean_df = pd.read_csv(mean_path, index_col='model')
    std_df = pd.read_csv(std_path, index_col='model')

    t1_tex = generate_t1_main_comparison(mean_df, std_df)
    with open(os.path.join(TABLES_DIR, 't1_main_comparison.tex'), 'w', encoding='utf-8') as f:
        f.write(t1_tex)
    print("Wrote paper/tables/t1_main_comparison.tex")

    t2_tex = generate_t2_ablations(mean_df, std_df)
    with open(os.path.join(TABLES_DIR, 't2_ablations.tex'), 'w', encoding='utf-8') as f:
        f.write(t2_tex)
    print("Wrote paper/tables/t2_ablations.tex")

    t3_tex = generate_t3_significance()
    with open(os.path.join(TABLES_DIR, 't3_significance.tex'), 'w', encoding='utf-8') as f:
        f.write(t3_tex)
    print("Wrote paper/tables/t3_significance.tex")

    t3b_tex = generate_t3b_significance()
    with open(os.path.join(TABLES_DIR, 't3b_significance.tex'), 'w', encoding='utf-8') as f:
        f.write(t3b_tex)
    print("Wrote paper/tables/t3b_significance.tex")

    t4_tex = generate_t4_zeroshot()
    with open(os.path.join(TABLES_DIR, 't4_zeroshot.tex'), 'w', encoding='utf-8') as f:
        f.write(t4_tex)
    print("Wrote paper/tables/t4_zeroshot.tex")

    t5_tex = generate_t5_stability()
    with open(os.path.join(TABLES_DIR, 't5_stability.tex'), 'w', encoding='utf-8') as f:
        f.write(t5_tex)
    print("Wrote paper/tables/t5_stability.tex")

    print("Wrote", write_preview())


if __name__ == '__main__':
    main()
