"""Descriptive mode-sensitivity numbers from the saved profile layer statistics.

For q/k/v spike layers, report the mean spike count among firing neurons
(mean_spikes_per_neuron / frac_nonzero). No retraining, no claims.

    python experiments/mode_sensitivity.py [--runs_dir results]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def per_job(path):
    result = json.load(open(path, encoding='utf-8'))
    layers = result.get('profile', {}).get('layers', [])
    out = {}
    for kind in ('q', 'k', 'v'):
        vals = [l['mean_spikes_per_neuron'] / l['frac_nonzero']
                for l in layers if l['layer'].endswith(f'.{kind}_spike') and l['frac_nonzero'] > 0]
        out[kind] = float(np.mean(vals)) if vals else None
    vals = [out[k] for k in ('q', 'k', 'v') if out[k] is not None]
    out['all'] = float(np.mean(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'mode_sensitivity.md'))
    args = ap.parse_args()

    jobs = []
    for model in ('leafspike_no_maxshort', 'leafspike_final_sparse_w10'):
        for seed in (42, 43, 44):
            jid = f'{model}_s{seed}'
            path = os.path.join(args.runs_dir, 'runs', jid, 'result.json')
            if os.path.exists(path):
                jobs.append((model, seed, jid, path))

    rows = []
    for model, seed, jid, path in jobs:
        vals = per_job(path)
        rows.append(dict(model=model, seed=seed, **vals))
    df = pd.DataFrame(rows)

    lines = [
        '# Mode sensitivity: spike counts among firing neurons',
        '',
        'Descriptive only, computed from the saved `result.json` layer statistics (no retraining).',
        'For every q/k/v spike layer, the mean spike count among **firing** neurons is '
        '`mean_spikes_per_neuron / frac_nonzero`; the value for a model is the unweighted mean over its '
        'q/k/v layers, then over seeds.',
        '',
        '| Model | Seed | q | k | v | All |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| `{r['model']}` | {r['seed']} | {r['q']:.3f} | {r['k']:.3f} | {r['v']:.3f} | "
                     f"{r['all']:.3f} |")
    lines += ['', '| Model | q (mean ± std) | k | v | All |', '|---|---:|---:|---:|---:|']
    for model, g in df.groupby('model', sort=False):
        def ms(col):
            return f"{g[col].mean():.3f} ± {g[col].std():.3f}" if len(g) > 1 else f"{g[col].mean():.3f}"
        lines.append(f"| `{model}` | {ms('q')} | {ms('k')} | {ms('v')} | {ms('all')} |")

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out} ({len(df)} jobs)')


if __name__ == '__main__':
    main()
