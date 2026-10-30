"""Rebuild results/stability.md from the saved per-image prediction files.

Reports, per job:
1. FP32 run 1 vs run 2 (determinism of the deterministic evaluation)
2. FP32 batch 64 vs batch 1 (batch-size sensitivity, LeafSpike runs only)
3. FP32 vs the original fp16 training-time evaluation (precision stability)

    python experiments/compare_runs.py [--runs_dir results] [--out results/stability.md]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'stability.md'))
    args = ap.parse_args()

    runs_dir = args.runs_dir
    rows = []
    for d in sorted(glob.glob(os.path.join(runs_dir, 'runs', '*'))):
        if not os.path.isdir(d):
            continue
        jid = os.path.basename(d)
        fp32 = _read(os.path.join(d, 'test_predictions_fp32.csv'))
        if fp32 is None:
            continue
        lab = fp32['label'].to_numpy()
        pred = fp32['pred'].to_numpy()
        row = {'job': jid, 'errors_fp32': int(np.sum(pred != lab)),
               'acc_fp32': float(np.mean(pred == lab) * 100)}
        for name, col in [('test_predictions.csv', 'fp16'),
                          ('test_predictions_fp32_run1.csv', 'run1'),
                          ('test_predictions_bs1.csv', 'bs1')]:
            df = _read(os.path.join(d, name))
            if df is None:
                continue
            other = df['pred'].to_numpy()
            row[f'errors_{col}'] = int(np.sum(other != lab))
            row[f'acc_{col}'] = float(np.mean(other == lab) * 100)
            row[f'flips_{col}'] = int(np.sum(pred != other))
        rows.append(row)

    if not rows:
        raise SystemExit(f'no prediction files found under {runs_dir}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(args.out), 'stability.csv'), index=False)

    fp16 = df.dropna(subset=['flips_fp16'])
    run1 = df.dropna(subset=['flips_run1'])
    bs1 = df.dropna(subset=['flips_bs1'])

    lines = [
        '# Prediction Stability Report',
        '',
        'All accuracies in the paper come from the deterministic FP32 evaluation '
        '(`test_predictions_fp32.csv`). This report quantifies how stable those predictions are.',
        '',
        '## 1. Determinism: FP32 batch 64, run 1 vs run 2',
        '',
        f'- Jobs evaluated twice: **{len(run1)}**',
        f'- Jobs with any changed prediction: **{int((run1["flips_run1"] > 0).sum())}**',
        f'- Total changed predictions: **{int(run1["flips_run1"].sum())}**',
        '',
        '## 2. Batch-size sensitivity: batch 64 vs batch 1',
        '',
        f'- Jobs tested: **{len(bs1)}** ({", ".join(bs1["job"])})',
    ]
    for _, r in bs1.iterrows():
        lines.append(f'  - `{r["job"]}`: {int(r["flips_bs1"])} predictions change, '
                     f'errors {int(r["errors_fp32"])} (bs64, {r["acc_fp32"]:.2f}%) -> '
                     f'{int(r["errors_bs1"])} (bs1, {r["acc_bs1"]:.2f}%)')
    lines += [
        '',
        '## 3. Precision: FP32 vs fp16 training-time evaluation',
        '',
        f'- Jobs with both files: **{len(fp16)}**',
        f'- Total changed predictions: **{int(fp16["flips_fp16"].sum())}**',
        f'- Jobs with more than 2 changes: **{int((fp16["flips_fp16"] > 2).sum())}**',
        '',
        '| Job | FP32 errors | fp16 errors | Changed predictions |',
        '|---|---|---|---|',
    ]
    for _, r in fp16.sort_values('flips_fp16', ascending=False).iterrows():
        lines.append(f'| `{r["job"]}` | {int(r["errors_fp32"])} | {int(r["errors_fp16"])} | '
                     f'{int(r["flips_fp16"])} |')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out} ({len(df)} jobs)')


if __name__ == '__main__':
    main()
