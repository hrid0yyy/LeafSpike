"""Evaluate saved checkpoints (best.pth) on the lab test set and optional field set.

Performs:
1. Deterministic FP32 / FP16 per-image predictions saved to runs/<job>/test_predictions[_fp32].csv.
2. Comparison and flip counts:
   (a) FP32 run 1 vs run 2
   (b) FP32 batch 64 vs batch 1
   (c) FP32 vs old FP16
3. Exact McNemar test (scipy.stats.binomtest) with Holm-Bonferroni correction and pooled seed tests.
4. Summary of significance results to runs/report/significance.md.
5. Optional zero-shot field evaluation on PlantDoc (--field <path>) with restricted argmax.
"""
import argparse
import json
import os
import sys

# Set CUBLAS deterministic workspace config before CUDA initialises
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lsx import data as D
from lsx.jobs import build_jobs
from lsx.train import build_model, metrics, read_json
from run_all import load_cfg

try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as _binom_test

    def binomtest(k, n, p=0.5, alternative='two-sided'):
        class Res:
            def __init__(self, pv):
                self.pvalue = pv
        return Res(_binom_test(k, n, p, alternative=alternative))


def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        sd = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_dataset(model, loader, device, fp32=False, allowed_classes=None):
    model.eval()
    all_preds, all_labels = [], []
    use_amp = (device.type == 'cuda' and not fp32)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
                out = model(x)
        else:
            out = model(x)
        if allowed_classes is not None:
            mask = torch.full_like(out, -float('inf'))
            mask[:, allowed_classes] = 0.0
            out = out + mask
        p = out.argmax(dim=1).cpu().numpy()
        all_preds.append(p)
        all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return preds, labels


def run_mcnemar_test(preds_a, preds_b, targets):
    """Exact McNemar test on discordant pairs using binomial test (two-sided, p=0.5)."""
    correct_a = (preds_a == targets)
    correct_b = (preds_b == targets)

    # b: A correct, B wrong
    b = int(np.sum(correct_a & (~correct_b)))
    # c: A wrong, B correct
    c = int(np.sum((~correct_a) & correct_b))

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        res = binomtest(k=b, n=n_discordant, p=0.5, alternative='two-sided')
        p_value = float(res.pvalue)

    err_a = int(np.sum(~correct_a))
    err_b = int(np.sum(~correct_b))
    return {
        'err_a': err_a,
        'err_b': err_b,
        'b_a_better': b,
        'c_b_better': c,
        'n_discordant': n_discordant,
        'p_value': p_value
    }


def holm_bonferroni(p_values):
    """Compute Holm-Bonferroni step-down adjusted p-values for a family of tests."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        raw_adj = p_values[idx] * factor
        running_max = max(running_max, raw_adj)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def run_significance_analysis(predictions_dict, targets, seeds=(42, 43, 44)):
    """Run family-wise Holm-Bonferroni and pooled McNemar tests."""
    baselines = [
        ('DeiT-S', 'deit_s'),
        ('Swin-T', 'swin_t'),
        ('MobileViT-XS', 'mobilevit_xs'),
        ('MobileViT-XXS', 'mobilevit_xxs'),
    ]

    main_model = ('LeafSpike-S (Final, No Shortcut)', 'leafspike_no_maxshort')
    secondary_model = ('LeafSpike-S (With Shortcut)', 'leafspike')

    results = []

    for model_label, model_prefix in [main_model, secondary_model]:
        for base_label, base_prefix in baselines:
            family_tests = []
            for seed in seeds:
                m_id = f'{model_prefix}_s{seed}'
                b_id = f'{base_prefix}_s{seed}'
                if m_id in predictions_dict and b_id in predictions_dict:
                    stats = run_mcnemar_test(predictions_dict[m_id], predictions_dict[b_id], targets)
                    family_tests.append({
                        'model_label': model_label,
                        'base_label': base_label,
                        'seed': seed,
                        'err_a': stats['err_a'],
                        'err_b': stats['err_b'],
                        'b': stats['b_a_better'],
                        'c': stats['c_b_better'],
                        'n_disc': stats['n_discordant'],
                        'raw_p': stats['p_value']
                    })

            if family_tests:
                raw_ps = [t['raw_p'] for t in family_tests]
                adj_ps = holm_bonferroni(raw_ps)
                for t, adj in zip(family_tests, adj_ps):
                    t['adj_p'] = adj
                    t['sig'] = 'Yes (p<0.05)' if adj < 0.05 else 'No'
                    results.append(t)

                # Pooled test across seeds
                total_b = sum(t['b'] for t in family_tests)
                total_c = sum(t['c'] for t in family_tests)
                total_disc = total_b + total_c
                if total_disc == 0:
                    pooled_p = 1.0
                else:
                    pooled_p = float(binomtest(k=total_b, n=total_disc, p=0.5, alternative='two-sided').pvalue)
                results.append({
                    'model_label': model_label,
                    'base_label': base_label,
                    'seed': 'Pooled (3 seeds, descriptive)',
                    'err_a': sum(t['err_a'] for t in family_tests),
                    'err_b': sum(t['err_b'] for t in family_tests),
                    'b': total_b,
                    'c': total_c,
                    'n_disc': total_disc,
                    'raw_p': pooled_p,
                    'adj_p': pooled_p,
                    'sig': 'Yes (p<0.05)' if pooled_p < 0.05 else 'No',
                    'is_pooled': True
                })

    # Direct comparison: LeafSpike-S (No Shortcut) vs LeafSpike (With Shortcut)
    shortcut_family = []
    for seed in seeds:
        m1_id = f'leafspike_no_maxshort_s{seed}'
        m2_id = f'leafspike_s{seed}'
        if m1_id in predictions_dict and m2_id in predictions_dict:
            stats = run_mcnemar_test(predictions_dict[m1_id], predictions_dict[m2_id], targets)
            shortcut_family.append({
                'model_label': 'LeafSpike-S (No Shortcut)',
                'base_label': 'LeafSpike (With Shortcut)',
                'seed': seed,
                'err_a': stats['err_a'],
                'err_b': stats['err_b'],
                'b': stats['b_a_better'],
                'c': stats['c_b_better'],
                'n_disc': stats['n_discordant'],
                'raw_p': stats['p_value']
            })
    if shortcut_family:
        adj_ps = holm_bonferroni([t['raw_p'] for t in shortcut_family])
        for t, adj in zip(shortcut_family, adj_ps):
            t['adj_p'] = adj
            t['sig'] = 'Yes (p<0.05)' if adj < 0.05 else 'No'
            results.append(t)
        total_b = sum(t['b'] for t in shortcut_family)
        total_c = sum(t['c'] for t in shortcut_family)
        total_disc = total_b + total_c
        pooled_p = float(binomtest(k=total_b, n=total_disc, p=0.5, alternative='two-sided').pvalue) if total_disc > 0 else 1.0
        results.append({
            'model_label': 'LeafSpike-S (No Shortcut)',
            'base_label': 'LeafSpike (With Shortcut)',
            'seed': 'Pooled (3 seeds, descriptive)',
            'err_a': sum(t['err_a'] for t in shortcut_family),
            'err_b': sum(t['err_b'] for t in shortcut_family),
            'b': total_b,
            'c': total_c,
            'n_disc': total_disc,
            'raw_p': pooled_p,
            'adj_p': pooled_p,
            'sig': 'Yes (p<0.05)' if pooled_p < 0.05 else 'No',
            'is_pooled': True
        })

    # Phase E sparse comparisons against lightweight ViTs
    sparse_comparisons = [
        ('LeafSpike-S-sparse (w=10)', 'leafspike_final_sparse_w10'),
        ('LeafSpike-S-sparse (w=5)', 'leafspike_final_sparse_w5'),
    ]
    for sp_label, sp_prefix in sparse_comparisons:
        for base_label, base_prefix in [('MobileViT-XXS', 'mobilevit_xxs'), ('MobileViT-XS', 'mobilevit_xs')]:
            sp_family = []
            for seed in seeds:
                sp_id = f'{sp_prefix}_s{seed}'
                b_id = f'{base_prefix}_s{seed}'
                if sp_id in predictions_dict and b_id in predictions_dict:
                    stats = run_mcnemar_test(predictions_dict[sp_id], predictions_dict[b_id], targets)
                    sp_family.append({
                        'model_label': sp_label,
                        'base_label': base_label,
                        'seed': seed,
                        'err_a': stats['err_a'],
                        'err_b': stats['err_b'],
                        'b': stats['b_a_better'],
                        'c': stats['c_b_better'],
                        'n_disc': stats['n_discordant'],
                        'raw_p': stats['p_value']
                    })
            if sp_family:
                adj_ps = holm_bonferroni([t['raw_p'] for t in sp_family])
                for t, adj in zip(sp_family, adj_ps):
                    t['adj_p'] = adj
                    t['sig'] = 'Yes (p<0.05)' if adj < 0.05 else 'No'
                    results.append(t)
                total_b = sum(t['b'] for t in sp_family)
                total_c = sum(t['c'] for t in sp_family)
                total_disc = total_b + total_c
                pooled_p = float(binomtest(k=total_b, n=total_disc, p=0.5, alternative='two-sided').pvalue) if total_disc > 0 else 1.0
                results.append({
                    'model_label': sp_label,
                    'base_label': base_label,
                    'seed': f"Pooled ({len(sp_family)} seed{'s' if len(sp_family) > 1 else ''}, descriptive)",
                    'err_a': sum(t['err_a'] for t in sp_family),
                    'err_b': sum(t['err_b'] for t in sp_family),
                    'b': total_b,
                    'c': total_c,
                    'n_disc': total_disc,
                    'raw_p': pooled_p,
                    'adj_p': pooled_p,
                    'sig': 'Yes (p<0.05)' if pooled_p < 0.05 else 'No',
                    'is_pooled': True
                })

    return results


BUILD_KEYS = ('name', 'kind', 'arch', 'size', 'D', 'attn_mode', 'maxpool_shortcut')


def sync_jobs_from_results(runs_dir, jobs):
    """Use the job dict stored in each run's result.json as the single source of truth for evaluation.

    Falls back to the registry entry (and, for SNN runs without a stored job, to the presence of
    `lab_test_as_<other>_attention`) when result.json is missing. Warns when a key differs.
    """
    for i, job in enumerate(jobs):
        jdir = None
        for cand in (os.path.join(runs_dir, job['id']), os.path.join(runs_dir, 'runs', job['id'])):
            if os.path.exists(cand):
                jdir = cand
                break
        meta = read_json(os.path.join(jdir, 'result.json')) if jdir else None
        saved = (meta or {}).get('job')
        if saved:
            diff = {k: (job.get(k), saved.get(k)) for k in BUILD_KEYS
                    if k in saved and job.get(k) != saved.get(k)}
            if diff:
                print(f'[warn] {job["id"]}: result.json differs from jobs.py -> {diff}; using result.json')
            jobs[i] = {**job, **saved}
        elif job.get('kind') == 'snn' and meta:
            if 'lab_test_as_aligned_attention' in meta:
                job['attn_mode'] = 'integer'
            elif 'lab_test_as_integer_attention' in meta:
                job['attn_mode'] = 'aligned'


def load_saved_predictions(runs_dir, jobs, targets, filename='test_predictions_fp32.csv'):
    """Load per-image predictions already on disk for every job (used by the significance table)."""
    preds = {}
    for job in jobs:
        jid = job['id']
        for jdir in (os.path.join(runs_dir, jid), os.path.join(runs_dir, 'runs', jid)):
            path = os.path.join(jdir, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                labels = df['label'].to_numpy()
                if len(labels) == len(targets) and np.array_equal(labels, targets):
                    preds[jid] = df['pred'].to_numpy()
                break
    return preds


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved checkpoints and run McNemar significance tests')
    parser.add_argument('--config', default=None, help='Path to config.json')
    parser.add_argument('--runs_dir', default=None, help='Path to runs folder')
    parser.add_argument('--field', default=None, help='Path to PlantDoc dataset root for zero-shot eval')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 inference without autocast')
    parser.add_argument('--deterministic', action='store_true', help='Enforce deterministic PyTorch / cuDNN algorithms')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--save_suffix', default=None, help='Suffix for prediction CSV (e.g. _fp32 or _run1)')
    parser.add_argument('--compare_fp16', action='store_true', help='Compare predictions against existing test_predictions.csv')
    parser.add_argument('--only_jobs', default=None, help='Comma-separated list of job IDs to evaluate')
    parser.add_argument('--significance_only', action='store_true',
                        help='Skip inference; rebuild significance.md from the FP32 prediction files on disk')
    parser.add_argument('--other_attn', action='store_true',
                        help='Also evaluate every SNN job in its other attention mode (FP32)')
    parser.add_argument('--bs1_jobs', default=None,
                        help='Comma-separated job IDs to additionally evaluate at batch size 1')
    parser.add_argument('--field_only', action='store_true',
                        help='Skip the lab-test pass; only run the zero-shot field evaluation')
    args = parser.parse_args()
    bs1_jobs = set(args.bs1_jobs.split(',')) if args.bs1_jobs else set()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("Deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 enabled.")

    class DummyArgs:
        config = args.config
        smoke = False

    cfg = load_cfg(DummyArgs())
    if args.runs_dir:
        cfg['runs_dir'] = args.runs_dir
    runs_dir = cfg['runs_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating saved models using device: {device} (FP32={args.fp32}, batch_size={args.batch_size})')

    # Load split.csv for lab test
    split_path = os.path.join(runs_dir, 'split.csv')
    if not os.path.exists(split_path):
        split_path = os.path.join(os.path.dirname(runs_dir), 'split.csv')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'split.csv not found in {runs_dir}')

    split_df = pd.read_csv(split_path)
    split_df['path'] = [os.path.join(cfg['lab_root'], p) for p in split_df['relpath']]
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
    targets = test_df['label'].to_numpy()
    print(f'Lab test set: {len(test_df)} images across {len(np.unique(targets))} classes')

    jobs = build_jobs(cfg)
    if args.only_jobs:
        target_ids = set(args.only_jobs.split(','))
        jobs = [j for j in jobs if j['id'] in target_ids or j['name'] in target_ids]
    sync_jobs_from_results(runs_dir, jobs)
    if args.significance_only:
        jobs = []
    lab_jobs = [] if args.field_only else jobs

    # Disable pretrained download when building architectures for evaluation
    eval_cfg = dict(cfg)
    eval_cfg['sdt_checkpoint'] = None
    eval_cfg['pretrained_vits'] = False

    job_predictions = {}
    flip_report = []
    eval_rows = []
    out_dir = os.path.join(runs_dir, 'report') if not os.path.exists(os.path.join(runs_dir, 'summary.md')) else runs_dir
    os.makedirs(out_dir, exist_ok=True)

    pred_filename = 'test_predictions_fp32.csv' if args.fp32 else 'test_predictions.csv'
    if args.save_suffix:
        pred_filename = f'test_predictions{args.save_suffix}.csv'

    for job in lab_jobs:
        jid = job['id']
        jdir1 = os.path.join(runs_dir, jid)
        jdir2 = os.path.join(runs_dir, 'runs', jid)
        jdir = jdir1 if os.path.exists(jdir1) else jdir2
        best_pth = os.path.join(jdir, 'best.pth')
        if not os.path.exists(best_pth):
            continue

        print(f'Evaluating {jid} ...')
        model, _, mean, std, _, _ = build_model(job, eval_cfg)
        model = load_checkpoint_weights(model, best_pth, device)
        if hasattr(model, 'set_attn_mode'):
            model.set_attn_mode(job.get('attn_mode', 'aligned'))

        eval_tf = D.make_eval_tf(cfg['img_size'], mean, std)
        test_ds = D.LeafDataset(test_df, eval_tf)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda'))

        preds, _ = predict_dataset(model, test_loader, device, fp32=args.fp32)
        job_predictions[jid] = preds

        # Save per-image predictions
        pred_df = pd.DataFrame({
            'relpath': test_df['relpath'],
            'label': targets,
            'pred': preds
        })
        pred_csv = os.path.join(jdir, pred_filename)
        pred_df.to_csv(pred_csv, index=False)

        eval_errors = int(np.sum(preds != targets))
        eval_acc = float((len(targets) - eval_errors) / len(targets) * 100)
        from sklearn.metrics import f1_score
        eval_f1 = float(f1_score(targets, preds, average='macro', zero_division=0) * 100)

        # Optional: batch-1 accuracy for the same weights (deterministic FP32)
        if jid in bs1_jobs:
            bs1_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda'))
            bs1_preds, _ = predict_dataset(model, bs1_loader, device, fp32=args.fp32)
            pd.DataFrame({'relpath': test_df['relpath'], 'label': targets, 'pred': bs1_preds}) \
                .to_csv(os.path.join(jdir, 'test_predictions_bs1.csv'), index=False)
            print(f'   batch-1 predictions saved ({int(np.sum(bs1_preds != targets))} errors)')

        # Optional: this checkpoint in its other attention mode (FP32)
        other_acc = other_f1 = other_mode = None
        if args.other_attn and hasattr(model, 'set_attn_mode'):
            other_mode = 'integer' if job.get('attn_mode', 'aligned') == 'aligned' else 'aligned'
            model.set_attn_mode(other_mode)
            other_preds, _ = predict_dataset(model, test_loader, device, fp32=args.fp32)
            model.set_attn_mode(job.get('attn_mode', 'aligned'))
            pd.DataFrame({'relpath': test_df['relpath'], 'label': targets, 'pred': other_preds}) \
                .to_csv(os.path.join(jdir, 'test_predictions_fp32_othermode.csv'), index=False)
            other_acc = float(np.mean(other_preds == targets) * 100)
            other_f1 = float(f1_score(targets, other_preds, average='macro', zero_division=0) * 100)

        def _flips(name):
            p = os.path.join(jdir, name)
            if not os.path.exists(p):
                return None, None
            other = pd.read_csv(p)['pred'].to_numpy()
            return int(np.sum(preds != other)), int(np.sum(other != targets))

        flips_fp16, fp16_errors = _flips('test_predictions.csv')
        flips_r1_r2, _ = _flips('test_predictions_fp32_run1.csv')
        flips_bs1, bs1_errors = _flips('test_predictions_bs1.csv')

        result = read_json(os.path.join(jdir, 'result.json'))
        lab_ref = (result or {}).get('lab_test') or {}
        eval_rows.append({
            'job': jid,
            'name': job['name'],
            'seed': job['seed'],
            'acc_ref': float(lab_ref['acc'] * 100) if 'acc' in lab_ref else None,
            'f1_ref': float(lab_ref['macro_f1'] * 100) if 'macro_f1' in lab_ref else None,
            'acc_fp32': eval_acc,
            'f1_fp32': eval_f1,
            'fp32_errors': eval_errors,
            'fp16_errors': fp16_errors,
            'flips_fp16': flips_fp16,
            'flips_run1_run2': flips_r1_r2,
            'bs1_errors': bs1_errors,
            'flips_bs1': flips_bs1,
            'other_mode': other_mode,
            'acc_other_fp32': other_acc,
            'f1_other_fp32': other_f1,
        })

        # Kept for the console report
        old_pred_csv = os.path.join(jdir, 'test_predictions.csv')
        if args.compare_fp16 and flips_fp16 is not None:
            flip_report.append({
                'job': jid,
                'name': job['name'],
                'seed': job['seed'],
                'fp32_errors': eval_errors,
                'fp16_errors': fp16_errors,
                'flips': flips_fp16,
                'acc_fp32': eval_acc
            })

    if flip_report:
        print("\n=== Prediction Stability & Flip Report (FP32 vs FP16) ===")
        print(f"{'Job ID':<30} | {'FP32 Err':<9} | {'FP16 Err':<9} | {'Flips':<6} | {'FP32 Acc (%)':<12}")
        print("-" * 75)
        for r in flip_report:
            print(f"{r['job']:<30} | {r['fp32_errors']:<9} | {r['fp16_errors']:<9} | {r['flips']:<6} | {r['acc_fp32']:<12.2f}")

    # Save per-job FP32 metrics (acc/f1 and flip counts) for the summary tables
    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(os.path.join(out_dir, 'eval_metrics.csv'), index=False)
        print(f'Wrote per-job metrics to {os.path.join(out_dir, "eval_metrics.csv")}')

    # Run Significance Analysis on every FP32 prediction file on disk, so that
    # single-job invocations cannot truncate the table.
    saved_preds = load_saved_predictions(runs_dir, build_jobs(cfg), targets, pred_filename)
    if saved_preds:
        job_predictions = saved_preds
    sig_results = run_significance_analysis(job_predictions, targets)

    sig_path = os.path.join(out_dir, 'significance.md')

    lines = [
        '# Statistical Significance Analysis (Exact McNemar Test with Holm-Bonferroni Correction)',
        '',
        f'Evaluated on the exact PlantVillage-tomato lab test split ({len(test_df)} images).',
        'Tests exact two-sided binomial test on discordant pairs ($b$ = LeafSpike correct & other wrong, $c$ = LeafSpike wrong & other correct).',
        'Primary inference: exact McNemar test per seed, with Holm-Bonferroni correction within each model-pair family (3 seeds).',
        'Pooled rows are **descriptive only, not independent** (the same test images are reused across seeds); treat them as a summary, not a second test.',
        '',
        '| Model | Baseline | Seed | Errors (SNN/ViT) | Discordant (b/c) | Raw p-value | Holm-Adj p-value | Significant (α=0.05) |',
        '|---|---|---|---|---|---|---|---|'
    ]
    for r in sig_results:
        raw_p_str = f"{r['raw_p']:.4f}" if r['raw_p'] >= 0.0001 else f"{r['raw_p']:.2e}"
        adj_p_str = f"{r['adj_p']:.4f}" if r['adj_p'] >= 0.0001 else f"{r['adj_p']:.2e}"
        err_str = f"{r['err_a']}/{r['err_b']}"
        disc_str = f"{r['b']}/{r['c']}"
        seed_label = f"**{r['seed']}**" if r.get('is_pooled') else str(r['seed'])
        lines.append(f"| {r['model_label']} | {r['base_label']} | {seed_label} | {err_str} | {disc_str} | {raw_p_str} | {adj_p_str} | {r['sig']} |")

    # Optional Field Evaluation (PlantDoc)
    if args.field:
        field_root = args.field
        print(f'\nRunning zero-shot field evaluation on {field_root} (FP32={args.fp32}) ...')
        field_df = D.load_field(field_root)
        if field_df is not None and len(field_df) > 0:
            shared_classes = sorted(field_df['label'].unique().tolist())
            shared_names = [D.CLASSES[i] for i in shared_classes]
            counts = {D.CLASSES[c]: int(np.sum(field_df['label'] == c)) for c in shared_classes}
            print(f'Found {len(field_df)} field images across {len(shared_classes)} shared classes:')
            for name, cnt in counts.items():
                print(f'  - {name}: {cnt} images')

            field_results = []
            for job in jobs:
                jid = job['id']
                jdir1 = os.path.join(runs_dir, jid)
                jdir2 = os.path.join(runs_dir, 'runs', jid)
                jdir = jdir1 if os.path.exists(jdir1) else jdir2
                best_pth = os.path.join(jdir, 'best.pth')
                if not os.path.exists(best_pth):
                    continue

                model, _, mean, std, _, _ = build_model(job, eval_cfg)
                model = load_checkpoint_weights(model, best_pth, device)
                if hasattr(model, 'set_attn_mode'):
                    model.set_attn_mode(job.get('attn_mode', 'aligned'))

                eval_tf = D.make_eval_tf(cfg['img_size'], mean, std)
                f_ds = D.LeafDataset(field_df, eval_tf)
                f_loader = DataLoader(f_ds, batch_size=args.batch_size, shuffle=False,
                                      num_workers=cfg['num_workers'], pin_memory=(device.type == 'cuda'))

                f_preds, f_labels = predict_dataset(model, f_loader, device, fp32=args.fp32, allowed_classes=shared_classes)
                f_acc = float(np.mean(f_preds == f_labels) * 100)
                from sklearn.metrics import f1_score
                f_f1 = float(f1_score(f_labels, f_preds, average='macro', labels=shared_classes, zero_division=0) * 100)

                pd.DataFrame({'relpath': [os.path.relpath(p, field_root) for p in field_df['path']],
                              'label': f_labels, 'pred': f_preds}) \
                    .to_csv(os.path.join(jdir, 'zeroshot_predictions_fp32.csv'), index=False)

                field_results.append({
                    'job': jid,
                    'model': job['name'],
                    'seed': job['seed'],
                    'field_acc': f_acc,
                    'field_f1': f_f1
                })

            f_df = pd.DataFrame(field_results)
            f_df.to_csv(os.path.join(out_dir, 'zeroshot.csv'), index=False)
            print(f'Wrote per-job zero-shot results to {os.path.join(out_dir, "zeroshot.csv")}')
            g = f_df.groupby('model', sort=False)
            f_mean = g[['field_acc', 'field_f1']].mean()
            f_std = g[['field_acc', 'field_f1']].std()

            lines.extend([
                '',
                f'## PlantDoc Zero-Shot Robustness Extra ({len(shared_classes)} Shared Tomato Classes)',
                f'Evaluated {len(field_df)} images on field data without fine-tuning. Argmax restricted to shared classes.',
                '',
                '### Per-Class Field Sample Counts',
                '| Class Name | Image Count |',
                '|---|---|'
            ])
            for name, cnt in counts.items():
                lines.append(f"| {name} | {cnt} |")

            lines.extend([
                '',
                '### Field Accuracy and Macro-F1',
                '| Model | Field Accuracy (%) | Field Macro-F1 (%) |',
                '|---|---|---|'
            ])
            for m in f_mean.index:
                acc_s = f"{f_mean.loc[m, 'field_acc']:.2f}" + (f" ± {f_std.loc[m, 'field_acc']:.2f}" if not pd.isna(f_std.loc[m, 'field_acc']) else "")
                f1_s = f"{f_mean.loc[m, 'field_f1']:.2f}" + (f" ± {f_std.loc[m, 'field_f1']:.2f}" if not pd.isna(f_std.loc[m, 'field_f1']) else "")
                lines.append(f"| {m} | {acc_s} | {f1_s} |")

    # Keep the zero-shot section in its own file so a lab-only rerun cannot drop it,
    # but always include it in the combined report.
    zs_path = os.path.join(out_dir, 'zeroshot.md')
    zs_idx = next((i for i, l in enumerate(lines) if l.startswith('## PlantDoc')), None)
    zs_text = None
    if zs_idx is not None:
        zs_text = '\n'.join(lines[zs_idx:])
        with open(zs_path, 'w', encoding='utf-8') as f:
            f.write(zs_text + '\n')
        lines = lines[:zs_idx]
    elif os.path.exists(zs_path):
        with open(zs_path, encoding='utf-8') as f:
            zs_text = f.read().rstrip('\n')

    with open(sig_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
        if zs_text:
            f.write('\n' + zs_text + '\n')
    print(f'Wrote significance report to {sig_path}')


if __name__ == '__main__':
    main()
