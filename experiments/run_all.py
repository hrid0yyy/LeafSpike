"""Run every LeafSpike / ViT experiment of the paper (lsx/jobs.py), resumably.

    python run_all.py              run (or continue) the whole queue, then write the report
    python run_all.py --status     show progress and exit
    python run_all.py --report     rebuild the report from finished jobs and exit
    python run_all.py --smoke      tiny end-to-end check (separate folder, 1 epoch, few images)

Safe to start again at any time (e.g. by the Windows start-up task): finished jobs are skipped, an interrupted
job continues from its last finished epoch, and a second copy of the runner exits immediately (lock file).
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    root=os.environ.get('LEAFSPIKE_ROOT', os.path.join(os.path.expanduser('~'), 'leafspike')),
    lab_root=None,            # default: <root>/data/plantvillage
    field_root=None,          # optional PlantDoc folder for the zero-shot extra (None = skip)
    sdt_checkpoint=None,      # default: <root>/checkpoints/sdt_v3_s_imagenet.pth
    img_size=224, batch_size=64, num_workers=4,
    seeds=[42, 43, 44], epochs=60, warmup_epochs=3,
    lr_vit_pretrained=1e-4, lr_vit_scratch=1e-3, lr_snn_pretrained=2e-4, lr_snn_scratch=1e-3,
    new_param_lr_mult=10.0, weight_decay=0.05, label_smoothing=0.1, clip_grad=5.0, drop_path=0.1,
    mixup_alpha=0.2, cutmix_alpha=0.5, mix_prob=0.5, bg_paste_prob=0.5,
    firing_reg_target=0.05, firing_reg_last_frac=0.3,
    pretrained_vits=True, stats_batches=20, latency_warmup=20, latency_iters=100,
    max_attempts=3, keep_last_checkpoints=False, max_images_per_class=None, only_jobs=None,
    test_frac=0.15, val_frac=0.15, split_seed=42,
)


def load_cfg(args):
    cfg = dict(DEFAULTS)
    path = args.config or os.path.join(HERE, 'config.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:      # tolerate the BOM PowerShell 5 writes
            user = json.load(f)
        unknown = set(user) - set(DEFAULTS) - {'_comment'}
        if unknown:
            raise SystemExit(f'Unknown keys in {path}: {sorted(unknown)}')
        cfg.update({k: v for k, v in user.items() if k != '_comment'})
    root = cfg['root']
    cfg['lab_root'] = cfg['lab_root'] or os.path.join(root, 'data', 'plantvillage')
    cfg['sdt_checkpoint'] = cfg['sdt_checkpoint'] or os.path.join(root, 'checkpoints', 'sdt_v3_s_imagenet.pth')
    cfg['runs_dir'] = os.path.join(root, 'runs_smoke' if args.smoke else 'runs')
    if args.smoke:
        cfg.update(epochs=1, warmup_epochs=1, seeds=[42], max_images_per_class=16, stats_batches=1,
                   latency_warmup=1, latency_iters=2, batch_size=min(cfg['batch_size'], 8),
                   only_jobs=['deit_s_s42', 'mobilevit_xxs_s42', 'leafspike_s42', 'leafspike_D2_firereg_s42'])
    os.makedirs(cfg['runs_dir'], exist_ok=True)
    cache = os.path.join(root, 'cache')                      # persistent model-weight caches
    os.environ.setdefault('TORCH_HOME', os.path.join(cache, 'torch'))
    os.environ.setdefault('HF_HOME', os.path.join(cache, 'huggingface'))
    return cfg


class SingleInstance:
    """Exclusive lock on runs/runner.lock, released automatically if the process dies."""

    def __init__(self, path):
        self.path, self.f = path, None

    def __enter__(self):
        self.f = open(self.path, 'a+')
        try:
            if os.name == 'nt':
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise SystemExit('Another run_all.py is already running (lock held). Exiting.')
        return self

    def __exit__(self, *a):
        try:
            if os.name == 'nt':
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self.f.close()


def make_logger(path):
    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def job_status(cfg, job):
    from lsx.train import read_json
    jdir = os.path.join(cfg['runs_dir'], job['id'])
    if os.path.exists(os.path.join(jdir, 'result.json')):
        return 'done', read_json(os.path.join(jdir, 'state.json'), {}) or {}
    st = read_json(os.path.join(jdir, 'state.json'), {}) or {}
    if st.get('status') == 'failed':
        return 'failed', st
    if st:
        return 'partial', st
    return 'pending', st


def show_status(cfg, jobs):
    from lsx.train import read_json
    counts = {}
    for j in jobs:
        s, st = job_status(cfg, j)
        counts[s] = counts.get(s, 0) + 1
        extra = f" (attempts {st.get('attempts')}, batch {st.get('batch_size')}x{st.get('accum')})" if st else ''
        print(f'  [{s:7s}] phase {j["phase"]}  {j["id"]}{extra}')
    print('summary:', counts)
    p = read_json(os.path.join(cfg['runs_dir'], 'progress.json'))
    if p:
        print('latest epoch:', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    cfg = load_cfg(args)
    sys.path.insert(0, HERE)
    from lsx.jobs import build_jobs
    jobs = build_jobs(cfg)
    if args.status:
        show_status(cfg, jobs)
        return
    if args.report:
        from lsx.report import write_report
        write_report(cfg, jobs)
        return

    with SingleInstance(os.path.join(cfg['runs_dir'], 'runner.lock')):
        log = make_logger(os.path.join(cfg['runs_dir'], 'runner.log'))
        import torch
        from lsx import data as D
        from lsx.report import write_report
        from lsx.train import atomic_json, read_json, run_job
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        log(f'start | torch {torch.__version__} | device {device} '
            f'{torch.cuda.get_device_name(0) if device.type == "cuda" else ""} | runs {cfg["runs_dir"]}')
        if device.type != 'cuda' and not args.smoke:
            log('[warn] CUDA not available: training on CPU would take weeks. Check the PyTorch CUDA install.')
        atomic_json(cfg, os.path.join(cfg['runs_dir'], 'config_used.json'))

        log('preparing split ...')
        split_df = D.build_lab_split(cfg['lab_root'], os.path.join(cfg['runs_dir'], 'split.csv'),
                                     cfg['test_frac'], cfg['val_frac'], cfg['split_seed'],
                                     cfg['max_images_per_class'])
        log('split: ' + json.dumps(split_df.groupby('split').size().to_dict()))
        field_df = D.load_field(cfg['field_root']) if cfg['field_root'] else None
        if field_df is not None:
            log(f'field (zero-shot extra): {len(field_df)} images')

        for k, job in enumerate(jobs):
            while True:                      # loops again only to retry after CUDA out-of-memory
                s, st = job_status(cfg, job)
                if s in ('done', 'failed'):
                    break
                if st.get('attempts', 0) >= cfg['max_attempts']:
                    st['status'] = 'failed'
                    atomic_json(st, os.path.join(cfg['runs_dir'], job['id'], 'state.json'))
                    log(f'[{job["id"]}] giving up after {st["attempts"]} attempts')
                    break
                log(f'=== job {k+1}/{len(jobs)}: {job["id"]} (phase {job["phase"]}) ===')
                retry = False
                try:
                    res = run_job(job, cfg, split_df, field_df, device, log)
                    log(f'[{job["id"]}] done: lab test acc {res["lab_test"]["acc"]*100:.2f} '
                        f'F1 {res["lab_test"]["macro_f1"]*100:.2f}')
                    del res
                except Exception as e:   # keep the queue going; failed jobs are retried on the next start
                    oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'out of memory' in str(e).lower()
                    state_p = os.path.join(cfg['runs_dir'], job['id'], 'state.json')
                    st = read_json(state_p, {}) or {}
                    if oom and st.get('batch_size', cfg['batch_size']) > 8:
                        st['batch_size'] = st.get('batch_size', cfg['batch_size']) // 2
                        st['accum'] = st.get('accum', 1) * 2
                        st['attempts'] = max(0, st.get('attempts', 1) - 1)   # OOM retries do not count
                        log(f'[{job["id"]}] CUDA out of memory -> retrying with batch {st["batch_size"]} '
                            f'x accumulation {st["accum"]}')
                        retry = True
                    else:
                        log(f'[{job["id"]}] FAILED: {e}\n{traceback.format_exc()}')
                    st['status'] = 'error'
                    st['last_error'] = str(e)[:2000]
                    atomic_json(st, state_p)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if not retry:
                    break
            try:
                write_report(cfg, jobs)
            except Exception as e:
                log(f'[warn] report failed: {e}')
        write_report(cfg, jobs)
        log('queue finished')


if __name__ == '__main__':
    main()
