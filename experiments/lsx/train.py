"""Resumable training and evaluation of one job.

Each job lives in runs/<job_id>/:
  state.json   status (pending/running/done/failed), attempts, batch/accumulation actually used
  last.pth     full training state after the latest finished epoch (model, optimizer, scaler, history, RNG)
  last_prev.pth  the previous one (fallback if last.pth is damaged by a power cut)
  best.pth     weights of the best validation epoch (macro-F1)
  result.json  final metrics, ops and energy (written last; its presence means the job is done)
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from . import data as D
from .energy import gpu_latency_ms, profile_model
from .models import (IMAGENET_MEAN, IMAGENET_STD, MON, LeafSpike, build_leafspike, build_vit,
                     load_sdt_checkpoint, new_param_prefixes)


# ---------------------------------------------------------------- small utilities

def atomic_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def atomic_torch_save(obj, path, keep_prev=False):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    if keep_prev and os.path.exists(path):
        os.replace(path, path.replace('.pth', '_prev.pth'))
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AmpCtx:
    def __init__(self, device):
        self.device = device

    def __call__(self):
        if self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        return torch.autocast(device_type='cpu', enabled=False)


def make_scaler(device):
    if device.type != 'cuda':
        return None
    try:
        return torch.amp.GradScaler('cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def soft_targets(y, n, smoothing):
    t = torch.full((y.size(0), n), smoothing / n, device=y.device)
    t.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + smoothing / n)
    return t


def mix_batch(x, t, cfg, rng):
    if rng.random() >= cfg['mix_prob'] or x.size(0) < 2:
        return x, t
    perm = torch.randperm(x.size(0), device=x.device)
    if cfg['cutmix_alpha'] > 0 and rng.random() < 0.5:
        lam = rng.beta(cfg['cutmix_alpha'], cfg['cutmix_alpha'])
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = rng.integers(H), rng.integers(W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / float(H * W)
    else:
        lam = rng.beta(cfg['mixup_alpha'], cfg['mixup_alpha']) if cfg['mixup_alpha'] > 0 else 1.0
        x = lam * x + (1 - lam) * x[perm]
    return x, lam * t + (1 - lam) * t[perm]


def param_groups(model, base_lr, wd, new_prefixes, new_mult):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_new = any(n.startswith(s) or s in n for s in new_prefixes)
        no_wd = p.ndim <= 1
        key = (is_new, no_wd)
        if key not in groups:
            groups[key] = {'params': [], 'weight_decay': 0.0 if no_wd else wd,
                           'base_lr': base_lr * (new_mult if is_new else 1.0)}
        groups[key]['params'].append(p)
    return list(groups.values())


def make_loader(ds, train, bs, workers, device, generator=None):
    """Loaders are created once per job (persistent workers: spawning workers is slow on Windows).
    For training, `generator` is re-seeded every epoch so the shuffling order is reproducible on resume."""
    sampler = torch.utils.data.RandomSampler(ds, generator=generator) if train else None
    return torch.utils.data.DataLoader(
        ds, batch_size=bs, sampler=sampler, shuffle=False, num_workers=workers,
        pin_memory=(device.type == 'cuda'), drop_last=(train and len(ds) >= 2 * bs),
        persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        with amp():
            out = model(x.to(device, non_blocking=True))
        L.append(out.float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def metrics(logits, y, allowed=None):
    logits = logits.clone()
    if allowed is not None:
        mask = torch.full((logits.size(1),), float('-inf'))
        mask[list(allowed)] = 0
        logits = logits + mask
    pred, yt = logits.argmax(1).numpy(), y.numpy()
    labels = sorted(set(yt.tolist()))
    return {'n': int(len(yt)), 'acc': float(accuracy_score(yt, pred)),
            'macro_f1': float(f1_score(yt, pred, labels=labels, average='macro', zero_division=0)),
            'per_class_recall': {D.CLASSES[c]: float((pred[yt == c] == c).mean()) for c in labels}}


# ---------------------------------------------------------------- model construction

def build_model(job, cfg):
    """Returns (model, pretrained_loaded, mean, std, base_lr, new_prefixes)."""
    if job['kind'] == 'snn':
        m = build_leafspike(D.NUM_CLASSES, size=job.get('size', 's'), D=job.get('D', 4),
                            attn_mode=job.get('attn_mode', 'aligned'),
                            maxpool_shortcut=job.get('maxpool_shortcut', True), drop_path=cfg['drop_path'])
        loaded = False
        ck = cfg['sdt_checkpoint']
        if job.get('size', 's') == 's' and ck and os.path.exists(ck):
            frac = load_sdt_checkpoint(m, ck)
            loaded = frac >= 0.9
            print(f'   SDT-v3 checkpoint: {frac*100:.1f}% backbone tensors matched -> loaded={loaded}')
        else:
            print(f'   [warn] SDT-v3 checkpoint not found at {ck}; LeafSpike trains from scratch')
        lr = cfg['lr_snn_pretrained'] if loaded else cfg['lr_snn_scratch']
        return m, loaded, IMAGENET_MEAN, IMAGENET_STD, lr, (new_param_prefixes(m) if loaded else [])
    m, loaded, mean, std = build_vit(job['arch'], D.NUM_CLASSES, drop_path=cfg['drop_path'],
                                     pretrained=cfg['pretrained_vits'])
    lr = cfg['lr_vit_pretrained'] if loaded else cfg['lr_vit_scratch']
    return m, loaded, mean, std, lr, (new_param_prefixes(m) if loaded else [])


# ---------------------------------------------------------------- one job

def run_job(job, cfg, split_df, field_df, device, log):
    jdir = os.path.join(cfg['runs_dir'], job['id'])
    os.makedirs(jdir, exist_ok=True)
    state_p = os.path.join(jdir, 'state.json')
    state = read_json(state_p, {}) or {}
    bs = state.get('batch_size', cfg['batch_size'])
    accum = state.get('accum', 1)
    state.update(status='running', attempts=state.get('attempts', 0) + 1, batch_size=bs, accum=accum,
                 job=job, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(state, state_p)

    amp = AmpCtx(device)
    seed_everything(job['seed'])
    model, loaded, mean, std, base_lr, new_prefixes = build_model(job, cfg)
    model.to(device)
    snn = isinstance(model, LeafSpike)
    tr_df, va_df, te_df = (split_df[split_df.split == s] for s in ('train', 'val', 'test'))
    tr_ds = D.LeafDataset(tr_df, D.make_train_tf(cfg['img_size'], mean, std), bg_paste_prob=cfg['bg_paste_prob'])
    va_ds = D.LeafDataset(va_df, D.make_eval_tf(cfg['img_size'], mean, std))
    te_ds = D.LeafDataset(te_df, D.make_eval_tf(cfg['img_size'], mean, std))
    va_loader = make_loader(va_ds, False, bs, cfg['num_workers'], device)
    gen = torch.Generator()
    tr_loader = make_loader(tr_ds, True, bs, cfg['num_workers'], device, generator=gen)

    opt = torch.optim.AdamW(param_groups(model, base_lr, cfg['weight_decay'], new_prefixes,
                                         cfg['new_param_lr_mult']), lr=base_lr)
    scaler = make_scaler(device)
    epochs = job['epochs']
    steps_per_epoch = max(1, (len(tr_ds) // bs) // accum)
    total = epochs * steps_per_epoch
    warm = min(cfg['warmup_epochs'] * steps_per_epoch, max(1, total // 5))
    start_ep, step, best, hist = 0, 0, -1.0, []

    last_p = os.path.join(jdir, 'last.pth')
    for cand in (last_p, os.path.join(jdir, 'last_prev.pth')):
        if os.path.exists(cand):
            try:
                ck = torch.load(cand, map_location='cpu', weights_only=False)
                model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['opt'])
                if scaler is not None and ck.get('scaler'):
                    scaler.load_state_dict(ck['scaler'])
                start_ep, step, best, hist = ck['epoch'] + 1, ck['step'], ck['best'], ck['hist']
                torch.set_rng_state(ck['rng_torch'])
                np.random.set_state(ck['rng_np'])
                random.setstate(ck['rng_py'])
                log(f'   resumed from {os.path.basename(cand)} at epoch {start_ep}/{epochs}')
                break
            except Exception as e:
                log(f'   [warn] could not resume from {cand}: {e}')

    reg_w = job.get('firing_reg_weight', 0.0)
    reg_last_frac = job.get('firing_reg_last_frac', cfg['firing_reg_last_frac'])
    reg_start = int(epochs * (1 - reg_last_frac))
    reg_target = job.get('firing_reg_target', cfg['firing_reg_target'])
    rng = np.random.default_rng(job['seed'] * 1000 + start_ep)
    t_job = time.time()
    for ep in range(start_ep, epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        reg_on = snn and reg_w > 0 and ep >= reg_start
        gen.manual_seed(job['seed'] * 1000 + ep)
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(tr_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            t = soft_targets(y, D.NUM_CLASSES, cfg['label_smoothing'])
            x, t = mix_batch(x, t, cfg, rng)
            MON.collect_reg = reg_on
            MON.reg_terms = []
            with amp():
                out = model(x)
                loss = torch.sum(-t * F.log_softmax(out.float(), 1), 1).mean()
                if reg_on and MON.reg_terms:
                    rates = torch.stack(MON.reg_terms)
                    loss = loss + reg_w * F.relu(rates - reg_target).mean()
            MON.collect_reg = False
            MON.reg_terms = []
            loss_b = loss / accum
            if scaler is not None:
                scaler.scale(loss_b).backward()
            else:
                loss_b.backward()
            if (it + 1) % accum != 0:
                continue
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * max(f, 1e-3)
            if scaler is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
                opt.step()
            opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())
            nb += 1
            step += 1
        lg, yv = predict(model, va_loader, device, amp)
        m = metrics(lg, yv)
        hist.append({'epoch': ep + 1, 'loss': tot / max(1, nb), 'val_acc': m['acc'], 'val_f1': m['macro_f1'],
                     'sec': time.time() - t0})
        flag = ''
        if m['macro_f1'] > best:
            best = m['macro_f1']
            flag = ' *'
            atomic_torch_save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                              os.path.join(jdir, 'best.pth'))
        atomic_torch_save({'model': model.state_dict(), 'opt': opt.state_dict(),
                           'scaler': scaler.state_dict() if scaler is not None else None,
                           'epoch': ep, 'step': step, 'best': best, 'hist': hist,
                           'rng_torch': torch.get_rng_state(), 'rng_np': np.random.get_state(),
                           'rng_py': random.getstate()}, last_p, keep_prev=True)
        eta = (time.time() - t_job) / max(1, ep + 1 - start_ep) * (epochs - ep - 1) / 3600
        log(f'[{job["id"]}] ep {ep+1:3d}/{epochs} loss {tot/max(1,nb):.4f} val acc {m["acc"]*100:6.2f} '
            f'F1 {m["macro_f1"]*100:6.2f} ({time.time()-t0:.0f}s, job ETA {eta:.1f} h){flag}')
        atomic_json({'job': job['id'], 'epoch': ep + 1, 'epochs': epochs, 'val_f1': m['macro_f1'],
                     'eta_hours': eta, 'time': time.strftime('%Y-%m-%d %H:%M:%S')},
                    os.path.join(cfg['runs_dir'], 'progress.json'))

    # ---- final evaluation with the best weights
    model.load_state_dict(torch.load(os.path.join(jdir, 'best.pth'), map_location='cpu', weights_only=False))
    model.to(device)
    te_loader = make_loader(te_ds, False, bs, cfg['num_workers'], device)
    lg, yt = predict(model, te_loader, device, amp)
    res = {'job': job, 'pretrained': loaded, 'batch_size': bs, 'accum': accum, 'history': hist,
           'best_val_f1': best, 'lab_test': metrics(lg, yt)}
    if snn:
        other = 'integer' if job.get('attn_mode', 'aligned') == 'aligned' else 'aligned'
        model.set_attn_mode(other)
        lg2, yt2 = predict(model, te_loader, device, amp)
        res[f'lab_test_as_{other}_attention'] = metrics(lg2, yt2)
        model.set_attn_mode(job.get('attn_mode', 'aligned'))
    if field_df is not None and len(field_df):
        f_ds = D.LeafDataset(field_df, D.make_eval_tf(cfg['img_size'], mean, std))
        lg, yf = predict(model, make_loader(f_ds, False, bs, cfg['num_workers'], device), device, amp)
        res['zeroshot_field'] = metrics(lg, yf, allowed=sorted(field_df.label.unique().tolist()))
    res['profile'] = profile_model(model, te_loader, cfg['stats_batches'], device, cfg['img_size'], amp)
    res['latency_ms_bs1'] = gpu_latency_ms(model, device, cfg['img_size'], amp,
                                           warmup=cfg['latency_warmup'], iters=cfg['latency_iters'])
    res['train_hours'] = sum(h['sec'] for h in hist) / 3600
    atomic_json(res, os.path.join(jdir, 'result.json'))
    state.update(status='done', finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(state, state_p)
    for p in (last_p, os.path.join(jdir, 'last_prev.pth')):   # free disk; best.pth is kept
        if os.path.exists(p) and not cfg['keep_last_checkpoints']:
            os.remove(p)
    return res
