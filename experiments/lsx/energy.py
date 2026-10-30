"""Operation counts, firing statistics, energy estimates and latency.

Energy constants are 45 nm figures (Horowitz, ISSCC 2014), the convention used by spiking-transformer papers:
  FP32 MAC 4.6 pJ, FP32 add (AC) 0.9 pJ, INT8 MAC 0.23 pJ (0.2 mult + 0.03 add), INT8 add 0.03 pJ,
  SRAM ~1.25 pJ/byte (8 KB array, 64-bit read ~10 pJ), DRAM ~160 pJ/byte (~640 pJ per 32-bit access).

Four estimates are reported for every model:
  * compute_fp32_mJ  - the usual spiking-paper arithmetic-only estimate
  * compute_int8_mJ  - arithmetic only, INT8 weights/activations for both ANN and SNN
  * mem_int8_mJ      - compute_int8 + upper-bound memory model: weights fetched once from DRAM (INT8);
                       ANN activations written+read once (INT8, SRAM);
                       SNN: per spiking neuron and per virtual step, membrane read+write (16 bit) and
                       spike write+read (1 bit), SRAM.
  * mem_int8_lb_mJ   - compute_int8 + event-driven memory lower bound: weights fetched once from DRAM (INT8);
                       spike traffic as above (1 bit write+read per virtual step);
                       membrane read+write charged ONLY for neurons that fire (frac_nonzero * neurons * D * 4 bytes, SRAM).
                       For ANNs, mem_int8_lb_mJ equals mem_int8_mJ.
  * mem_int8_onchip_mJ / mem_int8_onchip_lb_mJ - same as mem_int8_mJ / mem_int8_lb_mJ but with the weights read
                       from on-chip SRAM (1.25 pJ/byte) instead of DRAM (160 pJ/byte). This isolates how much of
                       the memory estimate depends on the one DRAM-fetch assumption; it is a sensitivity analysis,
                       not a claim that a 4.9M-parameter model fits on chip.
These are rough, transparent estimates, not measurements; they are reported side by side on purpose.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .models import MON, LeafSpike

E = dict(fp32_mac=4.6, fp32_ac=0.9, int8_mac=0.23, int8_ac=0.03, sram_byte=1.25, dram_byte=160.0)  # pJ


def compute_mem_lb(prof):
    """Compute mem_int8_lb_mJ from a saved profile dict."""
    if 'mem_int8_lb_mJ' in prof:
        return prof['mem_int8_lb_mJ']
    if 'layers' in prof:
        layers = prof['layers']
        D = prof.get('D', 4)
        n_params = prof['params_M'] * 1e6
        mem_w = n_params * 1.0 * E['dram_byte']
        comp_int8 = prof['compute_int8_mJ'] * 1e9
        neurons = sum(l['neurons'] for l in layers)
        spike_bytes = neurons * D * (2 / 8.0)
        active_mem_bytes = sum(l['frac_nonzero'] * l['neurons'] for l in layers) * D * 4.0
        act_bytes_lb = spike_bytes + active_mem_bytes
        return (comp_int8 + mem_w + act_bytes_lb * E['sram_byte']) * 1e-9
    return prof.get('mem_int8_mJ', 0.0)


def compute_mem_onchip(prof, lower_bound=False):
    """Memory estimate with weights read from on-chip SRAM instead of DRAM (see module docstring)."""
    key = 'mem_int8_onchip_lb_mJ' if lower_bound else 'mem_int8_onchip_mJ'
    if key in prof:
        return prof[key]
    if 'layers' in prof:
        layers = prof['layers']
        D = prof.get('D', 4)
        neurons = sum(l['neurons'] for l in layers)
        if lower_bound:
            act_bytes = neurons * D * (2 / 8.0) + \
                sum(l['frac_nonzero'] * l['neurons'] for l in layers) * D * 4.0
        else:
            act_bytes = neurons * D * (2 * 2 + 2 / 8.0)
    elif 'activations_M' in prof:
        act_bytes = prof['activations_M'] * 1e6 * 2.0
    else:
        return prof.get('mem_int8_mJ', 0.0)
    n_params = prof['params_M'] * 1e6
    comp_int8 = prof['compute_int8_mJ'] * 1e9
    return (comp_int8 + n_params * 1.0 * E['sram_byte'] + act_bytes * E['sram_byte']) * 1e-9


def dense_macs(model, img_size, device='cpu'):
    """Dense-equivalent MACs for one image (conv, linear and attention matmuls), via torch's FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    fused = [m for m in model.modules() if hasattr(m, 'fused_attn')]
    saved = [m.fused_attn for m in fused]
    for m in fused:
        m.fused_attn = False              # explicit q@k / attn@v so the counter sees them
    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        macs = fc.get_total_flops() / 2.0
    finally:
        for m, s in zip(fused, saved):
            m.fused_attn = s
        model.train(was_training)
    return macs


def _layer_hooks(model, store, snn, D):
    def hook(m, inp, out):
        a = inp[0]
        B = a.shape[0]
        if isinstance(m, nn.Conv2d):
            per = out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Conv1d):
            per = out[0].numel() * (m.in_channels // m.groups) * m.kernel_size[0]
        else:
            per = (out.numel() / B) * m.in_features
        store['out_elems'] += out.numel()
        if snn:
            if getattr(m, '_is_stem', False):
                store['stem_macs'] += per * B
            else:
                norm = getattr(m, '_in_norm', float(D))
                store['sops'] += per * float(a.float().mean()) * norm * B
    return [m.register_forward_hook(hook) for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear))]


def profile_model(model, loader, n_batches, device, img_size, amp_ctx):
    """Ops, firing statistics and the four energy estimates for one model."""
    snn = isinstance(model, LeafSpike)
    D = 1
    if snn:
        D = next(m for m in model.modules() if m.__class__.__name__ == 'MultiSpike').max_value
    n_params = sum(p.numel() for p in model.parameters())
    macs = dense_macs(model.cpu(), img_size, 'cpu')
    model.to(device).eval()
    store = dict(out_elems=0.0, stem_macs=0.0, sops=0.0)
    hooks = _layer_hooks(model, store, snn, D)
    MON.reset()
    MON.collect_stats = snn
    seen = 0
    try:
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= n_batches:
                    break
                with amp_ctx():
                    model(x.to(device, non_blocking=True))
                seen += x.size(0)
    finally:
        MON.collect_stats = False
        for h in hooks:
            h.remove()
    seen = max(1, seen)
    res = {'params_M': n_params / 1e6, 'dense_GMACs': macs / 1e9}
    w_bytes = n_params * 1.0                                  # INT8 weights
    mem_w = w_bytes * E['dram_byte']
    if snn:
        sops = store['sops'] / seen + MON.attn_ops / seen
        stem = store['stem_macs'] / seen
        layers = [{'layer': t, 'mean_spikes_per_neuron': s / c, 'frac_nonzero': nz / c, 'neurons': c / seen}
                  for t, (s, nz, c) in MON.stats.items()]
        neurons = sum(l['neurons'] for l in layers)
        avg_spk = sum(l['mean_spikes_per_neuron'] * l['neurons'] for l in layers) / max(1, neurons)
        comp_fp32 = stem * E['fp32_mac'] + sops * E['fp32_ac']
        comp_int8 = stem * E['int8_mac'] + sops * E['int8_ac']
        act_bytes = neurons * D * (2 * 2 + 2 / 8.0)          # membrane r+w (16 bit) + spike w+r (1 bit) per step
        spike_bytes = neurons * D * (2 / 8.0)
        active_mem_bytes = sum(l['frac_nonzero'] * l['neurons'] for l in layers) * D * 4.0
        act_bytes_lb = spike_bytes + active_mem_bytes
        mem_lb = (comp_int8 + mem_w + act_bytes_lb * E['sram_byte']) * 1e-9
        mem_onchip_w = n_params * 1.0 * E['sram_byte']
        mem_onchip = (comp_int8 + mem_onchip_w + act_bytes * E['sram_byte']) * 1e-9
        mem_onchip_lb = (comp_int8 + mem_onchip_w + act_bytes_lb * E['sram_byte']) * 1e-9
        res.update(GSOPs=sops / 1e9, stem_GMACs=stem / 1e9, avg_spikes_per_neuron=avg_spk, D=D,
                   frac_neurons_firing=float(np.mean([l['frac_nonzero'] for l in layers])) if layers else 0.0,
                   neurons_M=neurons / 1e6, layers=layers)
    else:
        comp_fp32 = macs * E['fp32_mac']
        comp_int8 = macs * E['int8_mac']
        act_bytes = (store['out_elems'] / seen) * 2.0          # write + read once, INT8
        mem_lb = (comp_int8 + mem_w + act_bytes * E['sram_byte']) * 1e-9
        mem_onchip_w = n_params * 1.0 * E['sram_byte']
        mem_onchip = (comp_int8 + mem_onchip_w + act_bytes * E['sram_byte']) * 1e-9
        mem_onchip_lb = mem_onchip
        res.update(activations_M=store['out_elems'] / seen / 1e6)
    res.update(compute_fp32_mJ=comp_fp32 * 1e-9, compute_int8_mJ=comp_int8 * 1e-9,
               mem_int8_mJ=(comp_int8 + mem_w + act_bytes * E['sram_byte']) * 1e-9,
               mem_int8_lb_mJ=mem_lb,
               mem_int8_onchip_mJ=mem_onchip, mem_int8_onchip_lb_mJ=mem_onchip_lb)
    return res


def gpu_latency_ms(model, device, img_size, amp_ctx, warmup=20, iters=100):
    """Median batch-1 latency on this machine (PyTorch, no special kernels; SNN runs as dense tensors)."""
    model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    times = []
    with torch.no_grad():
        for i in range(warmup + iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with amp_ctx():
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))
