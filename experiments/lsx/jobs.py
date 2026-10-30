"""The experiment queue (31 jobs, phases A-E; see README.md). Jobs run in this order; finished jobs are skipped on restart."""

VIT_BASELINES = ['deit_s', 'swin_t', 'mobilevit_xs', 'mobilevit_xxs']

SNN_MAIN = dict(name='leafspike', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=True)
SNN_ABLATIONS = [
    dict(name='leafspike_integer_attn', kind='snn', D=4, attn_mode='integer', maxpool_shortcut=True),
    dict(name='leafspike_no_maxshort', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=False),
    dict(name='leafspike_D2', kind='snn', D=2, attn_mode='aligned', maxpool_shortcut=True),
    dict(name='leafspike_D1', kind='snn', D=1, attn_mode='aligned', maxpool_shortcut=True),
    dict(name='leafspike_firereg', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=True,
         firing_reg_weight=1.0),
    dict(name='leafspike_D2_firereg', kind='snn', D=2, attn_mode='aligned', maxpool_shortcut=True,
         firing_reg_weight=1.0),
]
# models repeated with extra seeds (phase C)
REPEATED = ['deit_s', 'swin_t', 'mobilevit_xs', 'mobilevit_xxs', 'leafspike', 'leafspike_D2_firereg']

# Phase D: extra shortcut seeds + aggressive sparsity push
PHASE_D_SPARSE = [
    dict(name='leafspike_sparse_w10', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=True,
         firing_reg_weight=10.0, firing_reg_target=0.02, firing_reg_last_frac=1.0),
    dict(name='leafspike_sparse_w30', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=True,
         firing_reg_weight=30.0, firing_reg_target=0.02, firing_reg_last_frac=1.0),
]

# Phase E: final sparse models (no shortcut)
PHASE_E_W10 = dict(name='leafspike_final_sparse_w10', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=False,
                   firing_reg_weight=10.0, firing_reg_target=0.02, firing_reg_last_frac=1.0)
PHASE_E_W5 = dict(name='leafspike_final_sparse_w5', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=False,
                  firing_reg_weight=5.0, firing_reg_target=0.02, firing_reg_last_frac=1.0)


def _vit(arch, seed, epochs):
    return dict(id=f'{arch}_s{seed}', name=arch, kind='vit', arch=arch, seed=seed, epochs=epochs, phase='')


def _snn(spec, seed, epochs):
    j = dict(spec)
    j.update(id=f'{spec["name"]}_s{seed}', seed=seed, epochs=epochs, phase='')
    return j


def build_jobs(cfg):
    seeds, E = cfg['seeds'], cfg['epochs']
    s0 = seeds[0]
    jobs = []
    for a in VIT_BASELINES:                                   # phase A: baselines + main model, first seed
        jobs.append(_vit(a, s0, E) | {'phase': 'A'})
    jobs.append(_snn(SNN_MAIN, s0, E) | {'phase': 'A'})
    for spec in SNN_ABLATIONS:                                # phase B: ablations, first seed
        jobs.append(_snn(spec, s0, E) | {'phase': 'B'})
    by_name = {SNN_MAIN['name']: SNN_MAIN} | {s['name']: s for s in SNN_ABLATIONS}
    for s in seeds[1:]:                                       # phase C: extra seeds
        for name in REPEATED:
            if name in VIT_BASELINES:
                jobs.append(_vit(name, s, E) | {'phase': 'C'})
            else:
                jobs.append(_snn(by_name[name], s, E) | {'phase': 'C'})
    # Phase D: leafspike_no_maxshort seeds 43/44 + aggressive sparsity w10/w30 seed 42
    no_maxshort_spec = dict(name='leafspike_no_maxshort', kind='snn', D=4, attn_mode='aligned', maxpool_shortcut=False)
    for s in seeds[1:]:
        jobs.append(_snn(no_maxshort_spec, s, E) | {'phase': 'D'})
    for spec in PHASE_D_SPARSE:
        jobs.append(_snn(spec, s0, E) | {'phase': 'D'})

    # Phase E: final sparse models (no maxpool shortcut)
    for s in seeds:
        jobs.append(_snn(PHASE_E_W10, s, E) | {'phase': 'E'})
    jobs.append(_snn(PHASE_E_W5, s0, E) | {'phase': 'E'})

    only = cfg.get('only_jobs')
    if only:
        jobs = [j for j in jobs if j['id'] in only or j['name'] in only]
    return jobs
