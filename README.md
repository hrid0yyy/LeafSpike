# LeafSpike: time-aligned attention for spike-driven vision transformers

Code, data split and per-image predictions for the paper

> **Train as you deploy: time-aligned attention closes the train–deploy gap of spike-driven vision transformers, with an edge plant-disease case study**
> S. M. Mahbub Uddin Ahmed. Manuscript in preparation.

Spike-driven transformers such as E-SpikeFormer (SDT-v3) are trained with integer spike counts and deployed as binary
spikes over $D$ virtual time steps. This is exact for convolutions, but not for self-attention, which multiplies spike
tensors with each other: a binary step-wise executor computes $\min(c_q, c_k, c_v)$ where integer training computes
$c_q c_k c_v$. **Time-aligned attention** trains the attention in exactly the form that binary hardware executes:

```python
# experiments/lsx/models.py, MS_Attention_linear.forward  (mode='aligned')
qc, kc, vc = q * D, k * D, v * D                      # integer spike counts
out = 0
for d in range(D):                                    # thermometer planes: spike at step d iff count > d
    qd, kd, vd = (qc - d).clamp(0, 1), (kc - d).clamp(0, 1), (vc - d).clamp(0, 1)
    out = out + qd @ (kd.transpose(-2, -1) @ vd)
out = out / float(D ** 3)
```

It adds no parameters, and a pretrained SDT-v3 checkpoint loads unchanged.

## Main results (PlantVillage-tomato lab test set, 2,593 images, mean ± s.d. over 3 seeds)

| Model | Params (M) | Accuracy (%) | FP32 arithmetic energy (mJ) |
|---|---|---|---|
| DeiT-S | 21.67 | 99.96 ± 0.04 | 21.153 |
| Swin-T | 27.53 | 99.97 ± 0.04 | 20.653 |
| MobileViT-XS | 1.94 | 99.56 ± 0.08 | 3.551 |
| MobileViT-XXS | 0.95 | 98.68 ± 0.16 | 1.385 |
| **LeafSpike-S** (time-aligned) | 4.91 | 99.73 ± 0.10 | 1.604 |
| **LeafSpike-S-sparse** ($w{=}10$) | 4.91 | 99.50 ± 0.07 | 1.032 |

* An integer-attention model drops from 99.69 % to 96.18 % when executed as binary spikes; the time-aligned model is
  evaluated in its deployed form by construction.
* Energy values are **estimates** (45 nm operation costs); memory-aware estimates with weights in DRAM favour both
  MobileViTs. See the paper and `results/summary.md` for all energy models, significance tests and limitations.
* Zero-shot on PlantDoc field images: 35–41 % accuracy for every model.

Full numbers: [`results/summary.md`](results/summary.md), [`results/significance.md`](results/significance.md),
[`results/zeroshot.md`](results/zeroshot.md), [`results/stability.md`](results/stability.md),
[`results/mode_sensitivity.md`](results/mode_sensitivity.md).

## Repository layout

```
experiments/
  lsx/models.py        LeafSpike-S (I-LIF neurons, time-aligned / integer attention) and timm ViT baselines
  lsx/data.py          PlantVillage-tomato loading, duplicate-aware stratified split, augmentation, PlantDoc loader
  lsx/train.py         training loop (AdamW, warm-up + cosine, mixup/CutMix, fp16, resume)
  lsx/energy.py        operation counts and energy estimates (FP32, INT8, INT8 + memory)
  lsx/jobs.py          the 31-job experiment queue (phases A–E)
  lsx/report.py        result tables and the accuracy–energy figure
  run_all.py           runs / resumes the whole queue and writes the report
  eval_saved.py        deterministic FP32 re-evaluation, cross-mode accuracy, McNemar + Holm, zero-shot PlantDoc
  compare_runs.py      prediction-stability report (results/stability.md)
  mode_sensitivity.py  q/k/v firing statistics (results/mode_sensitivity.md)
  make_tables.py       LaTeX tables of the paper (written to paper/tables/)
  windows/             optional unattended set-up for a Windows GPU PC (scheduled task, auto-resume)
figures/
  make_paper_figures.py  cross-mode and firing-profile figures from results/
results/
  split.csv            the fixed train/val/test split (18,146 images, seed 42)
  runs/<job>/          result.json and per-image predictions of every job (31 jobs)
  *.md, *.csv          summary tables, significance, zero-shot, stability, mode sensitivity
```

## Installation

Python 3.11 or 3.12 and a CUDA GPU (experiments ran on one RTX 3060, 12 GB).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements.txt
```

## Data

* **PlantVillage** (training and lab test): the colour images of the ten tomato classes. We used the Kaggle copy
  `abdallahalidev/plantvillage-dataset`, extracted to `<root>/data/plantvillage`. That copy contains every tomato image
  twice; `lsx/data.py` removes byte-identical and same-name duplicates (18,146 images remain). The split we used is
  in `results/split.csv` (columns `relpath, label, cls, md5, split`) and is reused automatically when it is placed in
  `<root>/runs/`. The `relpath` column uses Windows path separators; on Linux/macOS replace `\` with `/` first.
* **PlantDoc** (zero-shot field test, optional): the [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
  (CC BY 4.0); set `field_root` in the config or pass `--field` to `eval_saved.py`. Only the 9 tomato classes shared
  with PlantVillage are used.
* **SDT-v3 checkpoint**: the ImageNet-1k checkpoint of the small E-SpikeFormer model from the
  [official repository](https://github.com/BICLab/Spike-Driven-Transformer-V3), saved as
  `<root>/checkpoints/sdt_v3_s_imagenet.pth` (`setup.ps1` downloads it with `gdown`).

## Usage

Copy `experiments/config.example.json` to `experiments/config.json` and set `root` (data, checkpoints and runs live
there). Then:

```bash
python experiments/run_all.py --smoke     # ~5 min end-to-end check on a few images
python experiments/run_all.py             # run or resume all 31 jobs, then write <root>/runs/report/
python experiments/run_all.py --status    # progress
python experiments/run_all.py --report    # rebuild the report from finished jobs

# deterministic FP32 evaluation of saved checkpoints (the numbers in the paper)
python experiments/eval_saved.py --fp32 --deterministic --save_suffix _fp32 --other_attn --field <PlantDoc root>

# analyses that only need the committed results/
python experiments/compare_runs.py
python experiments/mode_sensitivity.py
python experiments/make_tables.py
python figures/make_paper_figures.py
```

Training takes 0.7–1.2 GPU hours per job on an RTX 3060 (about 31 GPU hours for the whole queue).
On Windows, `experiments/windows/setup.ps1` installs everything and registers a start-up task that resumes the queue
after reboots; see the comments at the top of that script.

## Reproducibility notes

* All reported accuracies come from deterministic FP32 evaluation at batch size 64, which is bit-reproducible
  (0 changed predictions over 27 repeated evaluations).
* Spike quantisation makes predictions sensitive to tiny numerical differences: batch size 1 instead of 64 changes
  4–6 test predictions of LeafSpike-S and 12–14 of LeafSpike-S-sparse (`results/stability.md`).
* Training itself is not bit-reproducible (fp16 mixed precision, data-loader workers).
* Trained checkpoints are not included in this repository because of their size.

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)); bibliographic details will be
updated after publication.

## Acknowledgements

LeafSpike-S builds on E-SpikeFormer / Spike-Driven Transformer V3 by Yao et al. (IEEE TPAMI, 2025); the model code here
is an independent implementation whose module names follow the official one so that its checkpoint can be loaded.
ViT baselines come from [timm](https://github.com/huggingface/pytorch-image-models).

## License

MIT for the code in this repository (see [`LICENSE`](LICENSE)). The datasets and the SDT-v3 checkpoint are not
redistributed here and remain under the terms of their providers.
