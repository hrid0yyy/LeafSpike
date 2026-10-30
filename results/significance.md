# Statistical Significance Analysis (Exact McNemar Test with Holm-Bonferroni Correction)

Evaluated on the exact PlantVillage-tomato lab test split (2593 images).
Tests exact two-sided binomial test on discordant pairs ($b$ = LeafSpike correct & other wrong, $c$ = LeafSpike wrong & other correct).
Primary inference: exact McNemar test per seed, with Holm-Bonferroni correction within each model-pair family (3 seeds).
Pooled rows are **descriptive only, not independent** (the same test images are reused across seeds); treat them as a summary, not a second test.

| Model | Baseline | Seed | Errors (SNN/ViT) | Discordant (b/c) | Raw p-value | Holm-Adj p-value | Significant (α=0.05) |
|---|---|---|---|---|---|---|---|
| LeafSpike-S (Final, No Shortcut) | DeiT-S | 42 | 4/1 | 1/4 | 0.3750 | 0.3750 | No |
| LeafSpike-S (Final, No Shortcut) | DeiT-S | 43 | 9/2 | 2/9 | 0.0654 | 0.1309 | No |
| LeafSpike-S (Final, No Shortcut) | DeiT-S | 44 | 8/0 | 0/8 | 0.0078 | 0.0234 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | DeiT-S | **Pooled (3 seeds, descriptive)** | 21/3 | 3/21 | 0.0003 | 0.0003 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | Swin-T | 42 | 4/2 | 2/4 | 0.6875 | 0.6875 | No |
| LeafSpike-S (Final, No Shortcut) | Swin-T | 43 | 9/0 | 0/9 | 0.0039 | 0.0117 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | Swin-T | 44 | 8/0 | 0/8 | 0.0078 | 0.0156 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | Swin-T | **Pooled (3 seeds, descriptive)** | 21/2 | 2/21 | 6.60e-05 | 6.60e-05 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XS | 42 | 4/13 | 10/1 | 0.0117 | 0.0352 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XS | 43 | 9/9 | 4/4 | 1.0000 | 1.0000 | No |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XS | 44 | 8/12 | 10/6 | 0.4545 | 0.9090 | No |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XS | **Pooled (3 seeds, descriptive)** | 21/34 | 24/11 | 0.0410 | 0.0410 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XXS | 42 | 4/38 | 36/2 | 5.40e-09 | 1.62e-08 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XXS | 43 | 9/30 | 24/3 | 4.92e-05 | 4.92e-05 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XXS | 44 | 8/35 | 32/5 | 7.43e-06 | 1.49e-05 | Yes (p<0.05) |
| LeafSpike-S (Final, No Shortcut) | MobileViT-XXS | **Pooled (3 seeds, descriptive)** | 21/103 | 92/10 | 9.40e-18 | 9.40e-18 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | DeiT-S | 42 | 4/1 | 1/4 | 0.3750 | 0.3750 | No |
| LeafSpike-S (With Shortcut) | DeiT-S | 43 | 11/2 | 2/11 | 0.0225 | 0.0674 | No |
| LeafSpike-S (With Shortcut) | DeiT-S | 44 | 5/0 | 0/5 | 0.0625 | 0.1250 | No |
| LeafSpike-S (With Shortcut) | DeiT-S | **Pooled (3 seeds, descriptive)** | 20/3 | 3/20 | 0.0005 | 0.0005 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | Swin-T | 42 | 4/2 | 2/4 | 0.6875 | 0.6875 | No |
| LeafSpike-S (With Shortcut) | Swin-T | 43 | 11/0 | 0/11 | 0.0010 | 0.0029 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | Swin-T | 44 | 5/0 | 0/5 | 0.0625 | 0.1250 | No |
| LeafSpike-S (With Shortcut) | Swin-T | **Pooled (3 seeds, descriptive)** | 20/2 | 2/20 | 0.0001 | 0.0001 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XS | 42 | 4/13 | 9/0 | 0.0039 | 0.0117 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XS | 43 | 11/9 | 6/8 | 0.7905 | 0.7905 | No |
| LeafSpike-S (With Shortcut) | MobileViT-XS | 44 | 5/12 | 10/3 | 0.0923 | 0.1846 | No |
| LeafSpike-S (With Shortcut) | MobileViT-XS | **Pooled (3 seeds, descriptive)** | 20/34 | 25/11 | 0.0288 | 0.0288 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XXS | 42 | 4/38 | 35/1 | 1.08e-09 | 3.23e-09 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XXS | 43 | 11/30 | 27/8 | 0.0019 | 0.0019 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XXS | 44 | 5/35 | 31/1 | 1.54e-08 | 3.07e-08 | Yes (p<0.05) |
| LeafSpike-S (With Shortcut) | MobileViT-XXS | **Pooled (3 seeds, descriptive)** | 20/103 | 93/10 | 5.20e-18 | 5.20e-18 | Yes (p<0.05) |
| LeafSpike-S (No Shortcut) | LeafSpike (With Shortcut) | 42 | 4/4 | 2/2 | 1.0000 | 1.0000 | No |
| LeafSpike-S (No Shortcut) | LeafSpike (With Shortcut) | 43 | 9/11 | 7/5 | 0.7744 | 1.0000 | No |
| LeafSpike-S (No Shortcut) | LeafSpike (With Shortcut) | 44 | 8/5 | 2/5 | 0.4531 | 1.0000 | No |
| LeafSpike-S (No Shortcut) | LeafSpike (With Shortcut) | **Pooled (3 seeds, descriptive)** | 21/20 | 11/12 | 1.0000 | 1.0000 | No |
| LeafSpike-S-sparse (w=10) | MobileViT-XXS | 42 | 12/38 | 32/6 | 2.43e-05 | 7.30e-05 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=10) | MobileViT-XXS | 43 | 12/30 | 23/5 | 0.0009 | 0.0011 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=10) | MobileViT-XXS | 44 | 15/35 | 26/6 | 0.0005 | 0.0011 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=10) | MobileViT-XXS | **Pooled (3 seeds, descriptive)** | 39/103 | 81/17 | 3.62e-11 | 3.62e-11 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=10) | MobileViT-XS | 42 | 12/13 | 9/8 | 1.0000 | 1.0000 | No |
| LeafSpike-S-sparse (w=10) | MobileViT-XS | 43 | 12/9 | 4/7 | 0.5488 | 1.0000 | No |
| LeafSpike-S-sparse (w=10) | MobileViT-XS | 44 | 15/12 | 8/11 | 0.6476 | 1.0000 | No |
| LeafSpike-S-sparse (w=10) | MobileViT-XS | **Pooled (3 seeds, descriptive)** | 39/34 | 21/26 | 0.5601 | 0.5601 | No |
| LeafSpike-S-sparse (w=5) | MobileViT-XXS | 42 | 10/38 | 34/6 | 8.36e-06 | 8.36e-06 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=5) | MobileViT-XXS | **Pooled (1 seed, descriptive)** | 10/38 | 34/6 | 8.36e-06 | 8.36e-06 | Yes (p<0.05) |
| LeafSpike-S-sparse (w=5) | MobileViT-XS | 42 | 10/13 | 10/7 | 0.6291 | 0.6291 | No |
| LeafSpike-S-sparse (w=5) | MobileViT-XS | **Pooled (1 seed, descriptive)** | 10/13 | 10/7 | 0.6291 | 0.6291 | No |


## PlantDoc Zero-Shot Robustness Extra (9 Shared Tomato Classes)
Evaluated 903 images on field data without fine-tuning. Argmax restricted to shared classes.

### Per-Class Field Sample Counts
| Class Name | Image Count |
|---|---|
| bacterial_spot | 110 |
| early_blight | 88 |
| late_blight | 111 |
| leaf_mold | 91 |
| septoria_leaf_spot | 157 |
| spider_mites | 2 |
| yellow_leaf_curl_virus | 238 |
| mosaic_virus | 54 |
| healthy | 52 |

### Field Accuracy and Macro-F1
| Model | Field Accuracy (%) | Field Macro-F1 (%) |
|---|---|---|
| deit_s | 37.91 ± 1.30 | 22.80 ± 0.33 |
| swin_t | 40.75 ± 1.94 | 25.22 ± 1.24 |
| mobilevit_xs | 34.88 ± 2.58 | 21.10 ± 1.04 |
| mobilevit_xxs | 36.25 ± 2.62 | 24.14 ± 1.30 |
| leafspike | 39.31 ± 0.84 | 24.09 ± 0.98 |
| leafspike_integer_attn | 38.54 | 23.55 |
| leafspike_no_maxshort | 37.73 ± 0.85 | 23.07 ± 1.29 |
| leafspike_D2 | 35.22 | 22.68 |
| leafspike_D1 | 20.60 | 5.94 |
| leafspike_firereg | 40.86 | 25.38 |
| leafspike_D2_firereg | 33.70 ± 1.96 | 21.46 ± 2.06 |
| leafspike_sparse_w10 | 36.99 | 23.33 |
| leafspike_sparse_w30 | 36.21 | 25.37 |
| leafspike_final_sparse_w10 | 39.39 ± 1.95 | 25.26 ± 1.66 |
| leafspike_final_sparse_w5 | 38.76 | 23.35 |
