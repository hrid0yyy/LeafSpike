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
