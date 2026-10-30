# Prediction Stability Report

All accuracies in the paper come from the deterministic FP32 evaluation (`test_predictions_fp32.csv`). This report quantifies how stable those predictions are.

## 1. Determinism: FP32 batch 64, run 1 vs run 2

- Jobs evaluated twice: **27**
- Jobs with any changed prediction: **0**
- Total changed predictions: **0**

## 2. Batch-size sensitivity: batch 64 vs batch 1

- Jobs tested: **6** (leafspike_final_sparse_w10_s42, leafspike_final_sparse_w10_s43, leafspike_final_sparse_w10_s44, leafspike_no_maxshort_s42, leafspike_no_maxshort_s43, leafspike_no_maxshort_s44)
  - `leafspike_final_sparse_w10_s42`: 13 predictions change, errors 12 (bs64, 99.54%) -> 13 (bs1, 99.50%)
  - `leafspike_final_sparse_w10_s43`: 14 predictions change, errors 12 (bs64, 99.54%) -> 18 (bs1, 99.31%)
  - `leafspike_final_sparse_w10_s44`: 12 predictions change, errors 15 (bs64, 99.42%) -> 17 (bs1, 99.34%)
  - `leafspike_no_maxshort_s42`: 4 predictions change, errors 4 (bs64, 99.85%) -> 8 (bs1, 99.69%)
  - `leafspike_no_maxshort_s43`: 4 predictions change, errors 9 (bs64, 99.65%) -> 9 (bs1, 99.65%)
  - `leafspike_no_maxshort_s44`: 6 predictions change, errors 8 (bs64, 99.69%) -> 8 (bs1, 99.69%)

## 3. Precision: FP32 vs fp16 training-time evaluation

- Jobs with both files: **27**
- Total changed predictions: **1177**
- Jobs with more than 2 changes: **14**

| Job | FP32 errors | fp16 errors | Changed predictions |
|---|---|---|---|
| `leafspike_D1_s42` | 1848 | 1843 | 896 |
| `leafspike_D2_firereg_s42` | 43 | 55 | 53 |
| `leafspike_D2_firereg_s43` | 57 | 57 | 52 |
| `leafspike_sparse_w30_s42` | 52 | 60 | 40 |
| `leafspike_D2_firereg_s44` | 54 | 38 | 33 |
| `leafspike_D2_s42` | 41 | 39 | 31 |
| `leafspike_sparse_w10_s42` | 16 | 24 | 20 |
| `leafspike_integer_attn_s42` | 8 | 9 | 9 |
| `leafspike_s44` | 5 | 8 | 9 |
| `leafspike_no_maxshort_s44` | 8 | 8 | 6 |
| `leafspike_s42` | 4 | 6 | 6 |
| `leafspike_s43` | 11 | 13 | 6 |
| `leafspike_firereg_s42` | 5 | 6 | 5 |
| `leafspike_no_maxshort_s43` | 9 | 6 | 5 |
| `leafspike_no_maxshort_s42` | 4 | 4 | 2 |
| `mobilevit_xxs_s43` | 30 | 31 | 2 |
| `mobilevit_xs_s42` | 13 | 14 | 1 |
| `mobilevit_xs_s43` | 9 | 8 | 1 |
| `deit_s_s44` | 0 | 0 | 0 |
| `deit_s_s43` | 2 | 2 | 0 |
| `deit_s_s42` | 1 | 1 | 0 |
| `mobilevit_xs_s44` | 12 | 12 | 0 |
| `mobilevit_xxs_s42` | 38 | 38 | 0 |
| `mobilevit_xxs_s44` | 35 | 35 | 0 |
| `swin_t_s42` | 2 | 2 | 0 |
| `swin_t_s43` | 0 | 0 | 0 |
| `swin_t_s44` | 0 | 0 | 0 |
