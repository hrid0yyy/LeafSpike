# Mode sensitivity: spike counts among firing neurons

Descriptive only, computed from the saved `result.json` layer statistics (no retraining).
For every q/k/v spike layer, the mean spike count among **firing** neurons is `mean_spikes_per_neuron / frac_nonzero`; the value for a model is the unweighted mean over its q/k/v layers, then over seeds.

| Model | Seed | q | k | v | All |
|---|---:|---:|---:|---:|---:|
| `leafspike_no_maxshort` | 42 | 2.047 | 1.778 | 1.827 | 1.884 |
| `leafspike_no_maxshort` | 43 | 2.046 | 1.773 | 1.822 | 1.880 |
| `leafspike_no_maxshort` | 44 | 2.054 | 1.778 | 1.820 | 1.884 |
| `leafspike_final_sparse_w10` | 42 | 1.875 | 1.871 | 1.971 | 1.906 |
| `leafspike_final_sparse_w10` | 43 | 1.879 | 1.863 | 1.967 | 1.903 |
| `leafspike_final_sparse_w10` | 44 | 1.880 | 1.879 | 1.975 | 1.911 |

| Model | q (mean ± std) | k | v | All |
|---|---:|---:|---:|---:|
| `leafspike_no_maxshort` | 2.049 ± 0.004 | 1.777 ± 0.003 | 1.823 ± 0.004 | 1.883 ± 0.002 |
| `leafspike_final_sparse_w10` | 1.878 ± 0.003 | 1.871 ± 0.008 | 1.971 ± 0.004 | 1.907 ± 0.004 |
