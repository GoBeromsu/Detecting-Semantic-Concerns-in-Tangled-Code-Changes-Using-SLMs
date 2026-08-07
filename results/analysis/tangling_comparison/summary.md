# Synthetic vs Reference Set Structural Comparison

Set 2 is 110 commits removed from the original CCS dataset during curation because they appeared to contain multiple concerns. This is an **indicative reference set** (single-annotator flag, not independently validated), following the reviewer's suggested comparison design (files / functions / lines). What follows is a **structural comparability check**, not a claim that Set 2 validates Set 1's realism or constitutes ground truth.

## Set 2 (reference set) construction
- real_tangled_shas.csv total rows: 110
- no reason/category column exists in real_tangled_shas.csv (columns = ['sha'] only) - all 110 rows kept, 0 dropped
- join to CCS Dataset.csv on sha: matched 110, unmatched 0

## Circularity control
- 263 synthetic (Set 1) rows share >=1 SHA with Set 2 - these are excluded from the disjoint variant (Set 1') reported below

## Mann-Whitney U + Cliff's delta - full Set 1 (as originally, includes rows overlapping with Set 2)

| metric | comparison | n_synth | median_synth | IQR_synth | n_real | median_real | IQR_real | U | p_raw | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Files changed | k=2 vs real | 350 | 3.00 | [2.00, 6.00] | 110 | 3.00 | [2.00, 7.00] | 20614.5 | 0.2533 | 1 | -0.071 | negligible |
| Files changed | k>=2 pooled vs real | 1400 | 7.00 | [4.00, 11.00] | 110 | 3.00 | [2.00, 7.00] | 111446.5 | 4.251e-15 | 6.802e-14 | -0.447 | medium |
| Hunks per commit | k=2 vs real | 350 | 6.00 | [3.00, 14.00] | 110 | 7.00 | [3.25, 13.00] | 18651.5 | 0.6218 | 1 | 0.031 | negligible |
| Hunks per commit | k>=2 pooled vs real | 1400 | 13.00 | [7.00, 22.00] | 110 | 7.00 | [3.25, 13.00] | 101087.5 | 4.405e-08 | 6.167e-07 | -0.313 | small |
| Hunks per file | k=2 vs real | 350 | 1.55 | [1.00, 2.28] | 110 | 1.75 | [1.12, 3.00] | 16922.5 | 0.05368 | 0.5905 | 0.121 | negligible |
| Hunks per file | k>=2 pooled vs real | 1400 | 1.67 | [1.26, 2.33] | 110 | 1.75 | [1.12, 3.00] | 71473.5 | 0.2088 | 1 | 0.072 | negligible |
| Distinct function contexts | k=2 vs real | 350 | 4.00 | [2.00, 7.00] | 110 | 3.00 | [2.00, 7.00] | 19502.0 | 0.8344 | 1 | -0.013 | negligible |
| Distinct function contexts | k>=2 pooled vs real | 1400 | 7.00 | [4.00, 13.00] | 110 | 3.00 | [2.00, 7.00] | 105997.5 | 4.095e-11 | 6.143e-10 | -0.377 | medium |
| Function-context coverage | k=2 vs real | 350 | 0.88 | [0.67, 1.00] | 110 | 0.92 | [0.69, 1.00] | 17808.0 | 0.2209 | 1 | 0.075 | negligible |
| Function-context coverage | k>=2 pooled vs real | 1400 | 0.85 | [0.67, 0.99] | 110 | 0.92 | [0.69, 1.00] | 65645.5 | 0.009246 | 0.111 | 0.147 | small |
| Within-function co-editing (0/1) | k=2 vs real | 350 | 1.00 | [0.00, 1.00] | 110 | 1.00 | [0.00, 1.00] | 18050.0 | 0.253 | 1 | 0.062 | negligible |
| Within-function co-editing (0/1) | k>=2 pooled vs real | 1400 | 1.00 | [0.00, 1.00] | 110 | 1.00 | [0.00, 1.00] | 83475.0 | 0.07127 | 0.7127 | -0.084 | negligible |
| Median line-gap (same file) | k=2 vs real | 262 | 34.50 | [20.00, 81.88] | 86 | 34.00 | [21.62, 58.25] | 11600.0 | 0.6803 | 1 | -0.030 | negligible |
| Median line-gap (same file) | k>=2 pooled vs real | 1220 | 35.50 | [22.00, 76.00] | 86 | 34.00 | [21.62, 58.25] | 54942.0 | 0.4629 | 1 | -0.047 | negligible |
| Total changed lines (+/-) | k=2 vs real | 350 | 62.00 | [22.00, 131.00] | 110 | 106.00 | [33.50, 225.75] | 15029.5 | 0.0005203 | 0.006764 | 0.219 | small |
| Total changed lines (+/-) | k>=2 pooled vs real | 1400 | 137.00 | [56.00, 259.25] | 110 | 106.00 | [33.50, 225.75] | 84830.0 | 0.0754 | 0.7127 | -0.102 | negligible |

## Mann-Whitney U + Cliff's delta - disjoint Set 1' (circularity-excluded)

| metric | comparison | n_synth | median_synth | IQR_synth | n_real | median_real | IQR_real | U | p_raw | p_holm | cliffs_delta | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Files changed | k=2 vs real | 318 | 3.00 | [2.00, 5.00] | 110 | 3.00 | [2.00, 7.00] | 18268.5 | 0.4777 | 1 | -0.045 | negligible |
| Files changed | k>=2 pooled vs real | 1154 | 6.00 | [4.00, 10.00] | 110 | 3.00 | [2.00, 7.00] | 89210.0 | 1.624e-12 | 2.598e-11 | -0.406 | medium |
| Hunks per commit | k=2 vs real | 318 | 6.00 | [3.00, 14.00] | 110 | 7.00 | [3.25, 13.00] | 16521.5 | 0.385 | 1 | 0.055 | negligible |
| Hunks per commit | k>=2 pooled vs real | 1154 | 12.00 | [6.00, 21.00] | 110 | 7.00 | [3.25, 13.00] | 80352.5 | 3.855e-06 | 5.397e-05 | -0.266 | small |
| Hunks per file | k=2 vs real | 318 | 1.50 | [1.00, 2.33] | 110 | 1.75 | [1.12, 3.00] | 15387.5 | 0.05792 | 0.6371 | 0.120 | negligible |
| Hunks per file | k>=2 pooled vs real | 1154 | 1.62 | [1.25, 2.25] | 110 | 1.75 | [1.12, 3.00] | 57832.0 | 0.1225 | 1 | 0.089 | negligible |
| Distinct function contexts | k=2 vs real | 318 | 3.00 | [2.00, 7.00] | 110 | 3.00 | [2.00, 7.00] | 17335.5 | 0.8892 | 1 | 0.009 | negligible |
| Distinct function contexts | k>=2 pooled vs real | 1154 | 6.00 | [3.00, 12.00] | 110 | 3.00 | [2.00, 7.00] | 84797.5 | 5.01e-09 | 7.514e-08 | -0.336 | medium |
| Function-context coverage | k=2 vs real | 318 | 0.88 | [0.67, 1.00] | 110 | 0.92 | [0.69, 1.00] | 16200.5 | 0.2337 | 1 | 0.074 | negligible |
| Function-context coverage | k>=2 pooled vs real | 1154 | 0.84 | [0.67, 1.00] | 110 | 0.92 | [0.69, 1.00] | 53837.0 | 0.007817 | 0.09381 | 0.152 | small |
| Within-function co-editing (0/1) | k=2 vs real | 318 | 1.00 | [0.00, 1.00] | 110 | 1.00 | [0.00, 1.00] | 16230.0 | 0.1922 | 1 | 0.072 | negligible |
| Within-function co-editing (0/1) | k>=2 pooled vs real | 1154 | 1.00 | [0.00, 1.00] | 110 | 1.00 | [0.00, 1.00] | 66940.0 | 0.2537 | 1 | -0.055 | negligible |
| Median line-gap (same file) | k=2 vs real | 236 | 32.25 | [19.00, 79.00] | 86 | 34.00 | [21.62, 58.25] | 10136.5 | 0.9881 | 1 | 0.001 | negligible |
| Median line-gap (same file) | k>=2 pooled vs real | 990 | 35.00 | [21.00, 76.88] | 86 | 34.00 | [21.62, 58.25] | 43714.0 | 0.6791 | 1 | -0.027 | negligible |
| Total changed lines (+/-) | k=2 vs real | 318 | 57.00 | [20.25, 124.75] | 110 | 106.00 | [33.50, 225.75] | 13199.0 | 0.0001246 | 0.00162 | 0.245 | small |
| Total changed lines (+/-) | k>=2 pooled vs real | 1154 | 120.50 | [52.00, 235.00] | 110 | 106.00 | [33.50, 225.75] | 66553.5 | 0.3993 | 1 | -0.049 | negligible |

## Do conclusions change between the full and disjoint variants?

- No metric/comparison cell changes magnitude label or Holm-corrected significance (p_holm < 0.05) between the full and disjoint variants - excluding the 263 circularity-overlapping rows does not change any conclusion.
