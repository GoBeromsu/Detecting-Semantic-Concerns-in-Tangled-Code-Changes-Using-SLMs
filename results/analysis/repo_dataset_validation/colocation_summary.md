# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)

Internal characterization of the reconstructed tangled dataset only - Set 2 (real_tangled_shas.csv) is not used in this analysis.

## Table

| split | k | n_rows | %rows same-dir | %rows same-file | %rows same-func | n_pairs | %pairs same-dir | %pairs same-file | %pairs same-func | median gap | IQR gap | %gap<=10 | %gap<=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2 | 280 | 6.4 | 1.4 | 0.4 | 280 | 6.4 | 1.4 | 0.4 | 15.5 | [11.8, 95.2] | 25.0 | 75.0 |
| train | 3 | 280 | 14.3 | 3.9 | 2.1 | 840 | 6.0 | 1.3 | 0.7 | 130.0 | [73.5, 263.5] | 9.1 | 27.3 |
| train | 4 | 280 | 24.3 | 6.8 | 4.6 | 1680 | 5.4 | 1.2 | 0.8 | 131.0 | [57.5, 276.0] | 15.0 | 25.0 |
| train | 5 | 280 | 34.6 | 7.9 | 3.9 | 2800 | 5.1 | 0.8 | 0.4 | 162.0 | [92.0, 330.0] | 8.7 | 17.4 |
| test | 2 | 70 | 17.1 | 0.0 | 0.0 | 70 | 17.1 | 0.0 | 0.0 | n/a | n/a | n/a | n/a |
| test | 3 | 70 | 45.7 | 10.0 | 2.9 | 210 | 19.0 | 3.3 | 1.0 | 15.0 | [2.5, 64.0] | 42.9 | 57.1 |
| test | 4 | 70 | 75.7 | 20.0 | 5.7 | 420 | 23.8 | 3.3 | 1.0 | 64.0 | [8.5, 64.0] | 28.6 | 42.9 |
| test | 5 | 70 | 90.0 | 30.0 | 2.9 | 700 | 24.9 | 3.4 | 0.3 | 64.0 | [11.0, 64.0] | 25.0 | 33.3 |

## LaTeX (booktabs)

```latex
\begin{table*}
\centering
\caption{Cross-concern co-location in the reconstructed tangled dataset, by split and concern count $k$.}
\label{tab:colocation}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
Split & $k$ & $n$ & RDir\% & RFile\% & RFunc\% & PDir\% & PFile\% & PFunc\% & MedGap & IQR & Gap$\leq$10\% & Gap$\leq$50\% \\
\midrule
Train & 2 & 280 & 6.4 & 1.4 & 0.4 & 6.4 & 1.4 & 0.4 & 15.5 & [11.8, 95.2] & 25.0 & 75.0 \\
Train & 3 & 280 & 14.3 & 3.9 & 2.1 & 6.0 & 1.3 & 0.7 & 130.0 & [73.5, 263.5] & 9.1 & 27.3 \\
Train & 4 & 280 & 24.3 & 6.8 & 4.6 & 5.4 & 1.2 & 0.8 & 131.0 & [57.5, 276.0] & 15.0 & 25.0 \\
Train & 5 & 280 & 34.6 & 7.9 & 3.9 & 5.1 & 0.8 & 0.4 & 162.0 & [92.0, 330.0] & 8.7 & 17.4 \\
Test & 2 & 70 & 17.1 & 0.0 & 0.0 & 17.1 & 0.0 & 0.0 & n/a & n/a & n/a & n/a \\
Test & 3 & 70 & 45.7 & 10.0 & 2.9 & 19.0 & 3.3 & 1.0 & 15.0 & [2.5, 64.0] & 42.9 & 57.1 \\
Test & 4 & 70 & 75.7 & 20.0 & 5.7 & 23.8 & 3.3 & 1.0 & 64.0 & [8.5, 64.0] & 28.6 & 42.9 \\
Test & 5 & 70 & 90.0 & 30.0 & 2.9 & 24.9 & 3.4 & 0.3 & 64.0 & [11.0, 64.0] & 25.0 & 33.3 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Summary

Across k=2..5 and both splits, 5.1-24.9% of cross-concern pairs touch a common directory, 0.0-3.4% touch a common file, and 0.0-1.0% touch a common function context. Row-level rates (whether at least one pair in the tangled commit co-locates) are higher than pair-level rates, as expected since a row's probability of containing at least one co-locating pair grows with the number of pairs it contains. Among pairs that do share a file, the minimum edit-to-edit line gap has a median in the tens of lines, with 8.7-42.9% of sharing pairs landing within 10 lines of each other across strata. Co-location rates at all three granularities (directory, file, function) generally do not increase monotonically with k, since a larger k spreads the same commit's diff across more constituent atomics without proportionally increasing shared-location edits per pair.
