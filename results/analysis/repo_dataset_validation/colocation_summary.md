# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)

Internal characterization of the reconstructed tangled dataset only. No externally-labelled tangled reference set is used.

## Table

| split | k | n_rows | %rows same-dir | %rows same-file | %rows same-func | n_pairs | %pairs same-dir | %pairs same-file | %pairs same-func | median gap | IQR gap | %gap<=10 | %gap<=50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2 | 280 | 3.2 | 0.7 | 0.4 | 280 | 3.2 | 0.7 | 0.4 | 598.0 | [307.5, 888.5] | 0.0 | 50.0 |
| train | 3 | 280 | 13.9 | 2.9 | 1.4 | 840 | 6.1 | 1.0 | 0.5 | 63.0 | [17.8, 171.2] | 12.5 | 37.5 |
| train | 4 | 280 | 28.6 | 10.7 | 6.1 | 1680 | 7.0 | 1.9 | 1.0 | 105.0 | [17.0, 330.0] | 18.8 | 34.4 |
| train | 5 | 280 | 39.6 | 8.6 | 3.6 | 2800 | 6.0 | 0.9 | 0.4 | 136.5 | [26.5, 442.0] | 19.2 | 30.8 |
| test | 2 | 70 | 22.9 | 2.9 | 0.0 | 70 | 22.9 | 2.9 | 0.0 | 0.0 | [0.0, 0.0] | 100.0 | 100.0 |
| test | 3 | 70 | 50.0 | 10.0 | 4.3 | 210 | 20.5 | 3.3 | 1.4 | 22.0 | [1.0, 48.5] | 42.9 | 71.4 |
| test | 4 | 70 | 65.7 | 14.3 | 4.3 | 420 | 17.9 | 2.4 | 0.7 | 64.0 | [10.5, 64.0] | 30.0 | 40.0 |
| test | 5 | 70 | 92.9 | 22.9 | 4.3 | 700 | 22.1 | 2.4 | 0.6 | 64.0 | [3.0, 109.0] | 41.2 | 47.1 |

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
Train & 2 & 280 & 3.2 & 0.7 & 0.4 & 3.2 & 0.7 & 0.4 & 598.0 & [307.5, 888.5] & 0.0 & 50.0 \\
Train & 3 & 280 & 13.9 & 2.9 & 1.4 & 6.1 & 1.0 & 0.5 & 63.0 & [17.8, 171.2] & 12.5 & 37.5 \\
Train & 4 & 280 & 28.6 & 10.7 & 6.1 & 7.0 & 1.9 & 1.0 & 105.0 & [17.0, 330.0] & 18.8 & 34.4 \\
Train & 5 & 280 & 39.6 & 8.6 & 3.6 & 6.0 & 0.9 & 0.4 & 136.5 & [26.5, 442.0] & 19.2 & 30.8 \\
Test & 2 & 70 & 22.9 & 2.9 & 0.0 & 22.9 & 2.9 & 0.0 & 0.0 & [0.0, 0.0] & 100.0 & 100.0 \\
Test & 3 & 70 & 50.0 & 10.0 & 4.3 & 20.5 & 3.3 & 1.4 & 22.0 & [1.0, 48.5] & 42.9 & 71.4 \\
Test & 4 & 70 & 65.7 & 14.3 & 4.3 & 17.9 & 2.4 & 0.7 & 64.0 & [10.5, 64.0] & 30.0 & 40.0 \\
Test & 5 & 70 & 92.9 & 22.9 & 4.3 & 22.1 & 2.4 & 0.6 & 64.0 & [3.0, 109.0] & 41.2 & 47.1 \\
\bottomrule
\end{tabular}
\end{table*}
```

## Summary

Across k=2..5 and both splits, 3.2-22.9% of cross-concern pairs touch a common directory, 0.7-3.3% touch a common file, and 0.0-1.4% touch a common function context. Row-level rates (whether at least one pair in the tangled commit co-locates) are higher than pair-level rates, as expected since a row's probability of containing at least one co-locating pair grows with the number of pairs it contains. Among pairs that do share a file, the minimum edit-to-edit line gap has a median in the tens of lines, with 0.0-100.0% of sharing pairs landing within 10 lines of each other across strata. Co-location rates at all three granularities (directory, file, function) generally do not increase monotonically with k, since a larger k spreads the same commit's diff across more constituent atomics without proportionally increasing shared-location edits per pair.
