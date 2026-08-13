# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)

Internal characterization of the reconstructed tangled dataset only. No externally-labelled tangled reference set is used.

## Table

| split | k | n_rows | n_pairs | %rows same-file | %rows same-dir | %pairs same-file | %pairs same-dir |
|---|---|---|---|---|---|---|---|
| train | 2 | 280 | 280 | 0.7 | 3.2 | 0.7 | 3.2 |
| train | 3 | 280 | 840 | 2.9 | 13.9 | 1.0 | 6.1 |
| train | 4 | 280 | 1680 | 10.7 | 28.6 | 1.9 | 7.0 |
| train | 5 | 280 | 2800 | 8.6 | 39.6 | 0.9 | 6.0 |
| train | overall | 1120 | 5600 | 5.7 | 21.3 | 1.2 | 6.2 |
| test | 2 | 70 | 70 | 2.9 | 22.9 | 2.9 | 22.9 |
| test | 3 | 70 | 210 | 10.0 | 50.0 | 3.3 | 20.5 |
| test | 4 | 70 | 420 | 14.3 | 65.7 | 2.4 | 17.9 |
| test | 5 | 70 | 700 | 22.9 | 92.9 | 2.4 | 22.1 |
| test | overall | 280 | 1400 | 12.5 | 57.9 | 2.6 | 20.6 |
| all | overall | 1400 | 7000 | 7.1 | 28.6 | 1.5 | 9.1 |

## Summary

Over the 1400 multi-concern commits (train+test combined), 7.1% have at least one concern pair sharing a file and 28.6% have at least one pair sharing a folder. Across k=2..5 and both splits, 3.2-22.9% of cross-concern pairs touch a common folder and 0.7-3.3% touch a common file. Row-level rates (whether at least one pair in the tangled commit co-locates) are higher than pair-level rates, as expected since a row's probability of containing at least one co-locating pair grows with the number of pairs it contains. Co-location rates at both granularities (file, folder) generally do not increase monotonically with k, since a larger k spreads the same commit's diff across more constituent atomics without proportionally increasing shared-location edits per pair.
