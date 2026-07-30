# Tangled Dataset Combination-Coverage Audit

This report is generated read-only from the committed pool, split, and tangled CSVs.
Joined-token metrics re-encode `''.join(json.loads(diff))` with `cl100k_base`.

## C(7,k) combination coverage

### Train
| k | covered / C(7,k) |
| --- | --- |
| 1 | 7/7 |
| 2 | 21/21 |
| 3 | 35/35 |
| 4 | 35/35 |
| 5 | 21/21 |

### Test
| k | covered / C(7,k) |
| --- | --- |
| 1 | 7/7 |
| 2 | 21/21 |
| 3 | 29/35 |
| 4 | 33/35 |
| 5 | 20/21 |

## Missing test-split combinations: reachability and cause attribution

Reachability is necessary but **not sufficient** to call a gap a sampler defect.
A combo is *unreachable* when no single test repo can supply a <=12288-token witness — then the pool, not the sampler, is the cause.
A *reachable* combo that produced no row is only a defect if the sampler had enough draws to be expected to hit it; see the draw-budget section below.

| Combination | repos with all types | feasible repos | witness joined tokens | binding constraint | verdict |
| --- | --- | --- | --- | --- | --- |
| (build, ci, test) | 4 | 4 | 676 | none: a <= 12288-token single-repo witness exists | reachable |
| (build, feat, fix) | 9 | 9 | 741 | none: a <= 12288-token single-repo witness exists | reachable |
| (build, feat, test) | 10 | 10 | 777 | none: a <= 12288-token single-repo witness exists | reachable |
| (ci, docs, test) | 4 | 4 | 570 | none: a <= 12288-token single-repo witness exists | reachable |
| (docs, fix, refactor) | 7 | 7 | 452 | none: a <= 12288-token single-repo witness exists | reachable |
| (feat, refactor, test) | 12 | 12 | 930 | none: a <= 12288-token single-repo witness exists | reachable |
| (build, feat, fix, refactor) | 8 | 8 | 1055 | none: a <= 12288-token single-repo witness exists | reachable |
| (build, fix, refactor, test) | 9 | 9 | 795 | none: a <= 12288-token single-repo witness exists | reachable |
| (build, ci, feat, fix, refactor) | 3 | 3 | 2363 | none: a <= 12288-token single-repo witness exists | reachable |

**Reachability:** 9 reachable; 0 unreachable (pool artifact).

## Draw budget: is the gap explained without a sampler defect?

Under a correct sampler, each concern count draws independently, so the expected number of still-uncovered combinations after `draws` draws is `C * (1 - 1/C)^draws` (coupon collector). `draws needed` is the expected number required to cover every combination, `C * H(C)`.

| split | k | combos C | draws | expected uncovered | observed uncovered | draws needed |
| --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 7 | 280 | 0.00 | 0 | 18 |
| train | 2 | 21 | 280 | 0.00 | 0 | 77 |
| train | 3 | 35 | 280 | 0.01 | 0 | 145 |
| train | 4 | 35 | 280 | 0.01 | 0 | 145 |
| train | 5 | 21 | 280 | 0.00 | 0 | 77 |
| test | 1 | 7 | 70 | 0.00 | 0 | 18 |
| test | 2 | 21 | 70 | 0.69 | 0 | 77 |
| test | 3 | 35 | 70 | 4.60 | 6 | 145 |
| test | 4 | 35 | 70 | 4.60 | 2 | 145 |
| test | 5 | 21 | 70 | 0.69 | 1 | 77 |

**Totals:** expected 10.60 uncovered, observed 9.

**Cause verdict:** the observed gap tracks the draw-budget prediction, so it is a **draw-budget shortfall, not a sampler defect**. The test split draws only 70 times per concern count while covering C(7,3)=35 combinations needs ~145 draws in expectation; the train split's 280 draws clear that bar, which is exactly why train reaches full coverage and test does not. The generator targets marginal per-type uniformity (achieved exactly) and never promises joint C(7,k) coverage.

## Atomic-commit reuse

### Train
Used atomics: 881; mean: 4.77; median: 3.00; max: 79.
| SHA | reuse count |
| --- | --- |
| 7a28982e2ab8e4570aef78285f66e763de41104e | 79 |
| 625542c31043573c74271c28bcfdc504cc5f636b | 78 |
| 2ac99c0a66a1adc18ee4ef660608f814823dd198 | 56 |
| c81e6a1f4839502227f920c830a28a8b712de2d5 | 54 |
| ca41994bcffb835ceeba816c6787a07ccffbe37d | 51 |
| 1fb55e83b58354e8449ed0b6353e591f4c47e779 | 46 |
| 60ac03c08f942a8dda49b9f9f7d2ce7a63535414 | 40 |
| 6aa63c9b8d4dcdbb401743adc3c9a1020d943250 | 38 |
| 411be831591b2ea15ca9138eaf8db81f51b5101e | 35 |
| c5e86dbc00a13a355bffadeb2db197e2fea5640f | 35 |

### Test
Used atomics: 208; mean: 5.05; median: 2.00; max: 47.
| SHA | reuse count |
| --- | --- |
| 6f5b1103c0893d13b42f1f7c6504c9c339840be3 | 47 |
| be44bbdeb52c6025e81c99528ed7b0d932c5be18 | 45 |
| 45bbdaba1490a3216decb2c1a88b8f7041a3d505 | 39 |
| b7e38fb62aa6e8a30d72dec063b1adccd089d0aa | 35 |
| 0e4ede620dd9a2cdc13ab9cc6da05577b39fddb8 | 32 |
| 3730075261bc81df68f9d10f12a2d6bbffd8493c | 32 |
| ccd1946a375e2ac10c3c591b04a435c4bc5170c2 | 31 |
| 625cbbca0d92b8756eac6fcacc795d90527d8975 | 30 |
| 73211952964a79d97b434dd567e6d7d34be7feb5 | 29 |
| 076b9f5efc109b481074a81957e24ce9f4a69f08 | 28 |

## Per-type pool supply versus realized demand

### Train
| type | pool atomics | realized labels |
| --- | --- | --- |
| build | 96 | 600 |
| ci | 104 | 600 |
| docs | 104 | 600 |
| feat | 123 | 600 |
| fix | 92 | 600 |
| refactor | 351 | 600 |
| test | 135 | 600 |

### Test
| type | pool atomics | realized labels |
| --- | --- | --- |
| build | 31 | 150 |
| ci | 23 | 150 |
| docs | 37 | 150 |
| feat | 32 | 150 |
| fix | 42 | 150 |
| refactor | 98 | 150 |
| test | 41 | 150 |

## Joined-token distribution (cl100k_base)

### Train
| statistic | tokens |
| --- | --- |
| P50 | 2363.00 |
| P90 | 8031.30 |
| P95 | 10030.40 |
| P99 | 11726.35 |
| max | 12269 |

### Test
| statistic | tokens |
| --- | --- |
| P50 | 2500.50 |
| P90 | 8854.60 |
| P95 | 10073.00 |
| P99 | 11733.11 |
| max | 12221 |
