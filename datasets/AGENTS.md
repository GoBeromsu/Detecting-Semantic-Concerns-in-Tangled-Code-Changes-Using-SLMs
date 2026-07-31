# datasets/ — 3-STAGE TANGLED-COMMIT PIPELINE

Builds the multi-concern commit dataset. Run stages in order; each consumes the prior stage's output.

## Pipeline (`scripts/`)
| Stage | Script | In → Out |
|-------|--------|----------|
| 1 | `build_repo_pool.py` | `data/CCS Dataset.csv` → `data/repo_grouped_pool.csv` |
| 2 | `generate_repo_tangled.py` | pool → `data/tangled_ccs_dataset_{train,test}.csv` + `data/repo_split.json` |
| 3 | `validate_repo_dataset.py` | all of the above → PASS/FAIL report (+ dist charts) |
| upload | `upload_to_huggingface.py` | validated CSVs → HF dataset |

Stage 1: taxonomy remap (style/perf→refactor, chore dropped → 7 types), token filter ≤12288 (cl100k), repo eligibility ≥5 types.
Stage 2: **intra-repo tangling only** (no cross-repo — avoids style leakage), seeded reuse-weighted selection, repo-disjoint 80/20 split.
Stage 3: asserts repo-disjointness, per-type uniformity, exact per-count row counts, no duplicate SHA-sets, real tiktoken budget re-check.

Other scripts (`analyze_token_distribution`, `concern_token_boxplot`, `analyze_colocation`, `compare_synthetic_vs_real_tangled`, `audit_combination_coverage`, `verify_hf_sync`, plot_*) are diagnostics — not part of the build chain.

## Invariants (assertion boundaries — scripts fail loudly)
- `SEED=42`, `MAX_TOKENS=12288`, 7 types, concern counts 1–5.
- Train **1400** rows (280/count, 21 repos), Test **350** (70/count, 15 repos).
- Per-type uniformity 0.00pp deviation; train/test repo sets disjoint.
- `PINNED_TRAIN_REPO="camunda/zeebe"`.
- Note: the 27B path (`RQ/SLM/unsloth/`) further drops train row 1326 → 1399; that exclusion lives there, NOT here.

## Conventions
- CSV columns `types` and `shas` are **JSON-encoded** — parse with `json.loads`, not as plain strings.
- Stage 2 backs up prior train/test CSVs to `data/legacy/` before overwrite.

## DO NOT
- Hand-edit `data/*.csv` — regenerate via the owning stage script.
- Apply manual quality filtering beyond the token budget (by design).
