---
license: mit
task_categories:
  - text-generation
  - text-classification
language:
  - en
tags:
  - code
  - git
  - commits
  - software-engineering
  - concern-separation
size_categories:
  - 1K<n<10K
---

# Detecting Multiple Semantic Concerns in Tangled Code Commits using Small Language Models

This dataset contains commit data for training and evaluating models on software engineering tasks, specifically focusing on identifying and separating concerns in multi-concern commits.

Every tangled (multi-concern) commit in this dataset is composed exclusively of atomic commits from a single repository — resolving a structural weakness in earlier cross-repo tangles (which were trivially separable by repo-specific coding style, file layout, and tooling) — while per-type concern labels are exactly uniform within each split (1/7 each) and matched across train/test, and the train/test split is repo-disjoint (no repo appears in both splits).

## Dataset Description

This dataset is structured in two layers: **Atomic Commits** and **Tangled Commits**.

### 1. Atomic Commits (`original`)

- **File**: `data/repo_grouped_pool.csv`
- **Records**: 1,309 individual atomic commits with single concerns, across 36 repositories
- **Source**: Filtered from the [CCS Dataset](https://huggingface.co/datasets/0x404/ccs_dataset) (2,000 commits) — see the sampling funnel below
- **Description**: Repo-grouped pool of eligible atomic commits used as the source material for intra-repo tangling
- **Features**:
  - `repo`: Source repository (`owner/name`, derived from `commit_url`)
  - `annotated_type`: The type of concern/change in the commit (post-taxonomy remap; see below)
  - `masked_commit_message`: Commit message with sensitive information masked
  - `git_diff`: The actual code changes in diff format
  - `sha`: Git commit SHA hash
  - `token_count`: cl100k (GPT-4/tiktoken) token count of `git_diff` + `masked_commit_message`

### 2. Tangled Commits

Artificially generated multi-concern commits by combining atomic commits **from the same repository only**. Split into training and test sets, which are **repo-disjoint** (no repository contributes to both splits).

#### 2.1. Training Set (`train`)

- **File**: `data/tangled_ccs_dataset_train.csv`
- **Records**: 1,400 multi-concern commits, drawn from 21 repositories
- **Description**: Training dataset for model development
- **Features**:
  - `commit_message`: Combined commit messages of all concerns
  - `diff`: JSON string containing array of diffs for each concern
  - `concern_count`: Number of individual concerns combined (1-5)
  - `shas`: JSON string containing array of original commit SHAs
  - `types`: JSON string containing array of concern types
  - `repo`: The single repository all of this row's atomic commits were drawn from

#### 2.2. Test Set (`test`)

- **File**: `data/tangled_ccs_dataset_test.csv`
- **Records**: 350 multi-concern commits, drawn from 15 repositories
- **Description**: Test dataset for evaluation, generated separately from training data, from a disjoint set of repositories
- **Features**: Same as training set

## Dataset Statistics

### Sampling Funnel

The atomic commit pool is built by filtering the raw CCS Dataset down to repositories that can support every concern count (1-5) with distinct types:

| Stage | Commits | Repos | Change |
|-------|--------:|------:|--------|
| Raw CCS Dataset | 2,000 | 132 | — |
| Taxonomy remap (`style`/`perf` → `refactor`; `chore` dropped) | 1,800 | 127 | −200 commits |
| Token filter (diff + message ≤ 12,288 cl100k tokens) | 1,708 | 125 | −92 commits |
| Repo eligibility (≥5 distinct types per repo) | 1,309 | 36 | −399 commits, −89 repos |

The ≥5-type eligibility rule is a design necessity, not an ad-hoc quality filter: a repo must offer at least 5 distinct concern types to host a 5-concern intra-repo tangled commit, so every eligible repo can host every concern count (1-5) without a concern-count × repo confound.

Per-type supply in the final pool (1,309 commits, 36 repos):

| Type | feat | fix | refactor | test | docs | build | ci |
|------|-----:|----:|---------:|-----:|-----:|------:|---:|
| Count | 155 | 134 | 449 | 176 | 141 | 127 | 127 |

**Note**: No manual quality-exclusion list is applied anywhere in this pipeline — the only commit-level filter is the 12,288-token model-context limit above.

### Final Dataset

| Split | Repos | Rows | Labels/type | Per-concern-count rows |
|-------|------:|-----:|------------:|------------------------|
| Train | 21 | 1,400 | 600 | 280 × 5 concern counts (1-5) |
| Test | 15 | 350 | 150 | 70 × 5 concern counts (1-5) |

Per-type label share deviates **0.00pp** from the uniform target (1/7 ≈ 14.29%) in both splits, and the train/test per-type share gap is **0.00pp**.

### Concern Type Distribution

The dataset includes 7 conventional commit types:

- `feat`: New features
- `fix`: Bug fixes
- `refactor`: Code restructuring (also absorbs the legacy `style` and `perf` types)
- `test`: Test modifications
- `docs`: Documentation updates
- `build`: Build system changes
- `ci`: CI/CD configuration changes

## Construction Methodology

1. **Intra-repo tangling**: every tangled commit's atomic commits are drawn from exactly one repository, recorded in the `repo` column.
2. **Repo-disjoint 80/20 split**: repositories (not individual commits) are partitioned into train/test via a greedy, per-type-supply-balanced assignment that targets ~20% of each type's total supply landing in test. The largest repository (`camunda/zeebe`, 307 commits) is pinned to train. This partition is deterministic given the pool CSV's row order (no randomness involved in repo assignment) and is recorded in `data/repo_split.json` alongside each split's per-type commit supply.
3. **Adaptive seeded randomization** (`SEED=42`): within each split, tangled commits are generated by repeatedly picking the currently most under-represented concern types (random tie-breaks), a uniformly random supporting repository that contains all of those types, and atomic commits via reuse-weighted random choice (weight ∝ 1/(1 + times already used)). This is *not* a deterministic quota assignment — draws stay random — yet it converges to **exact** 1/7 per-type uniformity in both splits.
4. **Reuse-weighted atomic selection**: atomic commits may be reused within a split (never across splits, since splits are repo-disjoint). Observed reuse: mean ≈ 4.8× in train (max 79, 881 distinct atomics used), mean ≈ 5.0× in test (max 47, 208 distinct atomics used).
5. **Token budget enforcement**: combined tangled-commit diffs exceeding 12,288 cl100k tokens are rejected and re-sampled.
6. **Duplicate prevention**: SHA-set (frozenset) combinations are tracked and rejected if repeated, both within and across splits.
7. **No manual exclusion list**: no manual quality-exclusion filtering is applied — the only filter is the token budget above.

## Use Cases

1. **Commit Message Generation**: Generate appropriate commit messages for code changes
2. **Concern Classification**: Classify the type of concern addressed in a commit
3. **Commit Decomposition**: Break down multi-concern commits into individual concerns
4. **Code Change Analysis**: Understand the relationship between code changes and their descriptions

## Data Collection and Processing

The dataset is created through a three-stage pipeline (`datasets/scripts/`):

### Stage 1: Repo-Grouped Atomic Pool (`build_repo_pool.py`)

1. **Source**: [CCS Dataset](https://huggingface.co/datasets/0x404/ccs_dataset) (2,000 commits)
2. **Repo identity**: extracted from `commit_url` via regex, carried through the entire pipeline
3. **Taxonomy remap**: `style`/`perf` → `refactor`; `chore` dropped; only the 7 CCS types kept
4. **Token filtering**: removed commits where diff + masked message exceeds 12,288 cl100k tokens
5. **Repo eligibility**: kept only repos with ≥5 distinct concern types
6. **Output**: `data/repo_grouped_pool.csv` (1,309 commits, 36 repos)

### Stage 2: Repo-Grouped Tangled Generation (`generate_repo_tangled.py`)

1. **Repo partition**: greedy, type-supply-balanced, repo-disjoint train/test split (`camunda/zeebe` pinned to train); saved to `data/repo_split.json`
2. **Intra-repo tangling**: adaptive-randomized generation of 1-5-concern tangled commits per split, all atomics from a single repo, reuse-weighted atomic selection
3. **Token enforcement**: rejected combinations exceeding 12,288 tokens
4. **Duplicate prevention**: unique SHA-set combinations, tracked with frozensets
5. **Exact uniformity assertion**: fails loudly if per-type label counts are not exactly equal within a split
6. **Output**:
   - 1,400 training examples in `tangled_ccs_dataset_train.csv` (previous file backed up to `data/legacy/` first)
   - 350 test examples in `tangled_ccs_dataset_test.csv` (previous file backed up to `data/legacy/` first)

### Stage 3: Validation (`validate_repo_dataset.py`)

Independently verifies, on the Stage 2 outputs: repo consistency of every row's SHAs, repo-disjointness of the split, exact per-type uniformity (0.00pp deviation, 0.00pp train/test gap), exact row counts per concern count with no duplicate SHA sets, and the real combined-diff token budget (re-derived via tiktoken on the joined diff text, independent of the generation-time approximation). Also reports reuse statistics, per-repo row counts, and saves distribution charts to `results/analysis/repo_dataset_validation/`.

### Data Quality Measures

- All commit messages have sensitive information masked
- Diffs are validated for token limits to ensure model compatibility (checked both at generation time and independently re-verified at validation time)
- Train/test split is repo-disjoint, eliminating both data leakage and the cross-repo "style leakage" that made earlier cross-repo tangles trivially separable
- Exact (not merely approximate) uniform representation across all concern types in both splits

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{tangled_commits_dataset,
  title={Detecting Multiple Semantic Concerns in Tangled Code Commits using Small Language Models},
  author={Beromsu Koh},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs},
  note={Dataset includes 1,309 atomic commits (36 repos) and 1,750 intra-repo tangled multi-concern commits (1,400 train / 21 repos, 350 test / 15 repos), with exact per-type label uniformity in both splits}
}
```

## Scripts

- **`build_repo_pool.py`** (Stage 1): Builds the repo-grouped atomic commit pool from the raw CCS dataset

  - Reconstructs `repo` identity from `commit_url`
  - Applies the updated taxonomy (`style`/`perf` → `refactor`; `chore` dropped) and the 12,288-token filter
  - Keeps only repos with ≥5 distinct types
  - Produces `repo_grouped_pool.csv`

- **`generate_repo_tangled.py`** (Stage 2): Generates intra-repo tangled commits with exact per-type uniformity

  - Greedy repo-disjoint train/test partition, saved to `repo_split.json`
  - Adaptive seeded randomization (`SEED=42`) with reuse-weighted atomic selection
  - Enforces the token budget and rejects duplicate SHA-set combinations
  - Asserts exact per-type label uniformity before writing output
  - Backs up any existing train/test CSVs to `data/legacy/` before overwriting
  - Produces 1,400 train and 350 test examples

- **`validate_repo_dataset.py`** (Stage 3): Independently validates the Stage 2 outputs

  - Verifies repo consistency, repo-disjointness, exact per-type uniformity, row counts, duplicate-free SHA sets, and the real token budget
  - Reports reuse statistics and per-repo row counts
  - Saves distribution charts to `results/analysis/repo_dataset_validation/`
