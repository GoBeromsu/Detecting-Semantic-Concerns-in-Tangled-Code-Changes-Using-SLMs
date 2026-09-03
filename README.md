# Detecting Multiple Semantic Concerns in Tangled Code Commits

Replication package for the preprint **[Detecting Multiple Semantic Concerns in Tangled Code Commits](https://arxiv.org/abs/2601.21298)** (arXiv:2601.21298, cs.SE).

| | |
|---|---|
| **Paper** | [arXiv:2601.21298](https://arxiv.org/abs/2601.21298) — preprint; extended version currently under peer review |
| **Authors** | Beomsu Koh, Neil Walkinshaw, Donghwan Shin |
| **Affiliation** | The main part of this work was carried out at the **University of Sheffield** |
| **Dataset** | [`Berom0227/tangled-ccs-commits`](https://huggingface.co/datasets/Berom0227/tangled-ccs-commits) |
| **Model** | [`Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter`](https://huggingface.co/Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter) (LoRA adapter) |
| **License** | MIT |

Until the paper completes peer review, please cite the arXiv version ([citation](#citation)). This repository tracks the **revised manuscript**, whose primary SLM is Qwen3.6-27B; the arXiv v1 preprint reports the earlier Qwen3-14B configuration, which is kept here unchanged for provenance.

## Overview

Commits should be atomic, but developers routinely bundle several concerns into a single tangled commit. This repository frames multi-concern detection as a **multi-label classification** problem over the Conventional Commits Specification (CCS) taxonomy, builds a controlled dataset of artificially tangled commits from real-world data, and evaluates Small Language Models (SLMs) against a GPT-4.1 baseline.

The package contains everything needed to reproduce the study end to end:

1. **Dataset construction** — a 3-stage, seeded, repo-disjoint tangling pipeline (`datasets/`)
2. **Inference** — GPT-4.1 zero-shot and Qwen zero-shot / LoRA fine-tuned runs (`RQ/GPT/`, `RQ/SLM/`)
3. **Analysis** — one dispatcher covering RQ1–RQ4 with statistics and figures (`RQ/analysis/`)
4. **Exploration** — a Streamlit dashboard over the raw prediction CSVs (`app.py`, `visual_eval/`)

### Two model paths in this repository

| Path | Model | Status | Results |
|------|-------|--------|---------|
| **Current manuscript** | Qwen3.6-27B + LoRA (BF16, Unsloth), trained on a single local workstation GPU | Active — `RQ/analysis/config.yaml` points here, and every table/figure in the revision comes from it | `results/gpt-seed43/`, `results/Qwen3.6-27B/`, `results/Qwen3.6-27B-LoRA/` |
| **arXiv v1** | Qwen3-14B + LoRA, trained on Sheffield's Stanage HPC (A100/H100) | Frozen — scripts are hash-locked and must stay byte-identical | `results/gpt/`, `results/Qwen/`, `results/Qwen3-14B-LoRA/` |

`results/analysis/` is regenerated from whichever arm `RQ/analysis/config.yaml` selects; `supplementary/` mirrors the tables reported in the current manuscript. Re-point the `models:` block at the 14B result directories to reproduce the arXiv v1 figures.

## Quickstart

```bash
uv sync                             # base deps  (extras: --extra dev|hpc|local-gpu)
uv run pytest __test__/ -q          # CPU-only test suite (same as CI)
python RQ/analysis/run.py --list    # show every analysis script
python RQ/analysis/run.py --all     # regenerate results/analysis/
streamlit run app.py                # interactive result explorer
```

Python is pinned to 3.12 (`.python-version`); the package floor is `>=3.10`.
Environment variables: `OPENAI_API_KEY` (GPT inference), `HF_HUB_TOKEN` (HuggingFace), `WANDB_API_KEY` (27B training runs).

## Repository structure

```text
├── datasets/                       # 3-stage tangled-commit pipeline (see datasets/AGENTS.md)
│   ├── data/                       # CCS Dataset.csv, repo_grouped_pool.csv, repo_split.json,
│   │                               # tangled_ccs_dataset_{train,test}.csv, legacy/
│   └── scripts/
│       ├── build_repo_pool.py            # Stage 1: CCS Dataset.csv -> repo_grouped_pool.csv
│       ├── generate_repo_tangled.py      # Stage 2: pool -> train/test CSVs + repo_split.json
│       ├── validate_repo_dataset.py      # Stage 3: PASS/FAIL validation report
│       ├── upload_to_huggingface.py      # publish validated CSVs to HuggingFace
│       ├── structural_validity/          # diff-level colocation / structural-validity checks
│       └── analyze_*.py, plot_*.py       # diagnostics, not part of the build chain
│
├── RQ/
│   ├── GPT/infer.py                # GPT-4.1 zero-shot inference -> results/gpt*/
│   ├── SLM/                        # two independent model paths (see RQ/SLM/AGENTS.md)
│   │   ├── train.py infer.py convert_to_gguf.py   # frozen arXiv v1 14B path
│   │   ├── configs/                # legacy model configs + hosts/ profile
│   │   └── unsloth/                # Qwen3.6-27B BF16 LoRA package + RUNBOOK.md
│   ├── analysis/                   # unified RQ1-4 dispatcher (see RQ/analysis/AGENTS.md)
│   │   ├── config.yaml             # single source of truth for every RQ script
│   │   ├── run.py                  # entry point
│   │   ├── RQ1/ RQ2/ RQ3/ RQ4/     # per-RQ scripts
│   │   ├── data_aggregation/       # aggregate/average repeated experiment runs
│   │   └── plot_utils.py stats_utils.py summary_ci.py
│   └── main.py                     # header-preserving token-truncation helper
│
├── utils/                          # eval.py prompt.py model.py llms/{openai,hugging_face,lmstudio}
├── visual_eval/                    # Streamlit dashboard components
├── scripts/
│   ├── hpc/stanage-slurm/          # frozen Sheffield Stanage SLURM scripts (arXiv v1)
│   └── blackwell/                  # local GPU-host session helpers
├── results/                        # generated output only, timestamped, never hand-edited
├── supplementary/                  # extended result tables from the paper
├── __test__/                       # pytest suite (CPU-only), incl. unsloth/ subtree
└── app.py                          # Streamlit entrypoint
```

## Dataset

Built from the Conventional Commits Specification (CCS) dataset by tangling atomic commits **within the same repository** (no cross-repo tangling, which would leak coding style).

- **Train** — `tangled_ccs_dataset_train.csv`: 1400 rows, 280 per concern count, 21 repositories
- **Test** — `tangled_ccs_dataset_test.csv`: 350 rows, 70 per concern count, 15 repositories
- Concern counts 1–5, 7 concern types, per-type uniformity enforced to 0.00pp deviation
- Repo-disjoint 80/20 split, `SEED=43`, token budget 12288 (cl100k)

Reproduce with `build_repo_pool.py` → `generate_repo_tangled.py` → `validate_repo_dataset.py`. Never hand-edit `datasets/data/*.csv`.

## Research questions

| RQ | Question | Signal |
|----|----------|--------|
| **RQ1** | How does performance degrade as concern count grows? | `hamming_loss` × `concern_count` |
| **RQ2** | Do commit messages provide useful semantic cues? | `hamming_loss` × `with_message` |
| **RQ3** | How robust are models to header-preserving truncation? | `hamming_loss` × `context_len` (1024–12288) |
| **RQ4** | What drives inference latency? | `inference_time` × count / tokens / message |

RQ1–RQ3 include performance summaries, boxplots, pairwise significance tests (Wilcoxon / Mann-Whitney), and 95% confidence intervals for the manuscript tables; RQ4 reports inference-efficiency analyses. Run them through `RQ/analysis/run.py`; outputs land in `results/analysis/RQ<N>/`.

## Models

- **GPT-4.1** — OpenAI API baseline, zero-shot
- **Qwen3.6-27B** — base SLM, zero-shot
- **Qwen3.6-27B-LoRA** — fine-tuned SLM reported in the current manuscript (rank=32, alpha=48, dropout=0.05, BF16, unmerged PEFT adapter), released as [`Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter`](https://huggingface.co/Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter)
- **Qwen3-14B / Qwen3-14B-LoRA** — the arXiv v1 arm (rank=32, alpha=48), retained unchanged

### Qwen3.6-27B (current)

Full flags, memory-qualification ladder, and evidence requirements are in [`RQ/SLM/README.md`](RQ/SLM/README.md). Highlights:

- `Qwen/Qwen3.6-27B` pinned at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, text tower only, configured by `RQ/SLM/unsloth/configs/qwen3_6_27b.yml`
- BF16 end to end — 4-bit, 8-bit, `adamw_8bit`, `flash_attention_2`, left padding, packing, and `device_map: auto` are rejected by config validation
- Response-only supervision (loss masked to the assistant turn), recorded as `objective: response_only_json_eos`
- Training refuses to start until `python -m RQ.SLM.unsloth.memory` records `approved_16384` evidence whose config, host-profile, and measurement hashes still match on disk
- Dataset pinned to `Berom0227/tangled-ccs-commits` @ `65b09af76f3e9badf4a28bf7a641b1d2930a26b5`; at `max_seq_length=16384` no row overflows, so all 1400 train rows are retained

```bash
uv sync --frozen --python 3.12 --extra local-gpu
python -m RQ.SLM.unsloth.train --config RQ/SLM/unsloth/configs/qwen3_6_27b.yml
python -m RQ.SLM.unsloth.infer --adapter outputs/unsloth/<run>/adapter \
  --config RQ/SLM/unsloth/configs/qwen3_6_27b.yml --output results
```

### Qwen3-14B (arXiv v1)

```bash
python RQ/SLM/train.py --config RQ/SLM/configs/qwen.yml
python RQ/SLM/infer.py
python RQ/SLM/convert_to_gguf.py --model qwen
```

These three scripts and the Stanage SLURM job scripts under `scripts/hpc/stanage-slurm/` are hash-locked in `__test__/fixtures/slm/protected-files.sha256` so the earlier pipeline stays byte-identical.

## Supplementary results

Mean Hamming Loss with 95% confidence intervals, as reported in the current manuscript. Machine-readable copies live in `supplementary/`; their generated sources live under `results/analysis/RQ1/`, `RQ2/`, and `RQ3/`.

### By concern count (RQ1)

| Count | GPT-4.1 | Qwen3.6 | Qwen3.6-FT |
|-------|---------|---------|------------|
| 1 | 0.10 [0.07, 0.13] | 0.09 [0.06, 0.13] | 0.04 [0.02, 0.06] |
| 2 | 0.16 [0.12, 0.20] | 0.18 [0.13, 0.23] | 0.12 [0.08, 0.16] |
| 3 | 0.20 [0.16, 0.25] | 0.26 [0.20, 0.31] | 0.14 [0.10, 0.18] |
| 4 | 0.21 [0.17, 0.25] | 0.24 [0.20, 0.28] | 0.14 [0.10, 0.18] |
| 5 | 0.14 [0.11, 0.17] | 0.16 [0.13, 0.19] | 0.06 [0.03, 0.09] |
| **All** | **0.16 [0.15, 0.18]** | **0.19 [0.17, 0.21]** | **0.10 [0.09, 0.12]** |

### By commit-message inclusion (RQ2)

| Condition | GPT-4.1 | Qwen3.6 | Qwen3.6-FT |
|-----------|---------|---------|------------|
| Without Msg | 0.18 [0.17, 0.20] | 0.21 [0.19, 0.22] | 0.15 [0.13, 0.17] |
| With Msg | 0.16 [0.15, 0.18] | 0.19 [0.17, 0.21] | 0.10 [0.09, 0.12] |

For Qwen3.6-FT, including the commit message reduces Hamming Loss from 0.15 to 0.10 — a 31% relative reduction, the largest effect of the three models.

### By token budget (RQ3)

| Token budget | GPT-4.1 | Qwen3.6 | Qwen3.6-FT |
|--------------|---------|---------|------------|
| 1024 | 0.15 [0.14, 0.17] | 0.17 [0.15, 0.19] | 0.12 [0.10, 0.14] |
| 2048 | 0.16 [0.14, 0.17] | 0.18 [0.16, 0.19] | 0.11 [0.10, 0.13] |
| 4096 | 0.16 [0.14, 0.17] | 0.19 [0.17, 0.21] | 0.11 [0.09, 0.13] |
| 8192 | 0.16 [0.14, 0.17] | 0.19 [0.17, 0.21] | 0.10 [0.08, 0.12] |
| 12288 | 0.16 [0.15, 0.18] | 0.19 [0.17, 0.21] | 0.10 [0.09, 0.12] |

## Citation

```bibtex
@misc{koh2026detecting,
  title         = {Detecting Multiple Semantic Concerns in Tangled Code Commits},
  author        = {Koh, Beomsu and Walkinshaw, Neil and Shin, Donghwan},
  year          = {2026},
  eprint        = {2601.21298},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  url           = {https://arxiv.org/abs/2601.21298}
}
```

## License

MIT — see [LICENSE](LICENSE).
