# Detecting Multiple Semantic Concerns in Tangled Code Commits using Small Language Models

**Author:** Beomsu Koh
**Institution:** University of Sheffield
**Project Type:** MSc Computer Science Dissertation
**Dataset:** [Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs](https://huggingface.co/datasets/Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs)

## Overview

This repository contains the complete implementation and analysis for detecting semantic concerns in tangled code changes using Small Language Models (SLMs). The project investigates how fine-tuned SLMs can identify and separate different types of concerns (e.g., fixes, features, refactoring) in multi-concern commits.

## Project Structure

```text
├── datasets/                    # Dataset creation and processing
│   ├── data/                   # Raw and processed datasets
│   │   ├── CCS Dataset.csv
│   │   ├── repo_grouped_pool.csv
│   │   ├── repo_split.json
│   │   ├── tangled_ccs_dataset_train.csv
│   │   ├── tangled_ccs_dataset_test.csv
│   │   └── legacy/              # Backups of superseded cross-repo tangled CSVs
│   └── scripts/                # Dataset generation scripts (repo-grouped pipeline)
│       ├── build_repo_pool.py       # Stage 1: CCS Dataset.csv -> repo_grouped_pool.csv
│       ├── generate_repo_tangled.py # Stage 2: repo_grouped_pool.csv -> tangled_ccs_dataset_{train,test}.csv
│       ├── validate_repo_dataset.py # Stage 3: validates Stage 2 outputs
│       ├── analyze_token_distribution.py
│       ├── concern_token_boxplot.py
│       └── upload_to_huggingface.py
│
├── RQ/                          # Research Questions - Models and Analysis
│   ├── GPT/                    # GPT-4.1 inference pipeline
│   ├── SLM/                    # Small Language Models (see RQ/SLM/README.md)
│   │   ├── configs/            # Legacy configs + local host profile
│   │   │   └── hosts/          # Recorded facts for the local GPU host
│   │   ├── unsloth/            # Current Qwen3.6-27B BF16 LoRA package
│   │   │   ├── configs/qwen3_6_27b.yml
│   │   │   ├── {config,data,runtime,train,infer,memory,preflight}.py
│   │   │   └── {adapter,results,generation,infer_options}.py
│   │   ├── train.py            # Legacy: Qwen3-14B LoRA fine-tuning
│   │   ├── infer.py            # Legacy: inference script
│   │   └── convert_to_gguf.py  # Legacy: GGUF conversion for deployment
│   ├── analysis/               # Unified analysis scripts
│   │   ├── config.yaml         # Single source of truth for all RQs
│   │   ├── run.py              # Main analysis runner
│   │   ├── RQ1/                # Impact of Concern Count
│   │   ├── RQ2/                # Impact of Commit Message
│   │   ├── RQ3/                # Token-Budget Robustness
│   │   └── RQ4/                # Inference Efficiency
│   └── main.py
│
├── results/                     # Generated outputs only, never hand-edited
│   ├── analysis/               # Analysis results by RQ
│   │   ├── RQ1/
│   │   ├── RQ2/
│   │   ├── RQ3/
│   │   └── RQ4/
│   ├── gpt/                    # GPT-4.1 inference results
│   ├── Qwen/                   # Qwen3-14B inference results
│   ├── Qwen3-14B-LoRA/         # Fine-tuned Qwen3-14B results (paper)
│   └── Qwen3.6-27B-LoRA/       # Fine-tuned Qwen3.6-27B results (timestamped runs)
│
├── visual_eval/                 # Interactive Streamlit dashboard
│   ├── components.py
│   ├── dataset.py
│   ├── session.py
│   └── setup.py
│
├── scripts/                     # Deployment scripts
│   └── hpc/stanage-slurm/            # Frozen Sheffield Stanage SLURM scripts (14B paper results)
│       ├── setup_env.sh                  # HPC (Stanage) environment setup
│       ├── run_training.sh               # HPC legacy training
│       ├── run_lora_pipeline.sh
│       ├── run_infer_huggingface.sh
│       └── run_gguf_conversion.sh
│
├── utils/                       # Shared utilities
│   ├── eval.py                 # Evaluation metrics
│   ├── prompt.py               # Prompt templates
│   ├── model.py                # Data models
│   └── llms/                   # LLM API connectors
│       ├── openai.py
│       ├── hugging_face.py
│       ├── lmstudio.py
│       └── constant.py
│
├── __test__/                    # Test suite
│   ├── test_api.py
│   └── test_eval.py
│
└── app.py                       # Main Streamlit application
```

## Research Questions

### RQ1: Impact of Concern Count

Evaluates model performance as semantic complexity increases:

- `performance_summary.py`: Performance comparison across models (GPT-4.1, Qwen, Fine-tuned Qwen)
- `concern_count_boxplot.py`: Box plot visualization by concern count
- `concerncount-by-model.py`: Performance comparison by model
- `model_comparison_analysis.py`: Head-to-head model comparison with failure analysis
- `concern_count_pairwise_pvalue.py`: Statistical significance testing

### RQ2: Impact of Commit Message Inclusion

Investigates whether commit messages provide additional semantic cues:

- `msg_impact_analysis.py`: Analyzes performance with/without commit messages
- `msg_impact_pairwise_pvalue.py`: Pairwise statistical comparison

### RQ3: Token-Budget Robustness

Examines model reliability when token budget is reduced (1024-12288 tokens):

- `context_length_performance.py`: Performance across context lengths
- `context_length_boxplot.py`: Box plot visualization by context length
- `context_length_pairwise_pvalue.py`: Statistical significance testing

### RQ4: Inference Efficiency

Analyzes how factors influence inference latency:

- `efficiency_commit_message.py`: Correlation with commit message presence
- `efficiency_concern_count.py`: Correlation with concern count
- `efficiency_input_tokens.py`: Correlation with input tokens
- `efficiency_concern_count_input_token.py`: Multiple regression analysis

## Key Components

### Models

- **GPT-4.1**: OpenAI API baseline (zero-shot)
- **Qwen3-14B**: Base SLM for comparison
- **Qwen3-14B-LoRA**: Fine-tuned SLM with LoRA (rank=32, alpha=48)
- **Qwen3.6-27B-LoRA**: Current fine-tuning target, BF16 LoRA with an unmerged PEFT adapter (rank=32, alpha=48, dropout=0.05)

The reported paper results come from the legacy Qwen3-14B path (`train.py` -> `infer.py` -> `convert_to_gguf.py`) on Sheffield Stanage SLURM. That path is unchanged. The Qwen3.6-27B path is separate: it trains through Unsloth, evaluates through Transformers and PEFT, and produces no merged model and no GGUF. The Stanage A100/H100 allocation is no longer accessible, so the 27B work targets a single local Blackwell workstation GPU instead and HPC execution is out of scope for it.

### Dataset

- **Train**: `tangled_ccs_dataset_train.csv` (80% split, 1400 validated rows)
- **Test**: `tangled_ccs_dataset_test.csv` (20% split, 350 rows)
- Based on Conventional Commits Specification (CCS)
- The 27B path pins `Berom0227/tangled-ccs-commits` at revision `234e8cb034bace7f6fa2a87e73e8c86bc0b04a7d` and drops the single train row that exceeds 16384 tokens, leaving **1399 retained rows**

### Qwen3.6-27B workflow

Full instructions, exact flags, and evidence requirements live in [`RQ/SLM/README.md`](RQ/SLM/README.md). In short:

- Model `Qwen/Qwen3.6-27B` pinned at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, text tower only, configured by `RQ/SLM/unsloth/configs/qwen3_6_27b.yml`.
- BF16 end to end. No quantization: 4-bit, 8-bit, and `adamw_8bit` are rejected by config validation, as are `flash_attention_2`, left padding, packing, and `device_map: auto`.
- Response-only supervision: loss is masked to the assistant turn, recorded in the manifest as `objective: response_only_json_eos`.
- Inference requires a real pad token that differs from the EOS token; if they match, the run stops rather than padding with EOS.

```bash
uv sync --frozen --python 3.12 --extra local-gpu
uv run pytest __test__/ -q
```

Phase A (CPU) is complete and covered by the test suite. Phase B is not done: loading the model, memory qualification, and training all allocate the entire GPU on a workstation that also drives a display. Full training refuses to start until `python -m RQ.SLM.unsloth.memory` records `approved_16384` qualification evidence whose config, host profile, and measurement hashes still match on disk.

`results/` contains generated output only. Regenerate it by re-running the step that produced it rather than editing files by hand.

### Utilities

- **`eval.py`**: Evaluation metrics (Hamming Loss, F1, Precision, Recall)
- **`prompt.py`**: Prompt templates for zero-shot and few-shot learning
- **`llms/`**: Unified API connectors for OpenAI, HuggingFace, and LM Studio

## Supplementary Materials

Extended result tables from the paper are available in the `supplementary/` directory.

### Mean Hamming Loss by Concern Count (RQ1)

| Count | GPT-4.1 | Qwen3 | Qwen3-FT |
|-------|---------|-------|----------|
| 1 | 0.07 | 0.11 | 0.04 |
| 2 | 0.09 | 0.23 | 0.13 |
| 3 | 0.09 | 0.33 | 0.15 |
| 4 | 0.10 | 0.33 | 0.20 |
| 5 | 0.12 | 0.27 | 0.17 |

### Mean Hamming Loss by Commit Message Inclusion (RQ2)

| Condition | GPT-4.1 | Qwen3 | Qwen3-FT |
|-----------|---------|-------|----------|
| Without Msg | 0.11 | 0.28 | 0.25 |
| With Msg | 0.09 | 0.25 | 0.14 |
| Delta | 0.02 | 0.03 | 0.11 |

> **Note**: For Qwen3-FT, commit message inclusion reduces Hamming Loss by 44% ((0.25 - 0.14) / 0.25 = 0.44).

### Mean Hamming Loss by Input Token Length (RQ3)

| Token Length | GPT-4.1 | Qwen3 | Qwen3-FT |
|--------------|---------|-------|----------|
| 1024 | 0.10 | 0.26 | 0.15 |
| 2048 | 0.10 | 0.26 | 0.15 |
| 4096 | 0.10 | 0.25 | 0.15 |
| 8192 | 0.10 | 0.25 | 0.14 |
| 12288 | 0.09 | 0.26 | 0.14 |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
