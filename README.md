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
│   │   ├── sampled_ccs_dataset.csv
│   │   ├── tangled_ccs_dataset_train.csv
│   │   └── tangled_ccs_dataset_test.csv
│   └── scripts/                # Dataset generation scripts
│       ├── sample_atomic_commites.py
│       ├── generate_tangled_commites.py
│       ├── analyze_token_distribution.py
│       ├── concern_token_boxplot.py
│       └── upload_to_huggingface.py
│
├── RQ/                          # Research Questions - Models and Analysis
│   ├── GPT/                    # GPT-4.1 inference pipeline
│   ├── SLM/                    # Small Language Models (Qwen3-14B)
│   │   ├── configs/            # Model and training configurations
│   │   ├── train.py            # LoRA fine-tuning script
│   │   ├── infer.py            # Inference script
│   │   └── convert_to_gguf.py  # GGUF conversion for deployment
│   ├── analysis/               # Unified analysis scripts
│   │   ├── config.yaml         # Single source of truth for all RQs
│   │   ├── run.py              # Main analysis runner
│   │   ├── RQ1/                # Impact of Concern Count
│   │   ├── RQ2/                # Impact of Commit Message
│   │   ├── RQ3/                # Token-Budget Robustness
│   │   └── RQ4/                # Inference Efficiency
│   └── main.py
│
├── results/                     # Generated outputs
│   ├── analysis/               # Analysis results by RQ
│   │   ├── RQ1/
│   │   ├── RQ2/
│   │   ├── RQ3/
│   │   └── RQ4/
│   ├── gpt/                    # GPT-4.1 inference results
│   ├── Qwen/                   # Qwen3-14B inference results
│   └── Qwen3-14B-LoRA/         # Fine-tuned model results
│
├── visual_eval/                 # Interactive Streamlit dashboard
│   ├── components.py
│   ├── dataset.py
│   ├── session.py
│   └── setup.py
│
├── scripts/                     # HPC deployment scripts
│   ├── setup_env.sh
│   ├── run_training.sh
│   ├── run_lora_pipeline.sh
│   ├── run_infer_huggingface.sh
│   └── run_gguf_conversion.sh
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
- **Qwen3-14B-LoRA**: Fine-tuned SLM with LoRA (rank=64, alpha=128)

### Dataset

- **Train**: `tangled_ccs_dataset_train.csv` (70% split)
- **Test**: `tangled_ccs_dataset_test.csv` (30% split)
- Based on Conventional Commits Specification (CCS)

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
