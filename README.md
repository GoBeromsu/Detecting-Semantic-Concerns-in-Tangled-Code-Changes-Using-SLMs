# Detecting Multiple Semantic Concerns in Tangled Code Commits using Small Language Models

**Author:** Beomsu Koh  
**Institution:** University of Sheffield  
**Project Type:** MSc Computer Science Dissertation  
**Dataset:** [Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs](https://huggingface.co/datasets/Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs)

## Overview

This repository contains the complete implementation and analysis for detecting semantic concerns in tangled code changes using Small Language Models (SLMs). The project investigates how fine-tuned SLMs can identify and separate different types of concerns (e.g., fixes, features, refactoring) in multi-concern commits.

## Project Structure

### 📊 Directory Tree

```text
├── datasets/              # Dataset creation and processing
│   ├── data/             # Raw and processed datasets
│   │   ├── CCS Dataset.csv
│   │   ├── sampled_ccs_dataset.csv
│   │   ├── tangled_ccs_dataset_train.csv
│   │   └── tangled_ccs_dataset_test.csv
│   └── scripts/          # Dataset generation scripts
│       ├── sample_atomic_commites.py
│       ├── generate_tangled_commites.py
│       └── upload_to_huggingface.py
│
├── RQ/                    # Research Questions - Models and Analysis
│   ├── GPT/              # GPT-4.1 inference
│   │   └── infer.py
│   ├── SLM/              # Small Language Models (Qwen3-14B)
│   │   ├── configs/      # Model and training configurations
│   │   ├── train.py      # LoRA fine-tuning script
│   │   ├── infer.py      # Inference script
│   │   └── convert_to_gguf.py
│   ├── analysis/         # Analysis scripts for all RQs
│   │   ├── RQ 1/         # Overall Performance Analysis
│   │   │   ├── config.yaml
│   │   │   ├── performance_summary.py
│   │   │   └── run.py
│   │   ├── RQ 2/         # Performance Analysis (detailed)
│   │   │   ├── config.yaml
│   │   │   ├── msg_impact_analysis.py
│   │   │   ├── concerncount-by-model.py
│   │   │   ├── context_length_performance.py
│   │   │   ├── model_comparison_analysis.py
│   │   │   ├── concern_count_boxplot.py
│   │   │   ├── context_length_boxplot.py
│   │   │   └── run.py
│   │   └── RQ 3/         # Efficiency Analysis
│   │       ├── config.yaml
│   │       ├── efficiency_commit_message.py
│   │       ├── efficiency_concern_count.py
│   │       ├── efficiency_input_tokens.py
│   │       ├── efficiency_concern_count_input_token.py
│   │       └── run.py
│   └── main.py           # Main entry point
│
├── results/              # Generated outputs (organized by RQ)
│   ├── analysis/
│   │   ├── RQ1/         # pf_* (performance) folders
│   │   ├── RQ2/         # pf_* (performance) folders
│   │   └── RQ3/         # ef_* (efficiency) folders
│   ├── gpt/             # GPT-4.1 inference results
│   ├── Qwen/            # Qwen3-14B inference results
│   └── Qwen3-14B-LoRA/  # Fine-tuned model results
│
├── visual_eval/          # Interactive Streamlit dashboard
│   ├── app.py
│   ├── components.py
│   ├── dataset.py
│   └── session.py
│
├── scripts/              # HPC deployment (University of Sheffield)
│   ├── setup_env.sh
│   ├── run_training.sh
│   ├── run_infer_huggingface.sh
│   └── run_gguf_conversion.sh
│
└── utils/                # Shared utilities
    ├── eval.py           # Evaluation metrics
    ├── prompt.py         # Prompt templates
    ├── model.py          # Model definitions
    └── llms/             # LLM API connectors
        ├── openai.py
        ├── hugging_face.py
        └── lmstudio.py
```

### 🗂️ Key Components

#### `datasets/`

Dataset creation pipeline for HuggingFace:

- **`sample_atomic_commites.py`**: Sample single-concern commits from CCS Dataset
- **`generate_tangled_commites.py`**: Create multi-concern commits by combining atomic commits
- **`upload_to_huggingface.py`**: Upload processed dataset to HuggingFace Hub

**Dataset Format:**

- Train: `tangled_ccs_dataset_train.csv` (70% split)
- Test: `tangled_ccs_dataset_test.csv` (30% split)

#### `RQ/` (Research Questions)

##### **RQ1: Overall Performance Analysis**

Evaluates model performance across different configurations:

- **`pf_summary/`**: Comprehensive performance comparison (GPT-4.1, Qwen, Fine-tuned Qwen)

##### **RQ2: Performance Analysis (Detailed)**

In-depth analysis of factors affecting model performance:

- **`pf_msg_impact/`**: Impact of commit messages on performance
- **`pf_concern_count/`**: Performance by number of concerns (1-5)
- **`pf_context_length/`**: Effect of context window size (1024-12288 tokens)
- **`pf_model_comparison/`**: Head-to-head model comparison with failure analysis
- **`pf_concern_count_boxplot/`**: Box plot visualization by concern count
- **`pf_context_length_boxplot/`**: Box plot visualization by context length

##### **RQ3: Efficiency Analysis**

Computational efficiency and inference time analysis:

- **`ef_commit_message/`**: Correlation between commit message presence and inference time
- **`ef_concern_count/`**: Correlation between concern count and inference time
- **`ef_input_tokens/`**: Correlation between input tokens and inference time
- **`ef_concern_count_input_tokens/`**: Multiple regression analysis (concern count + input tokens)

##### **Model Implementation**

- **`GPT/`**: OpenAI GPT-4.1 inference pipeline (zero-shot)
- **`SLM/`**: Qwen3-14B training and inference
  - LoRA fine-tuning with configurable hyperparameters
  - DeepSpeed integration for efficient training
  - GGUF conversion for deployment

#### `results/` Output Structure

##### **Naming Convention**

```
{factor_abbr}_{analysis_target}
```

**Factor Abbreviations:**

- `pf` = performance (RQ1, RQ2)
- `ef` = efficiency (RQ3)

**Benefits:**

- Alphabetical sorting automatically groups by factor
- Same script outputs to the same folder (overwrite mode)
- Clear, predictable naming pattern

##### **Analysis Outputs**

Each analysis folder contains:

- CSV files with raw results
- PNG visualizations (box plots, scatter plots, regression lines)
- JSON summaries with statistical analysis

#### `visual_eval/`

Interactive Streamlit dashboard for:

- Real-time model evaluation on custom inputs
- Results visualization across all RQs
- Dataset exploration and statistics
- Model performance comparison

#### `scripts/`

HPC deployment scripts for University of Sheffield's Stanage cluster:

- **`setup_env.sh`**: Conda environment setup
- **`run_training.sh`**: Submit LoRA fine-tuning jobs
- **`run_infer_huggingface.sh`**: Execute inference on trained models
- **`run_gguf_conversion.sh`**: Convert models to GGUF format for llama.cpp

#### `utils/`

Shared utilities across the project:

- **`eval.py`**: Evaluation metrics (Hamming Loss, F1, Precision, Recall)
- **`prompt.py`**: Prompt templates for zero-shot and few-shot learning
- **`model.py`**: Data models and type definitions
- **`llms/`**: Unified API connectors for different LLM providers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
