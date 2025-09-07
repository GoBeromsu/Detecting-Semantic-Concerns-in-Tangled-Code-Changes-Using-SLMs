# Detecting Semantic Concerns in Tangled Code Changes Using Small Language Models

**Author:** Beomsu Koh  
**Institution:** University of Sheffield  
**Project Type:** MSc Computer Science Dissertation  
**Dataset:** [Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs](https://huggingface.co/datasets/Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs)

## Overview

This repository contains the complete implementation and analysis for detecting semantic concerns in tangled code changes using Small Language Models (SLMs). The project investigates how fine-tuned SLMs can identify and separate different types of concerns (e.g., fixes, features, refactoring) in multi-concern commits.

## Project Structure

### 📊 Core Directories

```text
├── datasets/           # Dataset creation and processing scripts
├── RQ/                # Research Questions - Models and Analysis
│   ├── GPT/           # GPT-4.1 inference scripts
│   ├── Phi/           # Phi-4 model training and inference
│   ├── Qwen/          # Qwen3-14B model training and inference
│   └── analysis/      # Results analysis (RQ1, RQ2)
├── visual_eval/       # Streamlit dashboard for results visualization
├── scripts/           # HPC deployment scripts for University of Sheffield
├── results/           # Generated analysis results and outputs
└── utils/             # Shared utilities (prompts, evaluation, LLM APIs)
```

### 🗂️ Key Components

#### `datasets/`

Scripts and tools for creating the HuggingFace dataset used in this research:

- Sample atomic commits from repositories
- Generate tangled (multi-concern) commits
- Upload processed data to HuggingFace Hub

#### `RQ/` (Research Questions)

**RQ1: Performance Analysis**

- Model comparison (GPT-4.1 vs Phi-4 vs Fine-tuned Phi-4)
- Context length impact analysis
- Message inclusion effects

**RQ2: Efficiency Analysis**

- Inference time correlations
- Computational complexity analysis

**Model Implementation:**

- `GPT/`: OpenAI GPT-4.1 inference pipeline
- `Phi/`: Microsoft Phi-4 training and inference
- `Qwen/`: Qwen3-14B training and inference

#### `visual_eval/`

Interactive Streamlit dashboard for:

- Real-time model evaluation
- Results visualization
- Dataset exploration

#### `scripts/`

HPC deployment scripts for University of Sheffield's Stanage cluster:

- Environment setup
- Training job submission
- Inference execution
- GGUF model conversion


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
