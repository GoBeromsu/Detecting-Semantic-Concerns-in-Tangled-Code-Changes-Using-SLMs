#!/bin/bash
#SBATCH --job-name=phi4_commit_sft
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/phi4_training_%j.out
#SBATCH --error=logs/phi4_training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - A100 GPU Training + GGUF Conversion
# Multi-Concern Commit Classification with Phi-4

echo "Starting Phi-4 LoRA fine-tuning job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# Configure cache directories on fastdata to avoid filling home quota
export FASTDATA_BASE="/mnt/parscratch/users/$USER"
export HF_HOME="$FASTDATA_BASE/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_DIR="$FASTDATA_BASE/wandb"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

# Check if llama.cpp exists (should be built by setup_env.sh)
LLAMA_CPP_DIR="$FASTDATA_BASE/llama.cpp"
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp not found at $LLAMA_CPP_DIR"
    echo "Please run setup_env.sh first to build llama.cpp:"
    echo "sbatch scripts/setup_env.sh"
    exit 1
else
    echo "✅ llama.cpp found at $LLAMA_CPP_DIR"
fi

# Return to original directory
cd "$SLURM_SUBMIT_DIR"

# Activate environment using 'source activate' instead of 'conda activate'
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # Multi-GPU communication debugging

# Run training
echo "🔥 Starting training at $(date)"
python -u RQ/Phi/train.py

echo "✅ Training completed at $(date)"

# Display basic job info
echo "📊 Job Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode

###############################################
# Optional cleanup after successful HF upload #
###############################################
# Toggle cleanups (set to "true" to enable)
CLEAN_MODEL_ARTIFACTS=true
CLEAN_HF_MODEL_CACHE=false
CLEAN_WANDB_LOCAL=true

# Paths aligned with train.py
MODEL_NAME="microsoft/phi-4"
TRAIN_DIR="$FASTDATA_BASE/models/${MODEL_NAME}-LoRA"
MERGED_DIR="$FASTDATA_BASE/models/merged_model"

if [ "$CLEAN_MODEL_ARTIFACTS" = true ]; then
  echo "🧹 Removing trained artifacts under $FASTDATA_BASE/models"
  rm -rf "$TRAIN_DIR" "$MERGED_DIR" || true
fi

if [ "$CLEAN_HF_MODEL_CACHE" = true ]; then
  # Remove cached base model shards to free space; disable to keep for next runs
  echo "🧹 Removing Hugging Face model cache for $MODEL_NAME"
  rm -rf "$HUGGINGFACE_HUB_CACHE/models--microsoft--phi-4" || true
fi

if [ "$CLEAN_WANDB_LOCAL" = true ]; then
  echo "🧹 Cleaning W&B local directories"
  # Fastdata WANDB_DIR
  find "$WANDB_DIR" -maxdepth 1 -type d -name 'run-*' -mtime +0 -exec rm -rf {} + 2>/dev/null || true
  # Repo-local wandb (from previous runs using default location)
  REPO_WANDB_DIR="$PWD/wandb"
  if [ -d "$REPO_WANDB_DIR" ]; then
    rm -rf "$REPO_WANDB_DIR" || true
  fi
fi

source deactivate 