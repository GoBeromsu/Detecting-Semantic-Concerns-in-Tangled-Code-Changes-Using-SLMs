#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

MODEL_TYPE=${1:-"Phi"}
[[ "$MODEL_TYPE" =~ ^(Phi|Qwen)$ ]] || { echo "MODEL_TYPE must be Phi or Qwen"; exit 1; }

echo "MODEL_TYPE=${MODEL_TYPE}"
echo "Starting training job: $SLURM_JOB_ID"

# Load required modules
module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05
module load CMake/3.26.3-GCCcore-12.3.0

# Environment variables
export FASTDATA_BASE="/mnt/parscratch/users/$USER"

# Temporary workspace for GGUF conversion (if triggered in train.py)
export TMPDIR="${TMPDIR:-/tmp/gguf_conversion_$$}"

# llama.cpp location (external dependency)
export LLAMA_CPP_DIR="$FASTDATA_BASE/llama.cpp"
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

mkdir -p logs

# Activate Python environment
source activate venv
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # Multi-GPU communication debugging

# Start GPU monitoring
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}_${MODEL_TYPE}.csv"
echo "timestamp,power.draw[W],gpu.util[%],mem.util[%],mem.used[MiB],temp.gpu[C]" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 60
  done
) &
GPU_MON_PID=$!

# Cleanup function
cleanup() {
  kill "$GPU_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Start training
echo "Starting ${MODEL_TYPE} training at $(date)"
if [[ "$MODEL_TYPE" == "Phi" ]]; then
    python -u RQ/SLM/train.py --config RQ/SLM/configs/phi.yml
elif [[ "$MODEL_TYPE" == "Qwen" ]]; then
    python -u RQ/SLM/train.py --config RQ/SLM/configs/qwen.yml
else
    echo "Unsupported MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi
echo "Training completed at $(date)"

# Display basic job info
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode
source deactivate 