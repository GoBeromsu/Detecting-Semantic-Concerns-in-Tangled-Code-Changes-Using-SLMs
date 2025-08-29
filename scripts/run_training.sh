#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

# Sheffield HPC Stanage - A100 GPU Training + GGUF Conversion
# Multi-Concern Commit Classification with Phi-4

echo "Starting training job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE MB"

module purge
module load GCCcore/12.3.0
module load CUDA/12.1.1
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
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

# Ensure logs directory exists
mkdir -p logs

# Activate environment using 'source activate' instead of 'conda activate'
echo "🔧 Activating phi4_env..."
source activate phi4_env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # Multi-GPU communication debugging

# Start periodic GPU utilization logging (every 60 s)
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}.csv"
echo "timestamp,power.draw[W],gpu.util[%],mem.util[%],mem.used[MiB]" > "$GPU_LOG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used --format=csv,noheader >> "$GPU_LOG"
    sleep 60
  done
) &
GPU_MON_PID=$!

# Ensure GPU logger is terminated on script exit
cleanup() {
  kill "$GPU_MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Run training
echo "🔥 Starting training at $(date)"
python -u RQ/Phi/train.py

echo "✅ Training completed at $(date)"

# Display basic job info
echo "📊 Job Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode


source deactivate 