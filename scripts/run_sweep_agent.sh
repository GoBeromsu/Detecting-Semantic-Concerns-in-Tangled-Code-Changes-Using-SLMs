#!/bin/bash
#SBATCH --job-name=sweep-agent
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-3  # Run 3 agents in parallel
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

SWEEP_ID=${1:-"gobeumsu-university-of-sheffield/slm-concern-detection-qwen/pqyterlh"}

# Load required modules
module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05
module load CMake/3.26.3-GCCcore-12.3.0

# Environment variables
export FASTDATA_BASE="/mnt/parscratch/users/$USER"

# Return to original directory
cd "$SLURM_SUBMIT_DIR"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '^#' | xargs)
fi

mkdir -p logs

# Activate Python environment
source activate venv
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Login to wandb
echo "Logging in to wandb..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo $WANDB_API_KEY | wandb login
    echo "Successfully logged in to wandb"
fi

# Start GPU monitoring
GPU_LOG="logs/gpu_usage_agent_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
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

# Run wandb agent
echo "Starting wandb agent (Task ID: $SLURM_ARRAY_TASK_ID) at $(date)"
cd RQ/SLM
wandb agent $SWEEP_ID

echo "Sweep agent completed at $(date)"
source deactivate