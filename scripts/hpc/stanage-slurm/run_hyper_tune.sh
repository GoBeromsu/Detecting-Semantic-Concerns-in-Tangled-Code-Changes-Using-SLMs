#!/bin/bash
#SBATCH --job-name=hyper-tune
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-5  # Run 5 agents simultaneously
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

MODEL_TYPE=${1:-"Phi"}
[[ "$MODEL_TYPE" =~ ^(Phi|Qwen)$ ]] || { echo "MODEL_TYPE must be Phi or Qwen"; exit 1; }

echo "MODEL_TYPE=${MODEL_TYPE}"
echo "Starting hyperparameter tuning job: $SLURM_JOB_ID"

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
    echo "[ERROR] llama.cpp not found at $LLAMA_CPP_DIR"
    exit 1
fi
echo "[INFO] llama.cpp found."

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
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Start GPU monitoring
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}_${MODEL_TYPE}_sweep.csv"
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

# Create wandb sweep and run agent
echo "Creating wandb sweep for ${MODEL_TYPE} at $(date)"
cd RQ/SLM

# Login to wandb using API key from .env
echo "Logging in to wandb..."
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo $WANDB_API_KEY | wandb login
    echo "Successfully logged in to wandb"
else
    echo "ERROR: WANDB_API_KEY not found in .env file"
    exit 1
fi

# Create sweep and automatically start agent
echo "Creating sweep configuration..."
if [[ "$MODEL_TYPE" == "Phi" ]]; then
    wandb sweep configs/sweep.yaml --project "slm-concern-detection-phi" | tee sweep_output.log
    SWEEP_ID=$(grep "wandb agent" sweep_output.log | awk '{print $NF}')
elif [[ "$MODEL_TYPE" == "Qwen" ]]; then
    wandb sweep configs/sweep.yaml --project "slm-concern-detection-qwen" | tee sweep_output.log
    SWEEP_ID=$(grep "wandb agent" sweep_output.log | awk '{print $NF}')
else
    echo "Unsupported MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

echo ""
echo "🚀 Starting hyperparameter optimization..."
echo "Sweep ID: $SWEEP_ID"
echo "Starting agent at $(date)"

# Run sweep agent automatically
wandb agent $SWEEP_ID

echo "Hyperparameter tuning completed at $(date)"

# Clean up
rm -f sweep_output.log

# Display basic job info
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,State,ExitCode
source deactivate