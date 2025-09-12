#!/bin/bash
#SBATCH --job-name=infer
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bkoh3@sheffield.ac.uk

set -euo pipefail

# Convert input to lowercase and use directly
MODEL_INPUT=${1:-"phi"}
MODEL_PARAM=$(echo "$MODEL_INPUT" | tr '[:upper:]' '[:lower:]')
MODEL_TYPE=$(echo "$MODEL_INPUT" | sed 's/.*/\L&/; s/\b\w/\U&/g')

echo "MODEL_TYPE=${MODEL_TYPE}"

# Ensure logs directory exists
mkdir -p logs

module purge
module load GCCcore/12.3.0
module load CUDA/12.4.0
module load Anaconda3/2022.05

# Activate environment using 'source activate' per Stanage rules
echo "🔧 Activating venv..."
source activate venv

# Start periodic GPU utilization logging (every 60 s)
GPU_LOG="logs/gpu_usage_${SLURM_JOB_ID}_${MODEL_TYPE}.csv"
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

echo "🚀 Starting inference at $(date)"

python -u RQ/SLM/infer.py --model "${MODEL_PARAM}" | tee -a "logs/infer_output_${SLURM_JOB_ID}_${MODEL_TYPE}.log"

echo "✅ Inference completed at $(date)"

source deactivate


